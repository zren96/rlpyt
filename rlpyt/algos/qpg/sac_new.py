
import numpy as np
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.gaussian import Gaussian
from rlpyt.distributions.gaussian import DistInfo as GaussianDistInfo
from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done


OptInfo = namedtuple("OptInfo",
    ["qLoss", "piLoss",
    "qGradNorm", "piGradNorm",
    "q1", "q2", "piMu", "piLogStd", "qMeanDiff", "alpha"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
    SamplesToBuffer._fields + ("timeout",))


class SACNew(RlAlgorithm):
    """Soft actor critic algorithm, training from a replay buffer."""
    # Assume no bootstrap limit; no policy_output_regularization; reparameterize=True; target_entropy='auto'

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.99,
            batch_size=256,
            min_steps_learn=int(1e4),
            replay_size=int(1e6),
            replay_ratio=256,  # data_consumption / data_generation
            target_update_tau=0.005,  # tau=1 for hard update.
            target_update_interval=1,  # 1000 for hard update, 1 for soft.
            actor_update_interval=1,
            initial_alpha=1,
            fixed_alpha=False,
            learning_rate=3e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,  # for all of them.
            initial_replay_buffer_dict=None,
            action_prior="uniform",  # or "gaussian"
            reward_scale=1,
            clip_grad_norm=1e9,
            n_step_return=1,
            updates_per_sync=1,  # For async mode only.
            ReplayBufferCls=None,  # Leave None to select by above options.
            ):
        """Save input arguments."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        assert action_prior in ["uniform", "gaussian"]
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """Stores input arguments and initializes replay buffer and optimizer.
        Use in non-async runners.  Computes number of gradient updates per
        optimization iteration as `(replay_ratio * sampler-batch-size /
        training-batch_size)`."""
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = int(self.replay_ratio * sampler_bs /
            self.batch_size)
        # logger.log(f"From sampler batch size {sampler_bs}, training "
            # f"batch size {self.batch_size}, and replay ratio "
            # f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            # f"updates per iteration.")
        self.min_itr_learn = self.min_steps_learn // sampler_bs
        agent.give_min_itr_learn(self.min_itr_learn)
        self.initialize_replay_buffer(examples, batch_spec)
        self.optim_initialize(rank)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Used in async runner only; returns replay buffer allocated in shared
        memory, does not instantiate optimizer. """
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.initialize_replay_buffer(examples, batch_spec, async_=True)
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = self.updates_per_sync
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        agent.give_min_itr_learn(self.min_itr_learn)
        return self.replay_buffer

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        self.pi_optimizer = self.OptimCls(self.agent.pi_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.q_optimizer = self.OptimCls(self.agent.q_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.fixed_alpha is False:
            self._log_alpha = torch.tensor(np.log([self.initial_alpha]), requires_grad=True)
            self._alpha = torch.exp(self._log_alpha.detach())
            self.alpha_optimizer = self.OptimCls((self._log_alpha,),
                lr=self.learning_rate, **self.optim_kwargs)
        else:
            self._log_alpha = torch.tensor([np.log(self.initial_alpha)])
            self._alpha = torch.tensor([self.initial_alpha])
            self.alpha_optimizer = None

        self.target_entropy = -np.prod(self.agent.env_spaces.action.shape)
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        if self.action_prior == "gaussian":
            self.action_prior_distribution = Gaussian(
                dim=np.prod(self.agent.env_spaces.action.shape), std=1.)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.
        """
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        ReplayCls = AsyncUniformReplayBuffer if async_ else UniformReplayBuffer
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            n_step_return=self.n_step_return,
            initial_replay_buffer_dict=self.initial_replay_buffer_dict,
        )
        if self.ReplayBufferCls is not None:
            ReplayCls = self.ReplayBufferCls
            logger.log(f"WARNING: ignoring internal selection logic and using"
                f" input replay buffer class: {ReplayCls} -- compatibility not"
                " guaranteed.")
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info

        for _ in range(self.updates_per_optimize):
            samples_r = self.replay_buffer.sample_batch(self.batch_size)

            #####
            agent_inputs, target_inputs, action = buffer_to(
                (samples_r.agent_inputs, samples_r.target_inputs, samples_r.action))

            if self.mid_batch_reset and not self.agent.recurrent:
                valid = torch.ones_like(samples_r.done, dtype=torch.float)  # or None
            else:
                valid = valid_from_done(samples_r.done)

            with torch.no_grad():
                target_action, target_log_pi, _ = self.agent.pi(*target_inputs)
                target_q1, target_q2 = self.agent.target_q(*target_inputs, target_action)
                
                #! before this block is outside of torch.no_grad()
                min_target_q = torch.min(target_q1, target_q2)
                target_value = min_target_q - self._alpha * target_log_pi
                disc = self.discount ** self.n_step_return
                y = (self.reward_scale * samples_r.return_ +
                    (1 - samples_r.done_n.float()) * disc * target_value)

            # Get current Q estimates
            q1, q2 = self.agent.q(*agent_inputs, action)
            q1_loss = 0.5 * valid_mean((y - q1) ** 2, valid)
            q2_loss = 0.5 * valid_mean((y - q2) ** 2, valid)
            q_loss = q1_loss + q2_loss

            self.q_optimizer.zero_grad()
            q_loss.backward()
            q_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q_parameters(), self.clip_grad_norm)
            self.q_optimizer.step()

            # Actor, do not update conv layers
            new_action, log_pi, (pi_mean, pi_log_std) = self.agent.pi(*agent_inputs, detach_encoder=True)
            log_target1, log_target2 = self.agent.q(*agent_inputs, new_action, detach_encoder=True)
            min_log_target = torch.min(log_target1, log_target2)
            prior_log_pi = self.get_action_prior(new_action.cpu())
            pi_losses = self._alpha * log_pi - min_log_target - prior_log_pi
            pi_loss = valid_mean(pi_losses, valid)

            if self.fixed_alpha is False:
                alpha_losses = - self._log_alpha * (log_pi.detach() + self.target_entropy)
                alpha_loss = valid_mean(alpha_losses, valid)
            else:
                alpha_loss = None
            #####

            # torch.autograd.set_detect_anomaly(True)

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(), self.clip_grad_norm)
            # print(self.agent.q_model.encoder.conv.conv_layers[0].weight.grad)
            if self.update_counter % self.actor_update_interval == 0:
                self.pi_optimizer.step()

            if alpha_loss is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.update_counter % self.actor_update_interval == 0:
                    self.alpha_optimizer.step()
                self._alpha = torch.exp(self._log_alpha.detach())

            losses = (q_loss, pi_loss, alpha_loss)
            values = tuple(val.detach() for val in (q1, q2, pi_mean,pi_log_std))
            grad_norms = (q_grad_norm, pi_grad_norm)

            self.append_opt_info_(opt_info, losses, grad_norms, values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

        return opt_info

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method."""
        samples_to_buffer = SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        return samples_to_buffer

    def get_action_prior(self, action):
        if self.action_prior == "uniform":
            prior_log_pi = 0.0
        elif self.action_prior == "gaussian":
            prior_log_pi = self.action_prior_distribution.log_likelihood(
                action, GaussianDistInfo(mean=torch.zeros_like(action)))
        return prior_log_pi

    def append_opt_info_(self, opt_info, losses, grad_norms, values):
        """In-place."""
        q_loss, pi_loss, alpha_loss = losses
        q_grad_norm, pi_grad_norm = grad_norms

        q1, q2, pi_mean, pi_log_std = values
        opt_info.qLoss.append(q_loss.item())
        opt_info.piLoss.append(pi_loss.item())
        opt_info.qGradNorm.append(q_grad_norm.clone().detach().item())  # backwards compatible
        opt_info.piGradNorm.append(pi_grad_norm.clone().detach().item())  # backwards compatible
        opt_info.q1.extend(q1[::10].numpy())  # Downsample for stats.
        opt_info.q2.extend(q2[::10].numpy())
        opt_info.piMu.extend(pi_mean[::10].numpy())
        opt_info.piLogStd.extend(pi_log_std[::10].numpy())
        opt_info.qMeanDiff.append(torch.mean(abs(q1 - q2)).item())
        opt_info.alpha.append(self._alpha.item())

    def optim_state_dict(self):
        return dict(
            pi_optimizer=self.pi_optimizer.state_dict(),
            q_optimizer=self.q_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            log_alpha=self._log_alpha.detach().item(),
        )

    def load_optim_state_dict(self, state_dict):
        self.pi_optimizer.load_state_dict(state_dict["pi_optimizer"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        if self.alpha_optimizer is not None and state_dict["alpha_optimizer"] is not None:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        with torch.no_grad():
            self._log_alpha[:] = state_dict["log_alpha"]
            self._alpha = torch.exp(self._log_alpha.detach())

    def replay_buffer_dict(self):
        return dict(buffer=self.replay_buffer.samples)
