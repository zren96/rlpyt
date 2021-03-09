
import numpy as np
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.sequence.frame import UniformSequenceReplayFrameBuffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.distributions.gaussian import Gaussian
from rlpyt.distributions.gaussian import DistInfo as GaussianDistInfo
from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.buffer import buffer_to, buffer_method


OptInfo = namedtuple("OptInfo",
    ["q1Loss", "q2Loss", "piLoss",
    "q1GradNorm", "q2GradNorm", "piGradNorm",
    "q1", "q2", "piMu", "piLogStd", "qMeanDiff", "alpha"])
SamplesToBufferLSTM = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done", "prev_rnn_state"])


class SAC_LSTM(RlAlgorithm):
    """Soft actor critic algorithm, training from a replay buffer."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.99,
            batch_T=80,
            batch_B=16,
            warmup_T=40,
            min_steps_learn=int(1e5),
            replay_size=int(1e6),
            replay_ratio=4,  # data_consumption / data_generation
            store_rnn_state_interval=40,
            target_update_tau=0.005,  # tau=1 for hard update.
            target_update_interval=1,  # 1000 for hard update, 1 for soft.
            learning_rate=3e-4,
            fixed_alpha=None, # None for adaptive alpha, float for any fixed value
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,  # for all of them.
            initial_replay_buffer_dict=None,
            action_prior="uniform",  # or "gaussian"
            reward_scale=1,
            target_entropy="auto",  # "auto", float, or None
            reparameterize=True,
            clip_grad_norm=1e3,
            n_step_return=5,
            ReplayBufferCls=None,  # Leave None to select by above options.
            ):
        """ Save input arguments.
        Args:
            store_rnn_state_interval (int): store RNN state only once this many steps, to reduce memory usage; replay sequences will only begin at the steps with stored recurrent state.
        Note:
            Typically ran with ``store_rnn_state_interval`` equal to the sampler's ``batch_T``, 40.  Then every 40 steps
            can be the beginning of a replay sequence, and will be guaranteed to start with a valid RNN state.  Only reset
            the RNN state (and env) at the end of the sampler batch, so that the beginnings of episodes are trained on.
        """
        if optim_kwargs is None:
            optim_kwargs = dict()
        assert action_prior in ["uniform", "gaussian"]
        save__init__args(locals())
        self._batch_size = (self.batch_T + self.warmup_T) * self.batch_B

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """Stores input arguments and initializes replay buffer and optimizer.
        Use in non-async runners.  Computes number of gradient updates per
        optimization iteration as `(replay_ratio * sampler-batch-size /
        training-batch_size)`."""
        self.agent = agent
        self.n_itr = n_itr  # num_itr
        self.sampler_bs = sampler_bs = batch_spec.size  # num_step_per_batch
        self.mid_batch_reset = mid_batch_reset  # True
        self.updates_per_optimize = max(1, round(self.replay_ratio * sampler_bs / self._batch_size))
        logger.log(f"From sampler batch size {batch_spec.size}, training "
            f"batch size {self._batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        agent.give_min_itr_learn(self.min_itr_learn)    # filling up replay_buffer
        self.initialize_replay_buffer(examples, batch_spec)
        self.optim_initialize(rank)

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        self.pi_optimizer = self.OptimCls(self.agent.pi_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.q1_optimizer = self.OptimCls(self.agent.q1_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.q2_optimizer = self.OptimCls(self.agent.q2_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.fixed_alpha is None:
            self._log_alpha = torch.zeros(1, requires_grad=True)
            self._alpha = torch.exp(self._log_alpha.detach())
            self.alpha_optimizer = self.OptimCls((self._log_alpha,),
                lr=self.learning_rate, **self.optim_kwargs)
        else:
            self._log_alpha = torch.tensor([np.log(self.fixed_alpha)])
            self._alpha = torch.tensor([self.fixed_alpha])
            self.alpha_optimizer = None
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.agent.env_spaces.action.shape)
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        if self.action_prior == "gaussian":
            self.action_prior_distribution = Gaussian(
                dim=np.prod(self.agent.env_spaces.action.shape), std=1.)

    def initialize_replay_buffer(self, examples, batch_spec):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer` namedarraytuple.
        """
        # hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print(examples["agent_info"])
        example_to_buffer = SamplesToBufferLSTM(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            prev_rnn_state=examples["agent_info"].prev_rnn_state,
        )
        ReplayCls = UniformSequenceReplayFrameBuffer
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=self.store_rnn_state_interval,
            initial_replay_buffer_dict=self.initial_replay_buffer_dict,
            batch_T=self.batch_T + self.warmup_T,
        )
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, itr, samples=None):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """

        # Update replay buffer
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info

        for _ in range(self.updates_per_optimize):

            samples_from_replay = self.replay_buffer.sample_batch(self.batch_B)
            losses, values = self.loss(samples_from_replay)
            q1_loss, q2_loss, pi_loss, alpha_loss = losses

            if alpha_loss is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self._alpha = torch.exp(self._log_alpha.detach())

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.pi_parameters(), self.clip_grad_norm)
            self.pi_optimizer.step()

            # Step Q's last because pi_loss.backward() uses them?
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q1_parameters(), self.clip_grad_norm)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.q2_parameters(), self.clip_grad_norm)
            self.q2_optimizer.step()

            grad_norms = (q1_grad_norm, q2_grad_norm, pi_grad_norm)

            self.append_opt_info_(opt_info, losses, grad_norms, values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

        return opt_info

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method."""
        samples_to_buffer = SamplesToBufferLSTM(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            prev_rnn_state=samples.agent.agent_info.prev_rnn_state,
        )
        return samples_to_buffer

    def loss(self, samples):
        """
        Computes losses for twin Q-values against the min of twin target Q-values and an entropy term.  Computes reparameterized policy loss, and loss for tuning entropy weighting, alpha.  
        
        Input samples have leading batch dimension [B,..] (but not time).
        """
        # SamplesFromReplay = namedarraytuple("SamplesFromReplay",
            # ["all_observation", "all_action", "all_reward", "return_", "done", "done_n", "init_rnn_state"])
        all_observation, all_action, all_reward = buffer_to(
            (samples.all_observation, samples.all_action, samples.all_reward),
            device=self.agent.device)   # all have (wT + bT + nsr) x bB
        wT, bT, nsr = self.warmup_T, self.batch_T, self.n_step_return
        if wT > 0:
            warmup_slice = slice(None, wT)  # Same for agent and target.
            warmup_inputs = AgentInputs(
                observation=all_observation[warmup_slice],
                prev_action=all_action[warmup_slice],
                prev_reward=all_reward[warmup_slice],
            )
        agent_slice = slice(wT, wT + bT)
        agent_inputs = AgentInputs(
            observation=all_observation[agent_slice],
            prev_action=all_action[agent_slice],
            prev_reward=all_reward[agent_slice],
        )
        target_slice = slice(wT, None)  # Same start t as agent. (wT + bT + nsr)
        target_inputs = AgentInputs(
            observation=all_observation[target_slice],
            prev_action=all_action[target_slice],
            prev_reward=all_reward[target_slice],
        )
        warmup_action = samples.all_action[1:wT+1]
        action = samples.all_action[wT + 1:wT + 1 + bT]  # 'current' action by shifting index by 1 from prev_action
        return_ = samples.return_[wT:wT + bT]
        done_n = samples.done_n[wT:wT + bT]
        if self.store_rnn_state_interval == 0:
            init_rnn_state = None
        else:
            # [B,N,H]-->[N,B,H] cudnn.
            init_rnn_state = buffer_method(samples.init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
        if wT > 0:  # Do warmup.
            with torch.no_grad():
                _, target_q1_rnn_state, _, target_q2_rnn_state = self.agent.target_q(*warmup_inputs,  warmup_action, init_rnn_state, init_rnn_state)
                _, _, _, init_rnn_state = self.agent.pi(*warmup_inputs, init_rnn_state)
            # Recommend aligning sampling batch_T and store_rnn_interval with
            # warmup_T (and no mid_batch_reset), so that end of trajectory
            # during warmup leads to new trajectory beginning at start of
            # training segment of replay.
            warmup_invalid_mask = valid_from_done(samples.done[:wT])[-1] == 0  # [B]
            init_rnn_state[:, warmup_invalid_mask] = 0  # [N,B,H] (cudnn)
            target_q1_rnn_state[:, warmup_invalid_mask] = 0
            target_q2_rnn_state[:, warmup_invalid_mask] = 0
        else:
            target_q1_rnn_state = init_rnn_state
            target_q2_rnn_state = init_rnn_state

        valid = valid_from_done(samples.done)[-bT:]

        q1, _, q2, _ = self.agent.q(*agent_inputs, action, init_rnn_state, init_rnn_state)
        with torch.no_grad():
            target_action, target_log_pi, _, _ = self.agent.pi(*target_inputs, init_rnn_state)
            target_q1, _, target_q2, _ = self.agent.target_q(*target_inputs, target_action, target_q1_rnn_state, target_q2_rnn_state)
            target_q1 = target_q1[-bT:]  # Same length as q.
            target_q2 = target_q2[-bT:]
            target_log_pi = target_log_pi[-bT:]

        min_target_q = torch.min(target_q1, target_q2)
        target_value = min_target_q - self._alpha * target_log_pi
        disc = self.discount ** self.n_step_return
        y = (self.reward_scale * return_ +
            (1 - done_n.float()) * disc * target_value)
        q1_loss = 0.5 * valid_mean((y - q1) ** 2, valid)
        q2_loss = 0.5 * valid_mean((y - q2) ** 2, valid)

        new_action, log_pi, (pi_mean, pi_log_std), _ = self.agent.pi(*agent_inputs, init_rnn_state)
        log_target1, _, log_target2, _ = self.agent.q(*agent_inputs, new_action, init_rnn_state, init_rnn_state)
        min_log_target = torch.min(log_target1, log_target2)
        prior_log_pi = self.get_action_prior(new_action.cpu())

        pi_losses = self._alpha * log_pi - min_log_target - prior_log_pi
        pi_loss = valid_mean(pi_losses, valid)

        if self.target_entropy is not None and self.fixed_alpha is None:
            alpha_losses = - self._log_alpha * (log_pi.detach() + self.target_entropy)
            alpha_loss = valid_mean(alpha_losses, valid)
        else:
            alpha_loss = None

        losses = (q1_loss, q2_loss, pi_loss, alpha_loss)
        values = tuple(val.detach() for val in (q1, q2, pi_mean, pi_log_std))
        return losses, values

    def get_action_prior(self, action):
        if self.action_prior == "uniform":
            prior_log_pi = 0.0
        elif self.action_prior == "gaussian":
            prior_log_pi = self.action_prior_distribution.log_likelihood(
                action, GaussianDistInfo(mean=torch.zeros_like(action)))
        return prior_log_pi

    def append_opt_info_(self, opt_info, losses, grad_norms, values):
        """In-place."""
        q1_loss, q2_loss, pi_loss, alpha_loss = losses
        q1_grad_norm, q2_grad_norm, pi_grad_norm = grad_norms
        q1, q2, pi_mean, pi_log_std = values
        opt_info.q1Loss.append(q1_loss.item())
        opt_info.q2Loss.append(q2_loss.item())
        opt_info.piLoss.append(pi_loss.item())
        opt_info.q1GradNorm.append(q1_grad_norm.clone().detach().item())  # backwards compatible
        opt_info.q2GradNorm.append(q2_grad_norm.clone().detach().item())  # backwards compatible
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
            q1_optimizer=self.q1_optimizer.state_dict(),
            q2_optimizer=self.q2_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            log_alpha=self._log_alpha.detach().item(),
        )

    def load_optim_state_dict(self, state_dict):
        self.pi_optimizer.load_state_dict(state_dict["pi_optimizer"])
        self.q1_optimizer.load_state_dict(state_dict["q1_optimizer"])
        self.q2_optimizer.load_state_dict(state_dict["q2_optimizer"])
        if self.alpha_optimizer is not None and state_dict["alpha_optimizer"] is not None:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        with torch.no_grad():
            self._log_alpha[:] = state_dict["log_alpha"]
            self._alpha = torch.exp(self._log_alpha.detach())

    def replay_buffer_dict(self):
        return dict(buffer=self.replay_buffer.samples)
