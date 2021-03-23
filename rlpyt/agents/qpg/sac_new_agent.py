
import numpy as np
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.models.qpg.conv import PiConvTiedModel, QDoubleConvTiedModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q"])


class SacNewAgent(BaseAgent):
    """Agent for SAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(
            self,
            model_kwargs=None,
            q_model_kwargs=None,
            initial_model_state_dict=None,  # All models.
            action_squash=2.,  # Max magnitude (or None).
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            saliency_dir=None,
            ):
        """Saves input arguments; network defaults stored within."""
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[256, 256])
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[256, 256])
        super().__init__(ModelCls=PiConvTiedModel, model_kwargs=model_kwargs,
            initial_model_state_dict=initial_model_state_dict)
        save__init__args(locals())
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None  # Don't let base agent try to load.
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.initial_model_state_dict = _initial_model_state_dict

        self.q_model = QDoubleConvTiedModel(**self.env_model_kwargs, **self.q_model_kwargs)
        self.target_q_model = QDoubleConvTiedModel(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

        #* Tie weights
        self.model.encoder.copy_conv_weights_from(self.q_model.encoder)


    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.q_model.to(self.device)
        self.target_q_model.to(self.device)

    def data_parallel(self):
        device_id = super().data_parallel
        self.q_model = DDP(
            self.q_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        return device_id

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    def q(self, observation, prev_action, prev_reward, action, detach_encoder=False):
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q1, q2 = self.q_model(*model_inputs, detach_encoder=detach_encoder)
        return q1.cpu(), q2.cpu()

    def target_q(self, observation, prev_action, prev_reward, action):
        """Compute twin target Q-values for state/observation and input
        action.""" 
        model_inputs = buffer_to((observation, prev_action,
            prev_reward, action), device=self.device)
        target_q1, target_q2 = self.target_q_model(*model_inputs)
        return target_q1.cpu(), target_q2.cpu()

    def pi(self, observation, prev_action, prev_reward, detach_encoder=False):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mean, log_std = self.model(*model_inputs, detach_encoder=detach_encoder)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_q_model, self.q_model.state_dict(), tau)

    @property
    def models(self):
        return Models(pi=self.model, q=self.q_model)

    def pi_parameters(self):
        return self.model.parameters()

    def q_parameters(self):
        return self.q_model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.q_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.q_model.eval()
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.q_model.eval()
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            q_model=self.q_model.state_dict(),
            target_q_model=self.target_q_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.q_model.load_state_dict(state_dict["q_model"])
        self.target_q_model.load_state_dict(state_dict["target_q_model"])