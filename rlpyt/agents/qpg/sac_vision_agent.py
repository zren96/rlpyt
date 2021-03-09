
import numpy as np
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP

from rlpyt.agents.base import BaseAgent, AgentStep, RecurrentAgentMixin
from rlpyt.agents.qpg.base import AgentInfo, AgentInfoRnn
from rlpyt.models.qpg.conv import QConvLSTMModel, PiConvLSTMModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


MIN_LOG_STD = -20
MAX_LOG_STD = 2

Models = namedtuple("Models", ["pi", "q1", "q2", "v"])



class SacVisionAgent(BaseAgent):
    """Agent for SAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(
            self,
            ModelCls=PiConvLSTMModel,  # Pi model.
            QModelCls=QConvLSTMModel,
            model_kwargs=None,  # Pi model.
            q_model_kwargs=None,
            initial_model_state_dict=None,  # All models.
            action_squash=2.,  # Max magnitude (or None).
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            ):
        """Saves input arguments; network defaults stored within."""
        if model_kwargs is None:
            model_kwargs = dict()
        if q_model_kwargs is None:
            q_model_kwargs = dict()
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs,
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
        self.q1_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.q2_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.target_q1_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q2_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q1_model.load_state_dict(self.q1_model.state_dict())
        self.target_q2_model.load_state_dict(self.q2_model.state_dict())
        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.target_q1_model.to(self.device)
        self.target_q2_model.to(self.device)

    def data_parallel(self):
        device_id = super().data_parallel
        self.q1_model = DDP(
            self.q1_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        self.q2_model = DDP(
            self.q2_model,
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

    def q(self, observation, prev_action, prev_reward, action, init_rnn_state_1,initi_rnn_state_2):
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs_1 = buffer_to((observation, prev_action, prev_reward,
            action, init_rnn_state_1), device=self.device)
        model_inputs_2 = buffer_to((observation, prev_action, prev_reward,
            action, initi_rnn_state_2), device=self.device)
        q1, q1_rnn_state = self.q1_model(*model_inputs_1)
        q2, q2_rnn_state = self.q2_model(*model_inputs_2)
        return q1.cpu(), q1_rnn_state, q2.cpu(), q2_rnn_state

    def target_q(self, observation, prev_action, prev_reward, action, init_rnn_state_1, init_rnn_state_2):
        """Compute twin target Q-values for state/observation and input
        action.""" 
        model_inputs_1 = buffer_to((observation, prev_action,
            prev_reward, action, init_rnn_state_1), device=self.device)
        model_inputs_2 = buffer_to((observation, prev_action,
            prev_reward, action, init_rnn_state_2), device=self.device)
        target_q1, q1_rnn_state = self.target_q1_model(*model_inputs_1)
        target_q2, q2_rnn_state = self.target_q2_model(*model_inputs_2)
        return target_q1.cpu(), q1_rnn_state, target_q2.cpu(), q2_rnn_state

    def pi(self, observation, prev_action, prev_reward, init_rnn_state):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state),  device=self.device)
        mean, log_std, rnn_state = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # action = self.distribution.sample(dist_info)
        # log_pi = self.distribution.log_likelihood(action, dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info, rnn_state  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mean, log_std, rnn_state = self.model(*model_inputs, self.prev_rnn_state)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)

        # Model handles None, but Buffer does not, make zeros if needed:
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)

        agent_info = AgentInfoRnn(dist_info=dist_info, 
                                  prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # update prev_rnn_state
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_q1_model, self.q1_model.state_dict(), tau)
        update_state_dict(self.target_q2_model, self.q2_model.state_dict(), tau)

    @property
    def models(self):
        return Models(pi=self.model, q1=self.q1_model, q2=self.q2_model)

    def pi_parameters(self):
        return self.model.parameters()

    def q1_parameters(self):
        return self.q1_model.parameters()

    def q2_parameters(self):
        return self.q2_model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.q1_model.train()
        self.q2_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            q1_model=self.q1_model.state_dict(),
            q2_model=self.q2_model.state_dict(),
            target_q1_model=self.target_q1_model.state_dict(),
            target_q2_model=self.target_q2_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.q1_model.load_state_dict(state_dict["q1_model"])
        self.q2_model.load_state_dict(state_dict["q2_model"])
        self.target_q1_model.load_state_dict(state_dict["target_q1_model"])
        self.target_q2_model.load_state_dict(state_dict["target_q2_model"])


class SacVisionLSTMAgent(RecurrentAgentMixin, SacVisionAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
