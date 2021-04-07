
import numpy as np
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class PiMlpModel(torch.nn.Module):
    """Action distrubition MLP model for SAC agent."""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.mlp = nn.Sequential(
                        nn.Linear(int(np.prod(observation_shape)), hidden_sizes[0]),
                        nn.LayerNorm(hidden_sizes[0]),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ELU(),
                        nn.Linear(hidden_sizes[1], action_size*2),
        )

    def forward(self, observation, prev_action, prev_reward, detach_encoder=False): # dummy
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class QofMuMlpModel(torch.nn.Module):
    """Q portion of the model for DDPG, an MLP."""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.q1_mlp = nn.Sequential(
                        nn.Linear(int(np.prod(observation_shape)) + action_size, hidden_sizes[0]),
                        nn.LayerNorm(hidden_sizes[0]),
                        nn.ELU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ELU(),
                        nn.Linear(hidden_sizes[1], 1),
        )
        self.q2_mlp = nn.Sequential(
                        nn.Linear(int(np.prod(observation_shape)) + action_size, hidden_sizes[0]),
                        nn.LayerNorm(hidden_sizes[0]),
                        nn.ELU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ELU(),
                        nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, observation, prev_action, prev_reward, action, detach_encoder=False): # dummy
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        q1 = self.q1_mlp(q_input).squeeze(-1)
        q1 = restore_leading_dims(q1, lead_dim, T, B)
        q2 = self.q2_mlp(q_input).squeeze(-1)
        q2 = restore_leading_dims(q2, lead_dim, T, B)
        return q1, q2



class MuMlpModel(torch.nn.Module):
    """MLP neural net for action mean (mu) output for DDPG agent."""
    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            output_max=1,
            ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self._output_max = output_max
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=action_size,
        )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        mu = self._output_max * torch.tanh(self.mlp(observation.view(T * B, -1)))
        mu = restore_leading_dims(mu, lead_dim, T, B)
        return mu


class VMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size=None,  # Unused but accept kwarg.
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        v = self.mlp(observation.view(T * B, -1)).squeeze(-1)
        v = restore_leading_dims(v, lead_dim, T, B)
        return v
