
import numpy as np
import torch
from torch import nn

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dHeadModel, Conv2dModel
from rlpyt.utils.spatial_softmax import SpatialSoftmax
from rlpyt.models.utils import tie_weights

RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work


class PiConvModel(torch.nn.Module):
    """Action distrubition model for SAC Vision."""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=[64,64],  # mlp after lstm
            fc_sizes=64, # Between conv and lstm
            channels=None,
            kernel_sizes=None,
            strides=None,
            paddings=None,
            use_maxpool=False,
            ):
        super().__init__()
        self._action_size = action_size

        self.conv = Conv2dHeadModel(
            image_shape=observation_shape,
            channels=channels or [4, 8],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )   # image -> conv (ReLU) -> linear (fc_sizes) - > ReLU

        self.mlp = MlpModel(
            input_size=self.conv.output_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )
        # print('Num of conv parameters: %d' % sum(p.numel() for p in self.conv.parameters() if p.requires_grad))
        # print('Num of mlp parameters: %d' % sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))
        # while 1:
        #     continue

    def forward(self, image, prev_action, prev_reward):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        fc_out = self.conv(image.view(T * B, *img_shape))
        output = self.mlp(fc_out.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std

class QConvModel(torch.nn.Module):
    """Q portion of the model for SAC Vision. Use single-branch for now (LSTM takes obs, last_action, and action). https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/common/value_networks.py"""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=[64,64],  # mlp after lstm
            fc_sizes=64, # Between conv and lstm
            channels=None,
            kernel_sizes=None,
            strides=None,
            paddings=None,
            use_maxpool=False,
            ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)

        self.conv = Conv2dHeadModel(
            image_shape=observation_shape,
            channels=channels or [4, 8],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )   # image -> conv (ReLU) -> linear (fc_sizes) - > ReLU
        self.mlp = MlpModel(
            input_size=self.conv.output_size+action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, image, prev_action, prev_reward, action):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        fc_out = self.conv(image.view(T * B, *img_shape))
        q_input = torch.cat(
            [fc_out.view(T * B, -1), 
             action.view(T * B, -1)], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

class Encoder(torch.nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self,
                observation_shape,
                channels=[16, 16, 16],
                kernel_sizes=[5, 3, 3],
                strides=[1, 1, 1],
                paddings=[2, 1, 1],
                spatial_softmax=False,
                ):
        super().__init__()
        self.observation_shape = observation_shape
        self.channels = channels
        self.conv = Conv2dModel(    # keep image dim
            in_channels=observation_shape[0],
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=False,
        )   # image -> conv (ReLU)
        # print(self.conv.conv_out_size(h=48, w=48))

        # Spatial softmax, output mum_channel x 2d pos
        self.spatial_softmax = spatial_softmax
        if spatial_softmax:
            self.sm = SpatialSoftmax(height=observation_shape[1], 
                                width=observation_shape[2], 
                                channel=channels[-1])

    def forward_conv(self, image):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        image = image.view(T * B, *img_shape)
        conv_out = self.conv(image) # 1 x channel(64) x 48 x 48
        # print(conv_out.shape)
        # while 1:
        #     continue
        if self.spatial_softmax:
            conv_out = self.sm(conv_out)
        return conv_out

    def forward(self, image, detach=False):
        out = self.forward_conv(image)
        if detach:
            out = out.detach()
        return out

    def output_size(self):
        if self.spatial_softmax:
            return self.channels[-1]*2
        else:
            return self.conv.conv_out_size(h=self.observation_shape[1], 
                                            w=self.observation_shape[2])

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(len(self.conv.conv_layers)):
            tie_weights(src=source.conv.conv_layers[i], trg=self.conv.conv_layers[i])

class PiConvTiedModel(torch.nn.Module):
    def __init__(self,
                observation_shape,
                action_size,
                hidden_sizes=[64, 64],  # mlp
                channels=[16, 16, 16],
                kernel_sizes=[5, 3, 3],
                strides=[1, 1, 1],
                paddings=[2, 1, 1],
                ):
        super().__init__()
        self._action_size = action_size
        self.encoder = Encoder(observation_shape,
                                channels,
                                kernel_sizes,
                                strides,
                                paddings)
        self.mlp = nn.Sequential(
                        nn.Linear(self.encoder.output_size(), hidden_sizes[0]),
                        nn.LayerNorm(hidden_sizes[0]),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], action_size*2),
        )
        # self.mlp = MlpModel(
        #     input_size=self.encoder.output_size(),    # 2d, and then two images
        #     hidden_sizes=hidden_sizes,
        #     output_size=action_size * 2,
        # )
        # print('Num of conv parameters: %d' % sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
        # print('Num of mlp parameters: %d' % sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))
        # while 1:
        #     continue


    def forward(self, image, prev_action, prev_reward, detach_encoder=False):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        conv_out = self.encoder.forward(image, detach=detach_encoder)
        output = self.mlp(conv_out.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class QDoubleConvTiedModel(torch.nn.Module):
    def __init__(self,
                observation_shape,
                action_size,
                hidden_sizes=[64, 64],  # mlp
                channels=[16, 16, 16],
                kernel_sizes=[5, 3, 3],
                strides=[1, 1, 1],
                paddings=[2, 1, 1],
                ):

        super().__init__()
        self.encoder = Encoder(observation_shape,
                                channels,
                                kernel_sizes,
                                strides,
                                paddings)
        self.q1_head = nn.Sequential(
                        nn.Linear(self.encoder.output_size()+action_size, hidden_sizes[0]),
                        nn.LayerNorm(hidden_sizes[0]),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], 1),
        )
        self.q2_head = nn.Sequential(
                        nn.Linear(self.encoder.output_size()+action_size, hidden_sizes[0]),
                        nn.LayerNorm(hidden_sizes[0]),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], 1),
        )
        # self.q1_head = MlpModel(
        #         input_size=self.encoder.output_size()+action_size,
        #         hidden_sizes=hidden_sizes,
        #         output_size=1,
        #     )
        # self.q2_head = MlpModel(
        #         input_size=self.encoder.output_size()+action_size,
        #         hidden_sizes=hidden_sizes,
        #         output_size=1,
        #     )

    def forward(self, image, prev_action, prev_reward, action, detach_encoder=False):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        conv_out = self.encoder.forward(image, detach=detach_encoder)
        q_input = torch.cat(
            [conv_out.view(T * B, -1), 
             action.view(T * B, -1)], dim=1)

        q1 = self.q1_head(q_input).squeeze(-1)
        q1 = restore_leading_dims(q1, lead_dim, T, B)
        q2 = self.q2_head(q_input).squeeze(-1)
        q2 = restore_leading_dims(q2, lead_dim, T, B)
        return q1, q2

class PiConvDoubleModel(torch.nn.Module):
    """Action distrubition model for SAC Vision."""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=[64, 64],  # mlp
            channels=[4, 8, 16],
            kernel_sizes=[5, 3, 3],
            strides=[1, 1, 1],
            paddings=[2, 1, 1],
            use_maxpool=False,
            ):
        # To keep image dimension, use stride=1, kernel=2*paddings+1
        super().__init__()
        self._action_size = action_size

        self.conv = Conv2dModel(    # keep image dim
            # image_shape=[observation_shape[0]//2, observation_shape[1], observation_shape[2]],
            in_channels=3,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=use_maxpool,
        )   # image -> conv (ReLU) (-> linear (fc_sizes) - > ReLU)

        # Spatial softmax, output mum_channel x 2d pos
        self.sm = SpatialSoftmax(height=observation_shape[1], 
                                width=observation_shape[2], 
                                channel=channels[-1])

        self.mlp = MlpModel(
            input_size=channels[-1]*2*2,    # 2d, and then two images
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )
        # print('Num of conv parameters: %d' % sum(p.numel() for p in self.conv.parameters() if p.requires_grad))
        # print('Num of mlp parameters: %d' % sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))
        # while 1:
        #     continue

    def forward(self, image, prev_action, prev_reward):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        image = image.view(T * B, *img_shape)
        image_cur = image[:,:3,:,:]
        image_prev = image[:,3:,:,:]

        fc_cur_out = self.sm(self.conv(image_cur))
        fc_prev_out = self.sm(self.conv(image_prev))
        fc_out = torch.cat((fc_cur_out, fc_prev_out), dim=-1)
        output = self.mlp(fc_out.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std

class QConvDoubleModel(torch.nn.Module):
    """Q portion of the model for SAC Vision. Use single-branch for now (LSTM takes obs, last_action, and action). https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/common/value_networks.py"""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=[64, 64],  # mlp
            channels=[4, 8, 16],
            kernel_sizes=[5, 3, 3],
            strides=[1, 1, 1],
            paddings=[2, 1, 1],
            use_maxpool=False,
            ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)

        self.conv = Conv2dModel(    # keep image dim
            # image_shape=[observation_shape[0]//2, observation_shape[1], observation_shape[2]],
            in_channels=3,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=use_maxpool,
        )   # image -> conv (ReLU) (-> linear (fc_sizes) - > ReLU)

        # Spatial softmax, output mum_channel x 2d pos
        self.sm = SpatialSoftmax(height=observation_shape[1], 
                                width=observation_shape[2], 
                                channel=channels[-1])

        self.mlp = MlpModel(
            input_size=channels[-1]*2*2+action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, image, prev_action, prev_reward, action):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)

        image = image.view(T * B, *img_shape)
        image_cur = image[:,:3,:,:]
        image_prev = image[:,3:,:,:]

        fc_cur_out = self.sm(self.conv(image_cur))
        fc_prev_out = self.sm(self.conv(image_prev))
        fc_out = torch.cat((fc_cur_out, fc_prev_out), dim=-1)
        q_input = torch.cat(
            [fc_out.view(T * B, -1), 
             action.view(T * B, -1)], dim=1)

        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

class PiConvLSTMModel(torch.nn.Module):
    """Action distrubition model for SAC Vision."""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=[64,64],  # mlp after lstm
            fc_sizes=128, # Between conv and lstm
            lstm_size=64,
            channels=[8, 16],
            kernel_sizes=[8, 4],
            strides=[4, 2],
            paddings=[0, 1],
            use_maxpool=False,
            ):
        super().__init__()
        self._action_size = action_size

        self.conv = Conv2dHeadModel(
            image_shape=observation_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )   # image -> conv (ReLU) -> linear (fc_sizes) - > ReLU

        self.lstm = torch.nn.LSTM(self.conv.output_size + action_size + 1, lstm_size)   # Input to LSTM: conv_output + prev_action + prev_reward 
        self.mlp = MlpModel(
            input_size=lstm_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )
        # print('Num of conv parameters: %d' % sum(p.numel() for p in self.conv.parameters() if p.requires_grad))
        # print('Num of lstm parameters: %d' % sum(p.numel() for p in self.lstm.parameters() if p.requires_grad))
        # print('Num of mlp parameters: %d' % sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))
        # while 1:
        #     continue

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        fc_out = self.conv(image.view(T * B, *img_shape))
        lstm_input = torch.cat([
            fc_out.view(T, B, -1),
            prev_action.view(T, B, -1),
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        output = self.mlp(lstm_out.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        next_rnn_state = RnnState(h=hn, c=cn)
        return mu, log_std, next_rnn_state

class QConvLSTMModel(torch.nn.Module):
    """Q portion of the model for SAC Vision. Use single-branch for now (LSTM takes obs, last_action, and action). https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/common/value_networks.py"""

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=[64,64],  # mlp after lstm
            fc_sizes=128, # Between conv and lstm
            lstm_size=64,
            channels=[8, 16],
            kernel_sizes=[8, 4],
            strides=[4, 2],
            paddings=[0, 1],
            use_maxpool=False,
            ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)

        self.conv = Conv2dHeadModel(
            image_shape=observation_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )   # image -> conv (ReLU) -> linear (fc_sizes) - > ReLU

        self.lstm = torch.nn.LSTM(self.conv.output_size + action_size + 1 + action_size, lstm_size)   # Input to LSTM: conv_output + prev_action + prev_reward + action
        self.mlp = MlpModel(
            input_size=lstm_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, image, prev_action, prev_reward, action, init_rnn_state):
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        fc_out = self.conv(image.view(T * B, *img_shape))
        lstm_input = torch.cat([
            fc_out.view(T, B, -1),
            prev_action.view(T, B, -1),
            prev_reward.view(T, B, 1),
            action.view(T, B, -1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        q = self.mlp(lstm_out.view(T * B, -1)).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        next_rnn_state = RnnState(h=hn, c=cn)
        return q, next_rnn_state

        # lead_dim, T, B, _ = infer_leading_dims(observation,
        #     self._obs_ndim)
        # q_input = torch.cat(
        #     [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        # q = self.mlp(q_input).squeeze(-1)
        # q = restore_leading_dims(q, lead_dim, T, B)
        # return q
