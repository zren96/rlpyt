
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel


class VisionModel(torch.nn.Module):
    """
    Feedforward model for vision agents: a convolutional network feeding an
    MLP with outputs distribution means, separate parameter for learned log_std, and separate MLP for state-value estimate.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            fc_sizes=64,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            mu_nonlinearity=torch.nn.Tanh,  # Module form.
            init_log_std=0.,
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self.conv = Conv2dHeadModel(
            image_shape=observation_shape,
            channels=channels or [16, 32],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        mu_mlp = MlpModel(
            input_size=self.conv.output_size,
            hidden_sizes=fc_sizes,
            output_size=action_size,
            nonlinearity=hidden_nonlinearity,
        )
        if mu_nonlinearity is not None:
            self.mu = torch.nn.Sequential(mu_mlp, mu_nonlinearity())
        else:
            self.mu = mu_mlp
        self.v = MlpModel(
            input_size=self.conv.output_size,
            hidden_sizes=fc_sizes,
            output_size=1,
            nonlinearity=hidden_nonlinearity,
        )
        self.log_std = torch.nn.Parameter(init_log_std *torch.ones(action_size))


    def forward(self, image, prev_action, prev_reward):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        # img = image.type(torch.float)  # Expect torch.uint8 inputs
        # img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        # Expects [0-1] float

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)

        fc_out = self.conv(image.view(T * B, *img_shape))
        mu = self.mu(fc_out)
        v = self.v(fc_out).squeeze(-1)
        log_std = self.log_std.repeat(T * B, 1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)
        return mu, log_std, v
