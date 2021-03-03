

from rlpyt.agents.pg.gaussian import (GaussianPgAgent,
    RecurrentGaussianPgAgent, AlternatingRecurrentGaussianPgAgent)
from rlpyt.models.pg.vision_model import VisionModel


class VisionMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    Now supports observation normalization, including multi-GPU.
    """
    _ddp = False  # Sets True if data parallel, for normalized obs

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract observation_shape and action_size."""
        assert len(env_spaces.action.shape) == 1
        return dict(observation_shape=env_spaces.observation.shape,
                    action_size=env_spaces.action.shape[0])

    def data_parallel(self, *args, **kwargs):
        super().data_parallel(*args, **kwargs)
        self._ddp = True


class VisionAgent(VisionMixin, GaussianPgAgent):

    def __init__(self, ModelCls=VisionModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
