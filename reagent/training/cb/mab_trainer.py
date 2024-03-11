# pyre-unsafe
import logging

from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.models.mab import MABBaseModel
from reagent.training.cb.base_trainer import BaseCBTrainerWithEval

logger = logging.getLogger(__name__)


class MABTrainer(BaseCBTrainerWithEval):
    def __init__(
        self,
        policy: Policy,
        *args,
        **kwargs,
    ):
        super().__init__(automatic_optimization=False, *args, **kwargs)
        assert isinstance(policy.scorer, MABBaseModel)
        self.scorer = policy.scorer

    def cb_training_step(self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0):
        self.scorer.learn(batch)

    def configure_optimizers(self):
        # no optimizers bcs we update state manually
        return None
