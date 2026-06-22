"""MMEngine hooks imported only in the cloud training environment."""

from mmaction.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class FreezeBackboneHook(Hook):  # type: ignore[misc]
    def __init__(self, freeze_epochs: int = 5) -> None:
        self.freeze_epochs = freeze_epochs

    def before_train_epoch(self, runner) -> None:  # type: ignore[no-untyped-def]
        freeze = runner.epoch < self.freeze_epochs
        for parameter in runner.model.backbone.parameters():
            parameter.requires_grad = not freeze
