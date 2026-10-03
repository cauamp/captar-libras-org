from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Mapping
from ignite.handlers.param_scheduler import ParamScheduler


class CustomReduceLROnPlateauScheduler:
    """Wrapper of torch.optim.lr_scheduler.ReduceLROnPlateau with __call__ method like other schedulers in
    contrib\handlers\param_scheduler.py"""

    def __init__(
        self,
        optimizer,
        name,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
    ):
        self.name = name
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )

    def __call__(self, engine, name=None):
        self.scheduler.step(engine.state.metrics[self.name])

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Copies parameters from :attr:`state_dict` into this BaseParamScheduler.

        Args:
            state_dict: a dict containing parameters.
        """
        self.scheduler.load_state_dict(state_dict)


class IgniteCustomReduceLROnPlateauScheduler(ParamScheduler):
    def __init__(
        self,
        optimizer,
        param_name,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
    ):
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )
        self.param_name = param_name
        self.optimizer = optimizer
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def __call__(self, engine, event_name=None):
        metric_value = engine.state.metrics.get(self.param_name)
        if metric_value is not None:
            self.scheduler.step(metric_value)
            self._last_lr = [group["lr"]
                             for group in self.optimizer.param_groups]

    def get_param(self):
        return self._last_lr[0]  # ou uma média se tiver múltiplos grupos

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.scheduler.load_state_dict(state_dict)
