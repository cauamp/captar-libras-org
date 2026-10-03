from ignite.contrib.handlers import (
    LinearCyclicalScheduler,
    ConcatScheduler,
)
from scheduler.reducelr import IgniteCustomReduceLROnPlateauScheduler


def get_warmup_with_reduceeonplateau(args, optimizer, dl):
    lr = args.lr
    if "lr_scale_factor" in args.lr_scheduler_params:
        lr_scale_factor = args.lr_scheduler_params["lr_scale_factor"]
        start_value = lr * lr_scale_factor
    else:
        start_value = args.lr_scheduler_params["start_lr"]
    warmup_epochs = args.lr_scheduler_params["warmup_epochs"]

    if args.train_length:
        epoch_length = args.train_length
    else:
        epoch_length = len(dl)

    scheduler_1 = LinearCyclicalScheduler(
        optimizer,
        "lr",
        start_value=start_value,
        end_value=lr,
        cycle_size=epoch_length * warmup_epochs * 2,
    )

    scheduler_2 = IgniteCustomReduceLROnPlateauScheduler(
        optimizer,
        param_name="lr",
        mode=args.lr_scheduler_params["mode"],
        factor=args.lr_scheduler_params["factor"],
        patience=args.lr_scheduler_params["patience"],
        threshold_mode=args.lr_scheduler_params["threshold_mode"],
        threshold=args.lr_scheduler_params["threshold"],
        cooldown=args.lr_scheduler_params["cooldown"],
        verbose=args.lr_scheduler_params["verbose"],
    )

    durations = [
        epoch_length * warmup_epochs,
    ]
    scheduler = ConcatScheduler(
        schedulers=[scheduler_1, scheduler_2], durations=durations
    )
    return scheduler
