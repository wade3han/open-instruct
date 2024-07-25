from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_constant_schedule_with_warmup_and_cooldown_lr_lambda(current_step: int, *,
                                                              num_warmup_steps: int,
                                                              num_training_steps: int,
                                                              num_cooldown_steps: int, ):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    if num_training_steps - current_step < num_cooldown_steps:
        return float(num_training_steps - current_step) / float(max(1.0, num_cooldown_steps))
    return 1.0


def get_constant_schedule_with_warmup_and_cooldown(optimizer: Optimizer,
                                                   num_warmup_steps: int,
                                                   num_cooldown_steps: int,
                                                   num_training_steps: int,
                                                   last_epoch: int = -1):
    lr_lambda = partial(_get_constant_schedule_with_warmup_and_cooldown_lr_lambda,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps,
                        num_cooldown_steps=num_cooldown_steps, )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
