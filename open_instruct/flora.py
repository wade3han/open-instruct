import math
import warnings
from typing import Callable, Iterable, TypeAlias, Union, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class Flora(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3,
            projection_steps: int = 1000,
            rank: int = 8,
            beta: float = 0.9,
            no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta} - should be in [0.0, 1.0)")
        defaults = dict(lr=lr, rank=rank, projection_steps=projection_steps, beta=beta)
        super().__init__(params, defaults)

        self.state['step'] = 0

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.state['step'] += 1

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                # State initialization
                if "seed" not in state:
                    state["seed"] = int(np.random.randint(low=0, high=2 ** 16, size=1)[0])
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)

                # Compression
                if (self.state['step'] - 1) % group['projection_steps'] == 0:
                    print("seed_initialize")
                    seed = state["seed"]
                    new_seed = int(np.random.randint(low=0, high=2 ** 16, size=1)[0])
                    generator = torch.Generator(device=p.device).manual_seed(seed)
                    new_generator = torch.Generator(device=p.device).manual_seed(new_seed)

                    state["seed"] = new_seed

                    rank = group["rank"]

                    if p.dim() < 2:
                        projection = torch.randn(rank, p.shape[0], generator=generator,
                                                 device=p.device) / math.sqrt(rank)
                        new_projection = torch.randn(rank, p.shape[0], generator=new_generator,
                                                     device=p.device) / math.sqrt(rank)

                        state["exp_avg"] = torch.matmul(torch.matmul(new_projection, projection.T), state["exp_avg"])
                        exp_avg = state["exp_avg"]
                        projection = new_projection
                        grad = torch.matmul(projection, grad)

                    elif p.dim() == 2:
                        if p.shape[0] >= p.shape[1]:
                            projection = torch.randn(rank, p.shape[0], generator=generator,
                                                     device=p.device) / math.sqrt(rank)
                            new_projection = torch.randn(rank, p.shape[0], generator=new_generator,
                                                         device=p.device) / math.sqrt(rank)

                            state["exp_avg"] = torch.matmul(torch.matmul(new_projection, projection.T),
                                                            state["exp_avg"])
                            exp_avg = state["exp_avg"]
                            projection = new_projection
                            grad = torch.matmul(projection, grad)

                        else:
                            projection = torch.randn(p.shape[1], rank, generator=generator,
                                                     device=p.device) / math.sqrt(rank)
                            new_projection = torch.randn(p.shape[1], rank, generator=new_generator,
                                                         device=p.device) / math.sqrt(rank)

                            state["exp_avg"] = torch.matmul(state["exp_avg"],
                                                            torch.matmul(projection.T, new_projection))
                            exp_avg = state["exp_avg"]
                            projection = new_projection
                            grad = torch.matmul(grad, projection)

                    else:
                        raise ValueError("Parameters that exceed 2 Dim are not supported currently.")

                else:
                    seed = state["seed"]
                    generator = torch.Generator(device=p.device).manual_seed(seed)

                    if p.dim() < 2:
                        projection = torch.randn(rank, p.shape[0], generator=generator,
                                                 device=p.device) / math.sqrt(rank)
                        exp_avg = state["exp_avg"]
                        grad = torch.matmul(projection, grad)

                    elif p.dim() == 2:
                        if p.shape[0] >= p.shape[1]:
                            projection = torch.randn(rank, p.shape[0], generator=generator,
                                                     device=p.device) / math.sqrt(rank)
                            exp_avg = state["exp_avg"]
                            grad = torch.matmul(projection, grad)

                        else:
                            projection = torch.randn(p.shape[1], rank, generator=generator,
                                                     device=p.device) / math.sqrt(rank)
                            exp_avg = state["exp_avg"]
                            grad = torch.matmul(grad, projection)

                    else:
                        raise ValueError("Parameters that exceed 2 Dim are not supported currently.")

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                beta = group["beta"]
                exp_avg.mul_(beta).add_(grad, alpha=(1.0 - beta))

                # Decompression
                if p.dim() < 2:
                    exp_avg = torch.matmul(projection.T, exp_avg)
                elif p.dim() == 2:
                    if p.shape[0] >= p.shape[1]:
                        exp_avg = torch.matmul(projection.T, exp_avg)
                    else:
                        exp_avg = torch.matmul(exp_avg, projection.T)
                else:
                    raise ValueError("Parameters that exceed 2 Dim are not supported currently.")

                lr = group["lr"]
                step_size = lr
                norm_grad = exp_avg

                p.add_(norm_grad, alpha=-step_size)

        return loss
