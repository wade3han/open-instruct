import math
import warnings
from typing import Callable, Iterable, Tuple, TypeAlias, Union, Dict, Any

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class LAdamW(Optimizer):
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
            rank: int = 8,
            betas: Tuple[float, float, float] = (0.9, 0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
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
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[2]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias,
                    "rank": rank}
        super().__init__(params, defaults)

        max_size = 0
        for group in self.param_groups:
            for p in group["params"]:
                if max_size < max(p.shape):
                    max_size = max(p.shape)

        self.state['projection'] = torch.randn(max_size * rank)
        self.state['beta0'] = betas[0]
        self.state['step'] = 0
        # self.state['hyperparams'] = (lr, rank, betas[0], betas[1], betas[2], eps, weight_decay, correct_bias)

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

        projection = self.state['projection']
        # lr, rank, beta0, beta1, beta2, eps, weight_decay, correct_bias = self.state['hyperparams']
        beta0 = self.state['beta0']

        projection.mul_(beta0).add_(torch.randn_like(projection), alpha=math.sqrt(1.0 - beta0 ** 2))

        self.state['step'] += 1

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if projection.device != p.device:
                    self.state['projection'] = projection.to(p.device)
                    projection = self.state['projection']

                rank = group["rank"]

                # Compression
                if p.dim() < 2:
                    grad = torch.matmul(projection[:rank * p.shape[0]].view(rank, p.shape[0]) / math.sqrt(rank), grad)  # shape: [rank]
                elif p.dim() == 2:
                    if p.shape[0] >= p.shape[1]:
                        grad = torch.matmul(projection[:rank * p.shape[0]].view(rank, p.shape[0]) / math.sqrt(rank),
                                            grad)
                    else:
                        grad = torch.matmul(grad,
                                            projection[:rank * p.shape[1]].view(p.shape[1], rank) / math.sqrt(rank))
                else:
                    raise ValueError("Parameters that exceed 2 Dim are not supported currently.")

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                beta1, beta2 = group["betas"][1:]
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Decompression
                if p.dim() < 2:
                    exp_avg = torch.matmul(projection[:rank * p.shape[0]].view(rank, p.shape[0]).T / math.sqrt(rank),
                                           exp_avg)
                    exp_avg_sq = torch.matmul(projection[:rank * p.shape[0]].view(rank, p.shape[0]).T / math.sqrt(rank),
                                              exp_avg_sq)
                elif p.dim() == 2:
                    if p.shape[0] >= p.shape[1]:
                        exp_avg = torch.matmul(
                            projection[:rank * p.shape[0]].view(rank, p.shape[0]).T / math.sqrt(rank), exp_avg)
                        exp_avg_sq = torch.matmul(
                            projection[:rank * p.shape[0]].view(rank, p.shape[0]).T / math.sqrt(rank), exp_avg_sq)
                    else:
                        exp_avg = torch.matmul(exp_avg,
                                               projection[:rank * p.shape[1]].view(p.shape[1], rank).T / math.sqrt(
                                                   rank))
                        exp_avg_sq = torch.matmul(exp_avg_sq,
                                                  projection[:rank * p.shape[1]].view(p.shape[1], rank).T / math.sqrt(
                                                      rank))
                else:
                    raise ValueError("Parameters that exceed 2 Dim are not supported currently.")

                lr = group["lr"]
                eps = group["eps"]
                correct_bias = group["correct_bias"]
                step_size = lr
                denom = exp_avg_sq.abs().sqrt().add_(eps)
                step = self.state["step"]
                if correct_bias:
                    bias_correction1 = ((1.0 - beta1) * (1.0 - (beta0 * beta1) ** step)) / (
                            1.0 - (beta0 * beta1))
                    bias_correction2 = ((1.0 - beta2) * (1.0 - (beta0 * beta2) ** step)) / (
                            1.0 - (beta0 * beta2))
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                weight_decay = group["weight_decay"]
                if weight_decay > 0.0:
                    p.add_(p, alpha=(-lr * weight_decay))

        return loss
