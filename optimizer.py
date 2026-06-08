from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # state 초기화 (처음 호출될 때만)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)      # 1차 모멘트 m
                    state["exp_avg_sq"] = torch.zeros_like(p.data)   # 2차 모멘트 v

                state["step"] += 1
                beta1, beta2 = group["betas"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # 1. 모멘트 업데이트
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 2. Bias correction
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = alpha * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = alpha

                # 3. 파라미터 업데이트
                p.data.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-step_size)

                # 4. Weight decay (gradient 업데이트 이후 적용)
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-alpha * group["weight_decay"])

        return loss
