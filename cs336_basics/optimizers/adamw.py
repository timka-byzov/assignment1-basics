import torch
from typing import Optional, Callable
import numpy as np

karpathy_constant: float = 3e-4


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=karpathy_constant,
        betas=(0.9, 0.999),
        weight_decay=0.1,
        eps=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        beta1, beta2, weight_decay, eps = (
            self.defaults["beta1"],
            self.defaults["beta2"],
            self.defaults["weight_decay"],
            self.defaults["eps"],
        )

        for group in self.param_groups:

            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # get state associated with p.
                t = state.get("t", 1)
                lr_t = (
                    lr * np.sqrt((1 - np.pow(beta2, t))) / (1 - np.pow(beta1, t))
                )  # bias correction

                p.data -= lr * weight_decay * p.data  # regularization

                grad: torch.Tensor = (
                    p.grad.data
                )  # Get the gradient of loss with respect to p.

                exp_avg: torch.Tensor = state.get("exp_avg", torch.zeros_like(p.data))
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad

                exp_avg_sq: torch.Tensor = state.get(
                    "exp_avg_sq", torch.zeros_like(p.data)
                )
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                p.data -= lr_t * exp_avg / (exp_avg_sq.sqrt() + eps)

                state["t"] = t + 1

        return loss
