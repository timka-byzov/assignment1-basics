import torch
from typing import Iterable


def gradient_global_clip_(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
) -> None:

    grads = [param.grad.data for param in parameters if param.grad is not None]
    flatten_grads = torch.cat(grads).flatten()
    l2_norm = (flatten_grads * flatten_grads).sum().sqrt()

    if l2_norm > max_l2_norm:
        for grad in grads:
            grad.mul_(max_l2_norm / (l2_norm + eps))
