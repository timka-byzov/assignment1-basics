import torch
import numpy as np
from cs336_basics.utils.tensor_init import create_linear_W


class StanofordLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.device = device or torch.device("cpu")

        self.W = torch.nn.Parameter(create_linear_W(in_features, out_features, device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
