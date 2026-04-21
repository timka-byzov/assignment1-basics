import torch
import numpy as np


class StanofordLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.device = device or torch.device("cpu")

        std = np.sqrt(2 / (out_features + in_features))
        data = torch.zeros((out_features, in_features), device=self.device)
        torch.nn.init.trunc_normal_(data, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W = torch.nn.Parameter(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
