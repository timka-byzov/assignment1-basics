import numpy as np
import torch


def create_linear_W(in_features: int, out_features: int, device: str | None = None):

    std = np.sqrt(2 / (out_features + in_features))
    data = torch.zeros((out_features, in_features), device=device or "cpu")
    torch.nn.init.trunc_normal_(data, mean=0, std=std, a=-3 * std, b=3 * std)

    return data
