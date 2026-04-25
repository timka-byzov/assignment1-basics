import torch

from cs336_basics.config.models import LLModelConfig


class StanfordRMSNorm(torch.nn.Module):
    def __init__(self, eps: float, d_model: int, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.g = torch.nn.Parameter(
            torch.normal(0, 1, (d_model,), device=self.device)
        )  # ???

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # (b, sl, dim) @ (b, dim, sl)
        RMS = torch.sqrt((torch.sum(x * x, dim=2) + self.eps) / self.d_model)  # (b, sl)

        result = (x * self.g) / RMS.unsqueeze(-1)
        # Return the result in the original dtype
        return result.to(in_dtype)
