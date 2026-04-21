import torch


class StanfordSwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device=None) -> None:
        super().__init__()

        self.d_ff = d_ff or self._get_dff(d_model)
        self.d_model = d_model

        self.device = device or torch.device("cpu")

        self.W1 = torch.nn.Parameter(
            torch.normal(0.0, 1.0, (self.d_ff, self.d_model), device=self.device)
        )
        self.W3 = torch.nn.Parameter(
            torch.normal(0.0, 1.0, (self.d_ff, self.d_model), device=self.device)
        )
        self.W2 = torch.nn.Parameter(
            torch.normal(0.0, 1.0, (self.d_model, self.d_ff), device=self.device)
        )

    def forward(self, x: torch.Tensor):
        return (self._silu(x @ self.W1.T) * (x @ self.W3.T)) @ self.W2.T

    def _get_dff(self, d_model: int) -> int:
        ceiled_dff = (d_model * 8 + 2) // 3
        dff_64 = ((ceiled_dff + 63) // 64) * 64
        return dff_64

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
