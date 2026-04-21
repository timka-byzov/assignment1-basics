import torch


class StanfordSoftMax(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int):
        dim_max = x.max(dim=dim, keepdim=True).values
        safe_x = x - dim_max

        exps = torch.exp(safe_x)

        return exps / exps.sum(dim=dim, keepdim=True)
