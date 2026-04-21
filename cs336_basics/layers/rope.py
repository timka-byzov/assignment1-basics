# R_q.T @ R_k = R_{k - q}

import torch
from jaxtyping import Float, Int
from typing import cast


class StanfordRoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        assert d_k % 2 == 0

        self.Theta = theta
        self.d_k = d_k  # dimension of query and key vectors
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")

        self._create_buffers()

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> torch.Tensor:  #   3.4.2 в RoFormer

        rot_sin = cast(torch.Tensor, self.sin_table)[token_positions].to(
            x.device, x.dtype
        )  # (..., seq_len, d_k // 2)
        rot_cos = cast(torch.Tensor, self.cos_table)[token_positions].to(
            x.device, x.dtype
        )  # (..., seq_len, d_k // 2)

        rot_sin2 = torch.repeat_interleave(rot_sin, 2, dim=-1)  # (..., seq_len, d_k)
        rot_cos2 = torch.repeat_interleave(rot_cos, 2, dim=-1)  # (..., seq_len, d_k)

        return (
            x * rot_cos2
            + x[..., cast(torch.Tensor, self.x_permute).to(x.device)]
            * cast(torch.Tensor, self.sgn).to(x.device)
            * rot_sin2
        )

    def _create_buffers(self) -> None:
        i = torch.arange(self.max_seq_len).reshape(-1, 1)
        j = torch.arange(self.d_k // 2)
        divisors = self.Theta ** (2 * j / self.d_k)
        angles = i / divisors

        self.register_buffer("sin_table", torch.sin(angles), persistent=False)
        self.register_buffer("cos_table", torch.cos(angles), persistent=False)

        sgn = (torch.arange(self.d_k, dtype=torch.long) % 2) * 2 - 1  # (d_k,)
        self.register_buffer("sgn", sgn, persistent=False)

        x_permute = (
            torch.stack(
                (
                    torch.arange(1, self.d_k, 2, dtype=torch.long),
                    torch.arange(0, self.d_k, 2, dtype=torch.long),
                ),
            )
            .transpose(-1, -2)
            .flatten()
        )  # (d_k,) меняем попарно активации местами
        self.register_buffer("x_permute", x_permute, persistent=False)
