import torch
from jaxtyping import Float
from typing import cast

from cs336_basics.layers.rope import StanfordRoPE
from cs336_basics.config.models import LLModelConfig, MHSAConfig
from cs336_basics.utils.tensor_init import create_linear_W
from cs336_basics.layers.scaled_dot_product_attention import StanfordSDPA


class StanfordMHSA(torch.nn.Module):
    def __init__(self, mhsa: MHSAConfig, d_model: int, max_seq_len: int):
        super().__init__()

        assert d_model % mhsa.num_heads == 0

        self.causal = mhsa.causal
        self.d_model = d_model
        self.d_k = d_model // mhsa.num_heads
        self.num_heads = mhsa.num_heads

        if mhsa.rope_config:
            self.rope = StanfordRoPE(
                mhsa.rope_config.theta, d_model // mhsa.num_heads, max_seq_len
            )
        else:
            self.rope = None

        self.sdpa = StanfordSDPA()

        self.W_out: Float[torch.Tensor, "d_model d_model"] = torch.nn.Parameter(
            create_linear_W(d_model, d_model)
        )
        self.W_q: Float[torch.Tensor, "d_model d_model"] = torch.nn.Parameter(
            create_linear_W(d_model, d_model)
        )
        self.W_k: Float[torch.Tensor, "d_model d_model"] = torch.nn.Parameter(
            create_linear_W(d_model, d_model)
        )
        self.W_v: Float[torch.Tensor, "d_model d_model"] = torch.nn.Parameter(
            create_linear_W(d_model, d_model)
        )

        mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        if mhsa.causal:
            mask = torch.tril(mask)
        mask = mask.view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self, x: Float[torch.Tensor, "... seq_len d_model"], causal: bool = True
    ):
        seq_len = x.shape[-2]

        q = x @ self.W_q.T
        k = x @ self.W_k.T
        v = x @ self.W_v.T

        q_heads = self._split_proj_to_heads(q)
        k_heads = self._split_proj_to_heads(k)
        if self.rope:
            q_heads = self.rope.forward(q_heads, torch.arange(seq_len))
            k_heads = self.rope.forward(k_heads, torch.arange(seq_len))
        v_heads = self._split_proj_to_heads(v)

        headed_attention = self.sdpa.forward(
            q_heads,
            k_heads,
            v_heads,
            cast(torch.Tensor, self.mask)[..., :seq_len, :seq_len],
        )  # (... b h seq_len d_k)
        headed_attention = headed_attention.transpose(-3, -2)  # (... b seq_len h d_k)

        # assume d_v = d_k
        attention = headed_attention.reshape(*x.shape)  # (... b seq_len h * d_k) concat
        return attention @ self.W_out.T

    def _split_proj_to_heads(
        self, x: Float[torch.Tensor, "... seq_len d_model"]
    ) -> Float[torch.Tensor, "... heads seq_len d_k"]:
        return torch.stack(
            [x[..., i * self.d_k : (i + 1) * self.d_k] for i in range(self.num_heads)],
            dim=-3,
        )  # (... h seq_len d_k)
