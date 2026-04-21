import torch
from jaxtyping import Float, Bool
from cs336_basics.layers.softmax import StanfordSoftMax


class StanfordAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... keys d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_v"]:

        d_k = Q.new_tensor(float(Q.shape[-1]))

        if mask is not None:
            add_mask = torch.zeros_like(mask, dtype=Q.dtype)
            add_mask[~mask] = -torch.inf
        else:
            add_mask = torch.zeros(Q.shape[-2], K.shape[-2], device=Q.device)

        w = (Q @ K.transpose(-1, -2)) / d_k.sqrt()  # seq_len, seq_len
        masked_w = StanfordSoftMax()(add_mask + w)

        return masked_w @ V
