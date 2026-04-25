import torch
from jaxtyping import Float, Int

from cs336_basics.layers.softmax import StanfordSoftMax
import einops


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: Float[torch.Tensor, "... seq_len vocab_size"],
        targets: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "1"]:

        safe_logits = logits - logits.max(dim=-1, keepdim=True).values

        target_logits = safe_logits.gather(
            dim=-1, index=targets.unsqueeze(-1)
        )  # (..., seq_len, 1)

        log_sum_exps = safe_logits.exp().sum(dim=-1, keepdim=True).log()

        return einops.reduce(log_sum_exps - target_logits, "... 1 -> 1", "mean")
