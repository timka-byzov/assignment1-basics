import torch
from jaxtyping import Float, Int

from cs336_basics.layers.softmax import StanfordSoftMax
import einops

from cs336_basics.utils.cross_entropy import get_target_log_probs


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: Float[torch.Tensor, "... seq_len vocab_size"],
        targets: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "1"]:

        return einops.reduce(
            get_target_log_probs(logits, targets), "... 1 -> 1", "mean"
        )
