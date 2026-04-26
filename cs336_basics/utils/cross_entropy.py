import torch
from jaxtyping import Float, Int


def get_target_log_probs(
    logits: Float[torch.Tensor, "... seq_len vocab_size"],
    targets: Int[torch.Tensor, "... seq_len"],
):

    safe_logits = logits - logits.max(dim=-1, keepdim=True).values

    target_logits = safe_logits.gather(
        dim=-1, index=targets.unsqueeze(-1)
    )  # (..., seq_len, 1)

    log_sum_exps = safe_logits.exp().sum(dim=-1, keepdim=True).log()

    return log_sum_exps - target_logits
