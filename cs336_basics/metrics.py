import torch
from jaxtyping import Float


def perplexity(natural_cross_entropy: Float[torch.Tensor, "1"]):
    return torch.exp(natural_cross_entropy)
