import torch
from jaxtyping import Float

from cs336_basics.layers.multihead_self_attention import StanfordMHSA
from cs336_basics.layers.positionwise_feedforward import StanfordSwiGLU
from cs336_basics.models import LLModelConfig, TransformerBlockConfig
from cs336_basics.layers.rmsnorm import StanfordRMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerBlockConfig, llm: LLModelConfig):
        super().__init__()

        self.rmsnorm_1 = StanfordRMSNorm(config.rms.eps, llm.d_model)
        self.mhsa = StanfordMHSA(config.mhsa, llm)
        self.rmsnorm_2 = StanfordRMSNorm(config.rms.eps, llm.d_model)
        self.swiglu = StanfordSwiGLU(llm.d_model, config.glu.d_ff)

    def forward(
        self, x: Float[torch.Tensor, "... seq_len d_model"]
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        residual = x
        x = self.rmsnorm_1(x)
        x = self.mhsa(x)
        x += residual

        residual = x
        x = self.rmsnorm_2(x)
        x = self.swiglu(x)
        x += residual

        return x
