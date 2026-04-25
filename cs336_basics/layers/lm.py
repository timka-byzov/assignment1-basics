import torch
from jaxtyping import Int, Float

from cs336_basics.layers.linear import StanofordLinear
from cs336_basics.config.models import LLModelConfig, TransformerBlockConfig
from cs336_basics.layers.transformer_block import TransformerBlock
from cs336_basics.layers.embedding import StanfordEmbedding
from cs336_basics.layers.rmsnorm import StanfordRMSNorm


class StanfordLM(torch.nn.Module):
    def __init__(self, block: TransformerBlockConfig, llm: LLModelConfig):
        super().__init__()

        self.embedding = StanfordEmbedding(
            num_embeddings=int(llm.vocab_size), embedding_dim=llm.d_model
        )
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    config=block, d_model=llm.d_model, max_seq_len=llm.context_length
                )
                for _ in range(llm.num_layers)
            ]
        )
        self.ln_final = StanfordRMSNorm(block.rms.eps, llm.d_model)
        self.lm_head = StanofordLinear(llm.d_model, llm.vocab_size)
        self.config = llm

    def forward(
        self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        x = self.embedding(in_indices)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)
