import torch


class StanfordEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        data = torch.zeros((num_embeddings, embedding_dim), device=self.device)
        torch.nn.init.trunc_normal_(data, mean=0, std=1, a=-3, b=3)
        self.E = torch.nn.Parameter(data=data)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.E[token_ids]
