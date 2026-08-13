import torch
import torch.nn as nn


class SarCondAdapter(nn.Module):
    """Small residual adapter that only adjusts SAR-side cond embeddings."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        # Start from a near-identity mapping so continuing Stage B is stable.
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, cond_embeddings: torch.Tensor) -> torch.Tensor:
        residual = self.mlp(cond_embeddings.to(dtype=self.mlp[0].weight.dtype))
        residual = residual.to(dtype=cond_embeddings.dtype)
        return cond_embeddings + self.residual_scale * residual
