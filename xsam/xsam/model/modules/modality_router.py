import torch
import torch.nn as nn


class ModalityRouter(nn.Module):
    """Light-weight SAR/optical router. Output g in (0, 1), 1 ~= SAR."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Route with pooled visual tokens."""
        x = tokens.mean(dim=1)
        x = x.to(dtype=self.fc[0].weight.dtype)
        return torch.sigmoid(self.fc(x)).squeeze(-1).float()
