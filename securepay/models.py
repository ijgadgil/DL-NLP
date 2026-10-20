"""Sequential neural networks (LSTM / GRU) and focal loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceFraudNet(nn.Module):
    """Recurrent encoder over a user's last T transactions + MLP head.

    The final step of the sequence is the transaction being scored, so the
    network learns how *this* transaction deviates from the user's recent
    behaviour (amount, frequency, timing, location, device ...).
    An attention-pooled context vector is concatenated with the last hidden
    state so the head sees both "what just happened" and "the usual pattern".
    """

    def __init__(self, n_features: int, cell: str = "lstm", hidden: int = 64,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell.lower()]
        self.cell = cell.lower()
        self.input_proj = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.rnn = rnn_cls(hidden, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F) ; last channel = present mask
        mask = x[..., -1] > 0.5
        out, _ = self.rnn(self.input_proj(x))
        last = out[:, -1]
        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e9)
        ctx = (torch.softmax(scores, dim=1).unsqueeze(-1) * out).sum(1)
        return self.head(torch.cat([last, ctx], dim=-1)).squeeze(-1)  # logits


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) on logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights the flood of easy legitimate transactions so training focuses
    on hard, rare fraud examples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()
