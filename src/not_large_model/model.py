"""Decoder-only transformer (GPT-style) with causal mask."""

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 6,
        seq_len: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T) token indices → logits (B, T, vocab_size)."""
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.tok_emb(x) + self.pos_emb(positions)              # (B, T, d_model)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)                                         # (B, T, vocab_size)

    def save(self, path: str | Path, mod: int) -> None:
        torch.save({"mod": mod, "vocab_size": self.tok_emb.num_embeddings,
                    "d_model": self.tok_emb.embedding_dim, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["TinyTransformer", int]:
        """Returns (model, mod)."""
        data = torch.load(path, weights_only=True)
        model = cls(vocab_size=data["vocab_size"], d_model=data["d_model"])
        model.load_state_dict(data["state_dict"])
        return model, data["mod"]
