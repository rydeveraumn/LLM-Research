# third party
import torch
import torch.nn as nn

# first party
from layers.normalization import LayerNorm
from layers.transformer import TransformerBlock


class GPT2Model(nn.Module):
    """
    Class that implements the GPT2 model from scratch.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        self.emb_dim = cfg["emb_dim"]
        self.context_length = cfg["context_length"]
        self.dropout = cfg["drop_rate"]
        self.n_layers = cfg["n_layers"]

        # Create the token and position embeddings
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.context_length, self.emb_dim)
        self.drop_emb = nn.Dropout(self.dropout)

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg=cfg) for _ in range(self.n_layers)]
        )
        self.final_norm = LayerNorm(self.emb_dim)
        self.out_head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape

        # Map tokens to embeddings
        tok_embeds = self.tok_emb(in_idx)

        # Map positions to embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
