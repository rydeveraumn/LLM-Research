# third party
import torch.nn as nn

# first party
from layers.attention import MultiHeadAttention
from layers.linear import FeedForward
from layers.normalization import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        # initialize variables
        self.d_in = self.d_out = self.emb_dim = cfg["emb_dim"]
        self.context_length = cfg["context_length"]
        self.num_heads = cfg["n_heads"]
        self.drop_rate = cfg["drop_rate"]
        self.qkv_bias = cfg["qkv_bias"]

        self.attn = MultiHeadAttention(
            d_in=self.d_in,
            d_out=self.d_out,
            context_length=self.context_length,
            num_heads=self.num_heads,
            dropout=self.drop_rate,
            qkv_bias=self.qkv_bias,
        )
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(emb_dim=self.emb_dim)
        self.norm2 = LayerNorm(emb_dim=self.emb_dim)
        self.drop_shortcut = nn.Dropout(self.drop_rate)

    def forward(self, x):
        # Create the short cut connection for the
        # attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)  # (batch, num_tokens, emb_size)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Skip connection

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
