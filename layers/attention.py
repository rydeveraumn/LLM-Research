# third party
import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    """
    Class that implements causal attention
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_out = d_out

        # Set up linear weight parameters
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Setup for the remaining mechanisms of attention
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x) -> torch.Tensor:
        # So this is (batch, number of tokens, input dimension)
        _, num_tokens, _ = x.shape

        # Set up the query, keys and values
        keys = self.W_key(x)  # (batch, num_tokens, d_out)
        queries = self.W_query(x)  # (batch, num_tokens, d_out)
        values = self.W_values(x)  # (batch, num_tokens, d_out)

        # Compute the attention scores
        # As an example keys & queries will have dimension
        # (batch, num_tokens, d_out)
        # keys.transpose(1, 2) -> (batch, d_out, num_tokens)
        # attn_scores -> (batch, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(1, 2)

        # Mask out attention values above the diagonal to prevent
        # leakage
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf,  # fill with inf so we can perform a mathematical trick
        )

        # Compute the attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the context vector
        context_vec = attn_weights @ values

        return context_vec
