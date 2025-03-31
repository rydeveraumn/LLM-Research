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
        values = self.W_value(x)  # (batch, num_tokens, d_out)

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
        # context_vec -> (batch, num_tokens, d_out)
        context_vec = attn_weights @ values

        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()

        # Stack the causal attention layers in
        # module list. This means we can iterate over the
        # causal attention layers for num_heads
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in=d_in,
                    d_out=d_out,
                    context_length=context_length,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x) -> torch.Tensor:
        # mha -> (batch, num_tokens, d_out * num_heads)
        # So we run this multiple times and then concatenate the
        # outputs
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0  # d_out must be divisible by the number of heads

        self.d_out = d_out
        self.num_heads = num_heads

        # Reduces the projection dim to match the desired
        # output dim
        self.head_dim = d_out // num_heads

        # Q, K, V parameters
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Use linear layers to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        # Get the keys, values and queries
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # New reshaping for multi-head attention
        # (batch, num_tokens, num_heads, dim_out // num_heads)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (batch, num_heads, num_tokens, dim_out // num_heads)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # attention scores
        # (batch, num_heads, context_length, context_length)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute the attention weights
        # (batch, num_heads, context_length, context_length)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vector (batch, num_heads, num_tokens, dim_out // num_heads)
        context_vec = attn_weights @ values

        # (batch, num_tokens, num_heads, dim_out // num_heads)
        context_vec = context_vec.transpose(1, 2)

        # (batch, num_tokens / context_length, d_out)
        # contiguous - makes a tensor have contiguous memory
        # layout. When you perform operations like transpose, permute,
        # or certain indexing, PyTorch may create a view of the tensor
        # that jumps around in memory rather than creating a new copy.
        # While efficient for storage, these non-contiguous tensors can't be
        # used with certain operations that expect data to be laid
        # out sequentially in memory.
        # NOTE: Also important to use with view
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # (batch, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
