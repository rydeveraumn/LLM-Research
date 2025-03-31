# third party
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Class that compute layer normalization. Layer normalization is
    different from batch normalization in the fact that we normalize
    across the feature dimension.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Take the mean along the feature dimension
        # and keep the same dimensions
        mean = x.mean(dim=-1, keepdim=True)

        # Compute the variance along the feature
        # dimension while keeping the same dimensions
        # unbiased is a distinction between dividing by
        # n or n = 1
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Compute the normalization
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
