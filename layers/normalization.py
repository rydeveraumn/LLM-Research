# third party
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()

    def forward(self, x):
        return x
