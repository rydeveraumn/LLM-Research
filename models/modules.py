# third party
import torch
import torch.nn as nn

# first party
from models.gpt2 import GPT2Model


class GPTModule(nn.Module):
    """
    Class that will serve as an alternative for
    Pytorch Lightning.
    """

    def __init__(self, model: GPT2Model) -> None:
        self.model = model

    def forward(self, inputs: torch.Tensor):
        logits = self.model(inputs)
        return logits

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        For each module we will have its own loss calculation.
        """
        # Reshape the inputs
        y_pred = y_pred.flatten(0, 1)
        y_true = y_true.flatten()

        # Compute the cross entropy loss
        loss = nn.functional.cross_entropy(y_pred, y_true)

        return loss
