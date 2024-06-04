from torch import nn
import torch
import itertools
from .shift_loss import ShiftLoss

class ShiftL2Loss(torch.nn.Module):
    """
    Implements the L2 version of the shifted loss
    """
    def __init__(self, ignore_value=None, min_measurements=5, radius=1):
        super(ShiftL2Loss, self).__init__()

        def loss_function(shifted_labels, predictions):
           return torch.pow(shifted_labels - predictions,2)

        self.shift_loss = ShiftLoss(loss_function, ignore_value, min_measurements, radius)

    def forward(self, predictions, labels):
        return self.shift_loss(predictions, labels)
