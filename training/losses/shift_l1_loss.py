from torch import nn
import torch
import itertools
from .shift_loss import ShiftLoss

class ShiftL1Loss(torch.nn.Module):
    """
    Implements the L1 version of the shifted loss
    """
    def __init__(self, ignore_value=None, min_measurements=5, radius=1):
        super(ShiftL1Loss, self).__init__()

        def loss_function(shifted_labels, predictions):
           return torch.abs(shifted_labels - predictions)

        self.shift_loss = ShiftLoss(loss_function, ignore_value, min_measurements, radius)

    def forward(self, predictions, labels):
        return self.shift_loss(predictions, labels)
