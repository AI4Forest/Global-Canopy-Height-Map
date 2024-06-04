from torch import nn
import torch
import itertools
from .shift_loss import ShiftLoss

class ShiftHuberLoss(torch.nn.Module):
    """
    Implements the Huber version of the shifted loss
    """
    def __init__(self, ignore_value=None, min_measurements=5, radius=1, delta=3):
        super(ShiftHuberLoss, self).__init__()

        def loss_function(shifted_labels, predictions):
            residuals_l1 = torch.abs(
                shifted_labels - predictions
            )

            residuals_l2 = torch.pow(
                shifted_labels - predictions, 2
            )

            residuals = torch.where(
                residuals_l1 < delta,
                (residuals_l2 * 0.5)/delta,
                residuals_l1 - (0.5 * delta)
            )

            return residuals

        self.shift_loss = ShiftLoss(loss_function, ignore_value, min_measurements, radius)

    def forward(self, predictions, labels):
        return self.shift_loss(predictions, labels)
