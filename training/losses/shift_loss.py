from torch import nn
import torch
import itertools


class ShiftLoss(torch.nn.Module):
    """
    Class for the L1 loss function with shifted GEDI labels up to 10 meters in each direction.
    """

    def __init__(self,loss_function, ignore_value=None, min_measurements=5, radius=1):
        """
        Constructor for the L1 loss function with shifted GEDI labels up to 10 meters in each direction.
        :param ignore_value: The value to ignore in the loss calculation.
        """
        super(ShiftLoss, self).__init__()

        self.ignore_value = ignore_value

        self.shift_conv = nn.Conv2d(
            in_channels=1,
            out_channels=((radius*2)+1)**2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.loss_function = loss_function

        self.radius = radius
        self.min_measurements = min_measurements

        directions = list(itertools.product(range(-radius, radius+1), range(-radius, radius+1)))

        weights = torch.zeros(((radius*2)+1)**2, 1, 3, 3)
        for i, (dx, dy) in enumerate(directions):
            weights[i, 0, radius + dy, radius + dx] = 1
        self.shift_conv.weight = nn.Parameter(weights)

        for param in self.shift_conv.parameters():
            param.requires_grad = False

    def forward(self, predictions, labels):
        """
        Forward pass of the loss function.
        :param predictions: The predictions of the network.
        :param labels: The labels of the data.
        :return: The loss.
        """

        # Move to correct device
        self.shift_conv.to(predictions.device)

        # Get all relevant dimensions
        batch, granule, height, width = labels.shape
        labels = labels.unsqueeze(2)
        labels = labels.view(batch * granule, 1, height, width)

        # Apply the shift convolution
        shifted_labels = self.shift_conv(labels)

        shifted_labels = shifted_labels.view(batch, granule, ((self.radius*2)+1)**2, height, width)

        predictions_view = predictions.view(batch, 1, 1, height, width)

        residuals = self.loss_function(shifted_labels, predictions_view)

        residuals[shifted_labels == self.ignore_value] = 0

        residuals_summed = residuals.sum(dim=(3, 4))
        indices = residuals_summed.min(dim=2).indices

        # Set every granule with less than 10 measurements to no shift (center in convolution)
        indices[torch.count_nonzero(labels, dim=(2, 3)).reshape(batch, granule) < self.min_measurements] = 4

        indices_expanded = (
            indices.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, height, width)
        )
        residuals_gathered = (
            torch.gather(residuals, 2, indices_expanded).squeeze().sum(dim=1)
        )
        resis = residuals_gathered[residuals_gathered != 0]

        resis = resis.mean()

        return resis