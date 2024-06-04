import torch.nn as nn
import torch


class L1Loss(nn.Module):
    """Mean Absolute error"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
        lower_threshold=None,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function
        self.lower_threshold = lower_threshold or 0.

    def forward(self, out, target):
        """
        Applies the L1 loss
        :param out: output of the network
        :param target: target
        :return: l1 loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        out = out.flatten()
        target = target.flatten()

        if self.ignore_value is not None:
            out = out[target != self.ignore_value]
            target = target[target != self.ignore_value]

        if self.lower_threshold > 0:
            out = out[target > self.lower_threshold]
            target = target[target > self.lower_threshold]

        loss = torch.abs(target - out)

        return loss.mean()
