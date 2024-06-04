import torch.nn as nn
import torch


class L2Loss(nn.Module):
    """Mean Squared error"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function

    def forward(self, out, target):
        """
        Applies the L2 loss
        :param out: output of the network
        :param target: target
        :return L2 loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        out = out.flatten()
        target = target.flatten()

        if self.ignore_value is not None:
            out = out[target != self.ignore_value]
            target = target[target != self.ignore_value]

        loss = (target - out) ** 2

        return loss.mean()
