from bisect import bisect_right

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random

class GeneralUtility:
    """General helper functions"""
    @staticmethod
    def fill_dict_with_none(d):
        for key in d:
            if isinstance(d[key], dict):
                GeneralUtility.fill_dict_with_none(d[key])  # Recursive call for nested dictionaries
            else:
                d[key] = None
        return d
    
    @staticmethod
    def update_config_with_default(configDict, defaultDict):
        """Update config with default values recursively."""
        for key, default_value in defaultDict.items():
            if key not in configDict:
                configDict[key] = default_value
            elif isinstance(default_value, dict):
                configDict[key] = GeneralUtility.update_config_with_default(configDict.get(key, {}), default_value)
        return configDict
    

class JointRandomRotationTransform:
    def __init__(self):
        self.angles = [90, 180, 270, 360]

    def __call__(self, image, label):
        angle = random.choice(self.angles)
        if angle == 360:
            return image, label
        return F.rotate(image, angle), F.rotate(label, angle)


class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()