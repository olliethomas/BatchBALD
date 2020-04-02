from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

import src.mc_dropout


class BayesianNet(src.mc_dropout.BayesianModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = src.mc_dropout.MCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = src.mc_dropout.MCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = src.mc_dropout.MCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    # Special forward pass used in BayesianModule
    def mc_forward_impl(self, input: Tensor) -> Tensor:
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input
