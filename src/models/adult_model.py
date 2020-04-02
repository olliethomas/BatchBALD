from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

import src.mc_dropout


class BayesianNet(src.mc_dropout.BayesianModule):
    def __init__(self, num_classes: int, input_dim: int) -> None:
        super().__init__(num_classes)

        self.layer_1 = nn.Linear(input_dim, 50)
        self.layer_2 = nn.Linear(50, 1)

    # Special forward pass used in BayesianModule
    def mc_forward_impl(self, input: Tensor) -> Tensor:
        input = F.relu(self.layer_1(input))
        input = self.layer_2(input)
        input = F.sigmoid(input)

        return input
