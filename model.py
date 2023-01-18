import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 1),
            nn.Flatten(0, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
