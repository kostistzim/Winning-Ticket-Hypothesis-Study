import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    LeNet-300-100 architecture.
    """

    def __init__(self):
        super(LeNet, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(28*28, 300)

        # Hidden layer
        self.fc2 = nn.Linear(300, 100)

        # Output layer (10 output features for digits 0-9)
        self.fc3 = nn.Linear(100, 10)

        self._initialize_weights()

    def forward(self, x):
        # Flatten the image
        x = x.view(-1, 28*28)

        # Input layer + Activation
        x = F.relu(self.fc1(x))

        # Hidden layer + Activation
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)