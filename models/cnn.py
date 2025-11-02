import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(67)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using:{device}")


# This is how the input has to be in this orderi!!

#   input = (Batch Size, Number of Muscles, Width of time)

# Last dimension is always length, and is the one that'll get pooled, keep in mind if we change data input

# NOTE:
# I decided to not use padding, for sligth performance boost and stricter evaluation

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=6 #number of muscle semg inputs,
                               out_channels=24, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(24)
        # F.maxPool1d will be applied, so take that in count 
        self.conv3 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=9)
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=72, kernel_size=16)
        self.bn2 = nn.BatchNorm1d(72)
        # Another maxPool1d
        # I am assuming we start with a 200ms time step
        self.fc1 = nn.Linear( 37 * 72, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2) # This is reducing the time length by half

        # Second block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2) # This is reducing the time length by half to 50

        # MLP Part
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
