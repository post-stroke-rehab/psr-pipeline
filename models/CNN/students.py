import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

def scale_channels(base_value: int, scale_factor: float) -> int: 
    """
    Scale number of channels by a factor, ensuring it's at least 1. Used for the different architectures' sizes
    Returns:
        Scaled value, minimum 1
    """
    return max(1, int(base_value * scale_factor))

class CNN_Nano(nn.Module): #Parameters: ~100K 
    def __init__(self, cfg: Config, scale_factor: float = 0.67):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Very small channels for speed
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 16
        fc1_out = scale_channels(cfg.fc1_out, 0.25)              # 256 → 64
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        
        # MLP
        # After 1 pool: 200→198→99
        self.flattened_size = 99 * conv1_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
        
        # For feature distillation
        self.intermediate_features = []
    
    def forward(self, x, return_features=False):
        self.intermediate_features = []
        
        # Single conv block
        x = self.conv1(x)              # (B, 16, 198)
        x = self.bn1(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 16, 99)
        
        # Minimal MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    
class CNN_Micro(nn.Module): #Parameters: ~250k
    def __init__(self, cfg: Config, scale_factor: float = 0.83):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Scale channels moderately
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 20
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)  # 24 → 20
        fc1_out = scale_channels(cfg.fc1_out, 0.5)               # 256 → 128
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # MLP
        # After 1 pool: 200→198→194→97
        self.flattened_size = 97 * conv2_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
        
        self.intermediate_features = []
    
    def forward(self, x, return_features=False):
        self.intermediate_features = []
        
        # Block 1
        x = F.relu(self.conv1(x))      # (B, 20, 198)
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv2(x)              # (B, 20, 194)
        x = self.bn1(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 20, 97)
        
        # MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    
class CNN_Base(nn.Module): #Parameters: ~700k

    def __init__(self, cfg: Config, scale_factor: float = 1.0):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Use base config values directly
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)
        conv3_out = scale_channels(cfg.conv3_out, scale_factor)
        conv4_out = scale_channels(cfg.conv4_out, scale_factor)
        fc1_out = scale_channels(cfg.fc1_out, scale_factor)
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # Block 2
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=cfg.conv3_kernel)
        self.bn2 = nn.BatchNorm1d(conv3_out)
        self.conv4 = nn.Conv1d(conv3_out, conv4_out, kernel_size=cfg.conv4_kernel)
        self.bn3 = nn.BatchNorm1d(conv4_out)
        
        # MLP
        # After 2 pools: 200→198→194→97→89→74→37
        self.flattened_size = 37 * conv4_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
        
        self.intermediate_features = []
    
    def forward(self, x, return_features=False):
        self.intermediate_features = []
        
        # Block 1
        x = F.relu(self.conv1(x))      # (B, 24, 198)
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv2(x)              # (B, 24, 194)
        x = self.bn1(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 24, 97)
        
        # Block 2
        x = self.conv3(x)              # (B, 48, 89)
        x = self.bn2(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv4(x)              # (B, 72, 74)
        x = self.bn3(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 72, 37)
        
        # MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class CNN_Large(nn.Module): #Parameters: ~1.5M

    def __init__(self, cfg: Config, scale_factor: float = 1.33):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Increase channel counts
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 32
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)  # 24 → 32
        conv3_out = scale_channels(cfg.conv3_out, scale_factor)  # 48 → 64
        conv4_out = scale_channels(cfg.conv4_out, scale_factor)  # 72 → 96
        conv5_out = scale_channels(cfg.conv5_out, scale_factor)  # 128 → 170
        fc1_out = scale_channels(cfg.fc2_out, scale_factor)      # 512 → 682
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # Block 2
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=cfg.conv3_kernel)
        self.bn2 = nn.BatchNorm1d(conv3_out)
        self.conv4 = nn.Conv1d(conv3_out, conv4_out, kernel_size=cfg.conv4_kernel)
        self.bn3 = nn.BatchNorm1d(conv4_out)
        
        # Block 3
        self.conv5 = nn.Conv1d(conv4_out, conv5_out, kernel_size=cfg.conv5_kernel)
        self.bn4 = nn.BatchNorm1d(conv5_out)
        
        # MLP
        # After 3 pools: 200→198→194→97→89→74→37→33→16
        self.flattened_size = 16 * conv5_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
        
        self.intermediate_features = []
    
    def forward(self, x, return_features=False):
        self.intermediate_features = []
        
        # Block 1
        x = F.relu(self.conv1(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 3
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class CNN_XLarge(nn.Module): #Parameters: ~4M

    def __init__(self, cfg: Config, scale_factor: float = 2.0):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Maximum channel counts
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 48
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)  # 24 → 48
        conv3_out = scale_channels(cfg.conv3_out, scale_factor)  # 48 → 96
        conv4_out = scale_channels(cfg.conv4_out, scale_factor)  # 72 → 144
        conv5_out = scale_channels(cfg.conv5_out, scale_factor)  # 128 → 256
        
        # Additional layers for XLarge
        conv6_out = scale_channels(192, scale_factor)  # 384
        conv7_out = scale_channels(160, scale_factor)  # 320
        
        fc1_out = scale_channels(cfg.fc2_out, scale_factor * 2)  # 512 → 2048
        fc2_out = scale_channels(cfg.fc1_out, scale_factor)      # 256 → 512
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # Block 2
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=cfg.conv3_kernel)
        self.bn2 = nn.BatchNorm1d(conv3_out)
        self.conv4 = nn.Conv1d(conv3_out, conv4_out, kernel_size=7)  # Smaller kernel for depth
        self.bn3 = nn.BatchNorm1d(conv4_out)
        
        # Block 3
        self.conv5 = nn.Conv1d(conv4_out, conv5_out, kernel_size=cfg.conv5_kernel)
        self.bn4 = nn.BatchNorm1d(conv5_out)
        self.conv6 = nn.Conv1d(conv5_out, conv6_out, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(conv6_out)
        
        # Block 4
        self.conv7 = nn.Conv1d(conv6_out, conv7_out, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(conv7_out)
        
        # MLP
        # After 4 pools: 200→198→194→97→89→82→41→37→32→16→14→7
        self.flattened_size = 4 * conv7_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.fc3 = nn.Linear(fc2_out, cfg.out_dim)
        
        self.intermediate_features = []
    
    def forward(self, x, return_features=False):
        self.intermediate_features = []
        
        # Block 1
        x = F.relu(self.conv1(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 3
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 4
        x = self.conv7(x)
        x = self.bn6(x)
        x = F.relu(x)
        if return_features:
            self.intermediate_features.append(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Deep MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if return_features:
            self.intermediate_features.append(x)
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x
