import torch
import torch.nn as nn
import torch.nn.functional as F


def _input_norm(in_channels: int, in_norm: bool):
    return nn.InstanceNorm1d(in_channels, affine=True) if in_norm else nn.Identity()


def _final_pool(pool_type: str, pool: int):
    return nn.AdaptiveMaxPool1d(pool) if pool_type == "max" else nn.AdaptiveAvgPool1d(pool)


class CNN_Nano(nn.Module):
    def __init__(self, in_channels: int = 768, num_classes: int = 5, in_norm: bool = False,
                 proj_ch: int = 32,
                 conv1_ch: int = 48, conv1_k: int = 3,
                 pool: int = 8, pool_type: str = "max", fc1: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_norm = _input_norm(in_channels, in_norm)
        self.proj = nn.Conv1d(in_channels, proj_ch, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(proj_ch)
        self.conv1 = nn.Conv1d(proj_ch, conv1_ch, kernel_size=conv1_k, padding=conv1_k // 2)
        self.bn1 = nn.BatchNorm1d(conv1_ch)
        self.pool = _final_pool(pool_type, pool)
        self.fc1 = nn.Linear(conv1_ch * pool, fc1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1, num_classes)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(self.bn0(self.proj(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNN_Micro(nn.Module):
    def __init__(self, in_channels: int = 768, num_classes: int = 5, in_norm: bool = False,
                 proj_ch: int = 48,
                 conv1_ch: int = 64, conv1_k: int = 5,
                 conv2_ch: int = 96, conv2_k: int = 5,
                 pool: int = 8, pool_type: str = "max", fc1: int = 96, dropout: float = 0.2):
        super().__init__()
        self.in_norm = _input_norm(in_channels, in_norm)
        self.proj = nn.Conv1d(in_channels, proj_ch, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(proj_ch)
        self.conv1 = nn.Conv1d(proj_ch, conv1_ch, kernel_size=conv1_k, padding=conv1_k // 2)
        self.bn1 = nn.BatchNorm1d(conv1_ch)
        self.conv2 = nn.Conv1d(conv1_ch, conv2_ch, kernel_size=conv2_k, padding=conv2_k // 2)
        self.bn2 = nn.BatchNorm1d(conv2_ch)
        self.pool = _final_pool(pool_type, pool)
        self.fc1 = nn.Linear(conv2_ch * pool, fc1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1, num_classes)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(self.bn0(self.proj(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNN_Base(nn.Module):
    def __init__(self, in_channels: int = 768, num_classes: int = 5, in_norm: bool = False,
                 proj_ch: int = 64,
                 conv1_ch: int = 96, conv1_k: int = 3,
                 conv2_ch: int = 128, conv2_k: int = 3,
                 conv3_ch: int = 192, conv3_k: int = 3,
                 conv4_ch: int = 192, conv4_k: int = 3,
                 pool: int = 4, pool_type: str = "max", fc1: int = 128, dropout: float = 0.3):
        super().__init__()
        self.in_norm = _input_norm(in_channels, in_norm)
        self.proj = nn.Conv1d(in_channels, proj_ch, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(proj_ch)
        self.conv1 = nn.Conv1d(proj_ch, conv1_ch, kernel_size=conv1_k, padding=conv1_k // 2)
        self.bn1 = nn.BatchNorm1d(conv1_ch)
        self.conv2 = nn.Conv1d(conv1_ch, conv2_ch, kernel_size=conv2_k, padding=conv2_k // 2)
        self.bn2 = nn.BatchNorm1d(conv2_ch)
        self.conv3 = nn.Conv1d(conv2_ch, conv3_ch, kernel_size=conv3_k, padding=conv3_k // 2)
        self.bn3 = nn.BatchNorm1d(conv3_ch)
        self.conv4 = nn.Conv1d(conv3_ch, conv4_ch, kernel_size=conv4_k, padding=conv4_k // 2)
        self.bn4 = nn.BatchNorm1d(conv4_ch)
        self.pool = _final_pool(pool_type, pool)
        self.fc1 = nn.Linear(conv4_ch * pool, fc1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1, num_classes)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(self.bn0(self.proj(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNN_Large(nn.Module):
    def __init__(self, in_channels: int = 768, num_classes: int = 5, in_norm: bool = False,
                 proj_ch: int = 96,
                 conv1_ch: int = 128, conv1_k: int = 5,
                 conv2_ch: int = 192, conv2_k: int = 5,
                 conv3_ch: int = 256, conv3_k: int = 3,
                 conv4_ch: int = 256, conv4_k: int = 3,
                 pool: int = 4, pool_type: str = "max", fc1: int = 192, dropout: float = 0.3):
        super().__init__()
        self.in_norm = _input_norm(in_channels, in_norm)
        self.proj = nn.Conv1d(in_channels, proj_ch, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(proj_ch)
        self.conv1 = nn.Conv1d(proj_ch, conv1_ch, kernel_size=conv1_k, padding=conv1_k // 2)
        self.bn1 = nn.BatchNorm1d(conv1_ch)
        self.conv2 = nn.Conv1d(conv1_ch, conv2_ch, kernel_size=conv2_k, padding=conv2_k // 2)
        self.bn2 = nn.BatchNorm1d(conv2_ch)
        self.conv3 = nn.Conv1d(conv2_ch, conv3_ch, kernel_size=conv3_k, padding=conv3_k // 2)
        self.bn3 = nn.BatchNorm1d(conv3_ch)
        self.conv4 = nn.Conv1d(conv3_ch, conv4_ch, kernel_size=conv4_k, padding=conv4_k // 2)
        self.bn4 = nn.BatchNorm1d(conv4_ch)
        self.pool = _final_pool(pool_type, pool)
        self.fc1 = nn.Linear(conv4_ch * pool, fc1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1, num_classes)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(self.bn0(self.proj(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNN_XLarge(nn.Module):
    def __init__(self, in_channels: int = 768, num_classes: int = 5, in_norm: bool = False,
                 proj_ch: int = 128,
                 conv1_ch: int = 192, conv1_k: int = 5,
                 conv2_ch: int = 256, conv2_k: int = 5,
                 conv3_ch: int = 384, conv3_k: int = 3,
                 conv4_ch: int = 512, conv4_k: int = 3,
                 pool: int = 2, pool_type: str = "max", fc1: int = 256, dropout: float = 0.4):
        super().__init__()
        self.in_norm = _input_norm(in_channels, in_norm)
        self.proj = nn.Conv1d(in_channels, proj_ch, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(proj_ch)
        self.conv1 = nn.Conv1d(proj_ch, conv1_ch, kernel_size=conv1_k, padding=conv1_k // 2)
        self.bn1 = nn.BatchNorm1d(conv1_ch)
        self.conv2 = nn.Conv1d(conv1_ch, conv2_ch, kernel_size=conv2_k, padding=conv2_k // 2)
        self.bn2 = nn.BatchNorm1d(conv2_ch)
        self.conv3 = nn.Conv1d(conv2_ch, conv3_ch, kernel_size=conv3_k, padding=conv3_k // 2)
        self.bn3 = nn.BatchNorm1d(conv3_ch)
        self.conv4 = nn.Conv1d(conv3_ch, conv4_ch, kernel_size=conv4_k, padding=conv4_k // 2)
        self.bn4 = nn.BatchNorm1d(conv4_ch)
        self.pool = _final_pool(pool_type, pool)
        self.fc1 = nn.Linear(conv4_ch * pool, fc1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1, num_classes)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(self.bn0(self.proj(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
