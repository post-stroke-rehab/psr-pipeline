import torch
import torch.nn as nn
import torch.nn.functional as F

layer50 = [3, 4, 6, 3]
layer101 = [3, 4, 23, 3]
layer152 = [3, 8, 36, 3]


class Bottleneck1D(nn.Module):
    """1D Bottleneck Block for ResNet50/101/152"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1x1 conv (dimensionality reduction)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 3x3 conv 
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 1x1 conv (dimensionality expansion)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetGeneric1D(nn.Module):
    """
    A flexible 1D ResNet that can become ResNet-50, 101, or 152 
    based on the 'layers' argument.
    """
    def __init__(self, block:Bottleneck1d, layers:List[int], input_channels=1, num_classes=1):
        super().__init__()
        self.in_channels = 64
        
        # Initial Stage
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers (The 'layers' list determines 50, 101, or 152)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Output Stage
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def forward(self, x, return_features=False):
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if return_features: features.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if return_features: features.append(x)
        
        x = self.layer2(x)
        if return_features: features.append(x)
        
        x = self.layer3(x)
        if return_features: features.append(x)
        
        x = self.layer4(x)
        if return_features: features.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if return_features: features.append(x)
        
        # Return raw logits for proper Knowledge Distillation
        logits = self.fc(x)

        if return_features:
            return logits, features
        return logits
