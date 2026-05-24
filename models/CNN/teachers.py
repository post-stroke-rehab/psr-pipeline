import torch
import torch.nn as nn


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class _ResNet1D(nn.Module):
    def __init__(self, layers, in_channels=768, num_classes=5):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * Bottleneck1D.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * Bottleneck1D.expansion),
            )
        layers = [Bottleneck1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Bottleneck1D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


class ResNet50_1D(_ResNet1D):
    def __init__(self, in_channels: int = 768, num_classes: int = 5):
        super().__init__([3, 4, 6, 3], in_channels, num_classes)


class ResNet101_1D(_ResNet1D):
    def __init__(self, in_channels: int = 768, num_classes: int = 5):
        super().__init__([3, 4, 23, 3], in_channels, num_classes)


class ResNet152_1D(_ResNet1D):
    def __init__(self, in_channels: int = 768, num_classes: int = 5):
        super().__init__([3, 8, 36, 3], in_channels, num_classes)
