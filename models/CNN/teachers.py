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
    # stem_stride/stem_pool/strides control how much the length axis is downsampled.
    # Default is low-downsampling (/8, deep layers see length ~5 not ~2): the ~39-step
    # EMG sequence is short, so the standard ResNet /32 wastes most of the depth.
    # For a stock ResNet pass stem_stride=2, stem_pool=True. dropout regularizes the head.
    def __init__(self, layers, in_channels=768, num_classes=5, in_norm=False,
                 stem_stride=1, stem_pool=False, strides=(1, 2, 2, 2), dropout=0.0):
        super().__init__()
        self.arch_kwargs = {"in_channels": in_channels, "num_classes": num_classes,
                            "in_norm": in_norm, "stem_stride": stem_stride,
                            "stem_pool": stem_pool, "strides": tuple(strides), "dropout": dropout}
        self.in_norm = nn.InstanceNorm1d(in_channels, affine=True) if in_norm else nn.Identity()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=stem_stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if stem_pool else nn.Identity()
        self.layer1 = self._make_layer(64, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
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
        x = self.in_norm(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.dropout(x)
        return self.fc(x)


class ResNet50_1D(_ResNet1D):
    def __init__(self, in_channels=768, num_classes=5, in_norm=False,
                 stem_stride=1, stem_pool=False, strides=(1, 2, 2, 2), dropout=0.0):
        super().__init__([3, 4, 6, 3], in_channels, num_classes, in_norm,
                         stem_stride, stem_pool, strides, dropout)


class ResNet101_1D(_ResNet1D):
    def __init__(self, in_channels=768, num_classes=5, in_norm=False,
                 stem_stride=1, stem_pool=False, strides=(1, 2, 2, 2), dropout=0.0):
        super().__init__([3, 4, 23, 3], in_channels, num_classes, in_norm,
                         stem_stride, stem_pool, strides, dropout)


class ResNet152_1D(_ResNet1D):
    def __init__(self, in_channels=768, num_classes=5, in_norm=False,
                 stem_stride=1, stem_pool=False, strides=(1, 2, 2, 2), dropout=0.0):
        super().__init__([3, 8, 36, 3], in_channels, num_classes, in_norm,
                         stem_stride, stem_pool, strides, dropout)
