import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # option A shortcut when downsampling (no learnable params)
        self._use_option_a = stride != 1 or in_ch != out_ch
        self._pad = out_ch - in_ch

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self._use_option_a:
            shortcut = x[:, :, ::2, ::2]
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self._pad))
        else:
            shortcut = x
        return F.relu(out + shortcut)


class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.block1 = BasicBlock(16, 16, stride=1)
        self.block2 = BasicBlock(16, 32, stride=2)
        self.block3 = BasicBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
