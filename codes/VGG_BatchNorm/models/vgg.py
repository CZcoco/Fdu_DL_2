"""
VGG model variants for CIFAR-10
"""
import sys
from pathlib import Path

import numpy as np
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import init_weights_


def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return parameters_n


def get_activation(name):
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choices: {sorted(activations)}")
    return activations[name]


class VGG_A(nn.Module):
    """VGG_A model (32x32 input)"""

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True, activation="relu"):
        super().__init__()
        Act = get_activation(activation)

        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), Act(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), Act(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), Act(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512), Act(inplace=True),
            nn.Linear(512, 512), Act(inplace=True),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGG_A_BatchNorm(nn.Module):
    """VGG_A with BatchNorm2d after every conv layer."""

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True, activation="relu"):
        super().__init__()
        Act = get_activation(activation)

        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), nn.BatchNorm2d(64), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), Act(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), Act(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), Act(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), Act(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512), Act(inplace=True),
            nn.Linear(512, 512), Act(inplace=True),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGG_A_Light(nn.Module):
    """Lightweight 2-stage VGG variant."""

    def __init__(self, inp_ch=3, num_classes=10, activation="relu"):
        super().__init__()
        Act = get_activation(activation)

        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 16, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128), Act(inplace=True),
            nn.Linear(128, 128), Act(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 32 * 8 * 8))
        return x


class VGG_A_Dropout(nn.Module):
    """VGG_A with Dropout in classifier."""

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True, activation="relu"):
        super().__init__()
        Act = get_activation(activation)

        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), Act(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), Act(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), Act(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), Act(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512), Act(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512), Act(inplace=True),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class ResBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(self, in_ch, out_ch, stride=1, activation="relu"):
        super().__init__()
        Act = get_activation(activation)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = Act(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = Act(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet_Small(nn.Module):
    """Small ResNet for CIFAR-10 (ResNet-18 style, fewer filters)."""

    def __init__(self, inp_ch=3, num_classes=10, activation="relu"):
        super().__init__()
        Act = get_activation(activation)

        self.prep = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Act(inplace=True),
        )

        self.layer1 = nn.Sequential(ResBlock(64, 64, activation=activation), ResBlock(64, 64, activation=activation))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2, activation=activation), ResBlock(128, 128, activation=activation))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2, activation=activation), ResBlock(256, 256, activation=activation))
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2, activation=activation), ResBlock(512, 512, activation=activation))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


if __name__ == '__main__':
    print(f"VGG_A: {get_number_of_parameters(VGG_A())}")
    print(f"VGG_A_BatchNorm: {get_number_of_parameters(VGG_A_BatchNorm())}")
    print(f"VGG_A_Light: {get_number_of_parameters(VGG_A_Light())}")
    print(f"VGG_A_Dropout: {get_number_of_parameters(VGG_A_Dropout())}")
    print(f"ResNet_Small: {get_number_of_parameters(ResNet_Small())}")
