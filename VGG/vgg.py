from __future__ import annotations  # noqa: D100

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class Conv(nn.Module):
    """Single cnovulution block for vgg."""

    def __init__(self, in_filters:int, out_filters:int) -> None:
        """Intialisation of class.

        Args:
            in_filters (_type_): Number of channels in the input image
            out_filters (_type_): Number of channels produced by the convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """Defing forward pass."""
        return F.relu(self.bn(self.conv(x)))


class VGG(nn.Module):
    """VGG model."""

    def __init__(self,cfgs:list | None=None,outputs:int=10,init_weights:bool=True)->None:
        """Intialising vgg vlass.

        Args:
            cfgs (list | None, optional): configration of model. If none is passed it will be defaulted to VGG19.
            outputs: output of vgg.
            init_weights: to intialise weights
        """
        super().__init__()
        if cfgs is None:
            cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]

        layers=[]
        filters=3

        for spec in cfgs:
            if spec=='M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Conv(filters,spec))
                filters=spec

        self.layers=nn.Sequential(*layers)
        self.fc=nn.Linear(512,outputs)
        self.criterion=nn.CrossEntropyLoss()

        if init_weights:
            self._initialize_weights()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Forward pass of VGG19."""
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self)->None:
        """Initialises weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = VGG()
    for m in model.modules():
        print(m)


