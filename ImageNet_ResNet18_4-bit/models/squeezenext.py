"""
    SqueezeNext for ImageNet-1K, implemented in PyTorch.
    Original paper: 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
"""

__all__ = ['SqueezeNext', 'sqnxt23_w1', 'sqnxt23_w3d2', 'sqnxt23_w2', 'sqnxt23v5_w1', 'sqnxt23v5_w3d2', 'sqnxt23v5_w2']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .quant import QuantizeConv2d, QuantizeLinear


class SqnxtUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,w_bits=4,a_bits=4):
        super(SqnxtUnit, self).__init__()
        if stride == 2:
            reduction_den = 1
            self.resize_identity = True
        elif in_channels > out_channels:
            reduction_den = 4
            self.resize_identity = True
        else:
            reduction_den = 2
            self.resize_identity = False

        self.conv1 = QuantizeConv2d(
            in_channels=in_channels,
            out_channels=(in_channels // reduction_den),
            stride=stride,kernel_size=1,
            bias=True,w_bits=w_bits,a_bits=a_bits)
        self.conv2 = QuantizeConv2d(
            in_channels=(in_channels // reduction_den),
            out_channels=(in_channels // (2 * reduction_den)),kernel_size=1,
            bias=True,w_bits=w_bits,a_bits=a_bits)
        self.conv3 = QuantizeConv2d(
            in_channels=(in_channels // (2 * reduction_den)),
            out_channels=(in_channels // reduction_den),
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            bias=True,w_bits=w_bits,a_bits=a_bits)
        self.conv4 = QuantizeConv2d(
            in_channels=(in_channels // reduction_den),
            out_channels=(in_channels // reduction_den),
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            bias=True,w_bits=w_bits,a_bits=a_bits)
        self.conv5 = QuantizeConv2d(
            in_channels=(in_channels // reduction_den),
            out_channels=out_channels,kernel_size=1,
            bias=True,w_bits=w_bits,a_bits=a_bits)

        if self.resize_identity:
            self.identity_conv = QuantizeConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=True,w_bits=w_bits,a_bits=a_bits)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + identity
        x = self.activ(x)
        return x


class SqnxtInitBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,w_bits=4,a_bits=4):
        super(SqnxtInitBlock, self).__init__()
        self.conv = QuantizeConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=1,
            bias=True,w_bits=w_bits,a_bits=a_bits)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class SqueezeNext(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000,w_bits=4,a_bits=4):
        super(SqueezeNext, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", SqnxtInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SqnxtUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", QuantizeConv2d(
            in_channels=in_channels,
            out_channels=final_block_channels,
            bias=True))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = QuantizeLinear(
            in_features=in_channels,
            out_features=num_classes,w_bits=w_bits,a_bits=a_bits)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_squeezenext(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".torch", "models"),
                    **kwargs):
    """
    Create SqueezeNext model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('23' or '23v5').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 64
    final_block_channels = 128
    channels_per_layers = [32, 64, 128, 256]

    if version == '23':
        layers = [6, 6, 8, 1]
    elif version == '23v5':
        layers = [2, 4, 14, 1]
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        final_block_channels = int(final_block_channels * width_scale)

    net = SqueezeNext(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)

    return net

def sqnxt23_w2():
  return get_squeezenext(version="23", width_scale=2.0, model_name="sqnxt23_w2", **kwargs)

def test():
    net = sqnxt23_w2()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
        


