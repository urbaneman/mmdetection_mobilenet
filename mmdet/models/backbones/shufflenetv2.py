import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

def conv_bn(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )



def conv_dw(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, pad, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class CEM(nn.Module):
    """Context Enhancement Module"""

    def __init__(self, in_channels1, in_channels2 ,in_channels3 ,backone, kernel_size=1, stride=1):
        super(CEM, self).__init__()
        self.backone  = backone
        self.conv4 = nn.Conv2d(in_channels1, CEM_FILTER, kernel_size, bias=True)
        self.conv5 = nn.Conv2d(in_channels2, CEM_FILTER, kernel_size, bias=True)
        self.convlast = nn.Conv2d(in_channels3, CEM_FILTER, kernel_size, bias=True)
        self.unsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,x):
        # in keras NHWC
        # in torch NCHW

        inputs = self.backone(x)
        C4_lat = self.conv4(inputs[0])
        C5_lat = self.conv5(inputs[1])
        C5_lat = self.unsample(C5_lat)
        Cglb_lat = self.convlast(inputs[2])

        return C4_lat + C5_lat + Cglb_lat


class BasicBlock(nn.Module):
    def __init__(self, in_channels, shuffle_groups=2):
        super().__init__()
        channels = in_channels // 2
        self.channels = channels
        self.conv1 = conv_bn(
            channels, channels, kernel_size=1,stride=1, pad= 0
        )
        self.conv2 = conv_dw(
            channels, channels, kernel_size=5,  stride=1, pad= 2
        )
        self.conv3 = conv_bn(
            channels, channels, kernel_size=1,stride=1, pad= 0
        )

        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x = x.contiguous()
        c = x.size(1) // 2

        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        # if self.with_se:
        #     x2 = self.se(x2)
        # print(x1.shape)
        # print(x2.shape)

        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2, **kwargs):
        super().__init__()
        channels = out_channels // 2
        self.conv11 = conv_dw(
            in_channels, in_channels, kernel_size=5, stride=2, pad= 2
        )
        self.conv12 = conv_bn(
            in_channels, channels, kernel_size=1, stride=1, pad= 0
        )
        self.conv21 = conv_bn(
            in_channels, channels, kernel_size=1, stride=1, pad= 0
        )
        self.conv22 = conv_dw(
            channels, channels, kernel_size=5, stride=2, pad= 2
        )
        self.conv23 = conv_bn(
            channels, channels, kernel_size=1,stride=1, pad= 0
        )
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x1 = self.conv11(x)

        x1 = self.conv12(x1)

        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


def channel_shuffle(x, g):
    n, c, h, w = x.size()
    x = x.view(n, g, c // g, h, w).permute(
        0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, g=self.groups)


class SNet(nn.Module):
    cfg = {
        49: [24, 60, 120, 240, 512],
        146: [24, 132, 264, 528],
        535: [48, 248, 496, 992],
    }

    def __init__(self,  version=49, **kwargs):
        super(SNet, self).__init__()
        num_layers = [4, 8, 4]
        self.num_layers = num_layers
        channels = self.cfg[version]
        self.channels = channels

        self.conv1 = conv_bn(
            3, channels[0], kernel_size=3, stride=2,pad = 1
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )

        self.stage1 = self._make_layer(
            num_layers[0], channels[0], channels[1], **kwargs)
        self.stage2 = self._make_layer(
            num_layers[1], channels[1], channels[2], **kwargs)
        self.stage3 = self._make_layer(
            num_layers[2], channels[2], channels[3], **kwargs)
        if len(self.channels) == 5:
            self.conv5 = conv_bn(
                channels[3], channels[4], kernel_size=1, stride=1, pad=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(channels[-1], num_classes)

    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = [DownBlock(in_channels, out_channels, **kwargs)]
        for i in range(num_layers - 1):
            layers.append(BasicBlock(out_channels, **kwargs))
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        if len(self.channels) == 5:
            c5 = self.conv5(c5)

        Cglb_lat = self.avgpool(c5)
        # Cglb_lat = Cglb_lat.view(-1, self.channels[-1], 1, 1)

        # x = self.fc(x)

        return c4, c5, Cglb_lat