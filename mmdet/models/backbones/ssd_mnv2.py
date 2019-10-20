import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from mmdet.models.backbones.mobilenetv2 import MobileNetV2, InvertedResidual
# from mobilenet_v2 import MobileNetV2, InvertedResidual
from mmcv.cnn import constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES

@BACKBONES.register_module
class SSDMNV2(nn.Module):

    def __init__(self, input_size, width_mult=1.0, out_feature_indices=(14,)):
        super(SSDMNV2, self).__init__()
        self.input_size = input_size
        self.out_feature_indices = out_feature_indices
        self.features = MobileNetV2(width_mult=width_mult).features
        # print(self.base_net)

        self.extra = ModuleList([
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 128, stride=2, expand_ratio=0.25)
        ])

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)


    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            if i in self.out_feature_indices:
                sub = getattr(self.features[i], "conv")
                for layer in sub[:3]:
                    x = layer(x)
                # add expand layers
                outs.append(x)
                for layer in sub[3:]:
                    x = layer(x)
            else:
                x = layer(x)
        # add last layer
        outs.append(x)
        for i, layer in enumerate(self.extra):
            x = layer(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

'''
if __name__ == '__main__':
    import numpy as np
    net = SSDMV2(320)
    image = np.zeros((2,3,320,320), dtype=np.float32)
    out = net.forward(torch.from_numpy(image))
    for i in range(len(out)):
        print(out[i].size())
'''