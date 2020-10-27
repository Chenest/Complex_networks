# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains several pooling functions for complex values.
# Cmax_pool2d, Cavg_pool2d

import torch
import torch.nn as nn
import torch.nn.functional as F

class Cmax_pool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
        super(Cmax_pool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input):
        # input:(batch_size, L, H, 2)
        output1 = F.max_pool2d(input[:, :, :, 0], self.kernel_size, self.stride, self.padding,
                               self.dilation, self.ceil_mode, self.return_indices).unsqueeze(-1)
        output2 = F.max_pool2d(input[:, :, :, 1], self.kernel_size, self.stride, self.padding,
                               self.dilation, self.ceil_mode, self.return_indices).unsqueeze(-1)
        return torch.cat([output1, output2], dim=-1)

class Cavg_pool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False, count_include_pad=False, divisor_override=None):
        super(Cavg_pool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input):
        # input:(batch_size, L, H, 2)
        output1 = F.avg_pool2d(input[:, :, :, 0], self.kernel_size, self.stride, self.padding,
                               self.dilation, self.ceil_mode, self.count_include_pad, self.divisor_override).unsqueeze(-1)
        output2 = F.avg_pool2d(input[:, :, :, 1], self.kernel_size, self.stride, self.padding,
                               self.dilation, self.ceil_mode, self.count_include_pad, self.divisor_override).unsqueeze(-1)
        return torch.cat([output1, output2], dim=-1)

