# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains several activation functions for complex values.
# --modReLU, CReLU, zReLU, mod_LeakyReLU, CLeakyReLU
import torch
import torch.nn as nn


class ComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input):
        assert (input.shape[-1] == 2)
        assert (len(input.shape) == 5)
        return torch.stack([self.fc_r(input[:, :, :, :, 0]) - self.fc_i(input[:, :, :, :, 1]),
                            self.fc_r(input[:, :, :, :, 1]) + self.fc_i(input[:, :, :, :, 0])], dim=-1)


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        assert (input.shape[-1] == 2)
        assert (len(input.shape) == 5)
        return torch.stack([self.fc_r(input[:, :, :, :, 0]) - self.fc_i(input[:, :, :, :, 1]),
                            self.fc_r(input[:, :, :, :, 1]) + self.fc_i(input[:, :, :, :, 0])], dim=-1)


class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        assert (input.shape[-1] == 2)
        assert (len(input.shape) == 3)
        return torch.stack([self.fc_r(input[:, :, 0]) - self.fc_i(input[:, :, 1]),
                            self.fc_r(input[:, :, 1]) + self.fc_i(input[:, :, 0])], dim=-1)
