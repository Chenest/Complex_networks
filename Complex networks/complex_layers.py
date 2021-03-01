# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains several activation functions for complex values.
# --ComplexConvTranspose2d, ComplexConv2d, ComplexLinear

import math
import torch
import torch.nn as nn
from torch.nn import Parameter, init




class ComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()
        self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(kernel_size) != tuple:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if type(stride) != tuple:
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if type(padding) != tuple:
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if type(dilation) != tuple:
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        if type(output_padding) != tuple:
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding

        if type(groups) != tuple:
            self.groups = (groups, groups)
        else:
            self.groups = groups

        self.conv_tran_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, groups, False, dilation, padding_mode)
        self.conv_tran_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, groups, False, dilation, padding_mode)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels, 2))
            self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            fan_in_r, _ = init._calculate_fan_in_and_fan_out(self.conv_tran_r.weight)
            bound = 1 / math.sqrt(fan_in_r)
            init.uniform_(self.bias[:, 0], -bound, bound)
            fan_in_i, _ = init._calculate_fan_in_and_fan_out(self.conv_tran_i.weight)
            bound = 1 / math.sqrt(fan_in_i)
            init.uniform_(self.bias[:, 1], -bound, bound)

    def forward(self, input):
        assert (input.shape[-1] == 2)
        assert (len(input.shape) == 5)
        input = torch.stack([self.conv_tran_r(input[:, :, :, :, 0]) - self.conv_tran_i(input[:, :, :, :, 1]),
                            self.conv_tran_r(input[:, :, :, :, 1]) + self.conv_tran_i(input[:, :, :, :, 0])], dim=-1).transpose(1, -2)
        return (input + self.bias).transpose(1, -2)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(kernel_size) != tuple:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if type(stride) != tuple:
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if type(padding) != tuple:
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if type(dilation) != tuple:
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        if type(groups) != tuple:
            self.groups = (groups, groups)
        else:
            self.groups = groups
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels, 2))
            self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            fan_in_r, _ = init._calculate_fan_in_and_fan_out(self.conv_r.weight)
            bound = 1 / math.sqrt(fan_in_r)
            init.uniform_(self.bias[:, 0], -bound, bound)
            fan_in_i, _ = init._calculate_fan_in_and_fan_out(self.conv_i.weight)
            bound = 1 / math.sqrt(fan_in_i)
            init.uniform_(self.bias[:, 1], -bound, bound)

    def forward(self, input):
        assert (input.shape[-1] == 2)
        assert (len(input.shape) == 5)
        input = torch.stack([self.conv_r(input[:, :, :, :, 0]) - self.conv_i(input[:, :, :, :, 1]),
                            self.conv_r(input[:, :, :, :, 1]) + self.conv_i(input[:, :, :, :, 0])], dim=-1).transpose(1, -2)
        return (input + self.bias).transpose(1, -2)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 2))
            self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            fan_in_r, _ = init._calculate_fan_in_and_fan_out(self.fc_r.weight)
            bound = 1 / math.sqrt(fan_in_r)
            init.uniform_(self.bias[:, 0], -bound, bound)
            fan_in_i, _ = init._calculate_fan_in_and_fan_out(self.fc_i.weight)
            bound = 1 / math.sqrt(fan_in_i)
            init.uniform_(self.bias[:, 1], -bound, bound)

    def forward(self, input):
        # bs*in_features*2
        assert (input.shape[-1] == 2)
        assert (len(input.shape) == 3)
        input = torch.stack([self.fc_r(input[:, :, 0]) - self.fc_i(input[:, :, 1]),
                            self.fc_r(input[:, :, 1]) + self.fc_i(input[:, :, 0])], dim=-1)
        return input + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
