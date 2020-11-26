# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains LayerNormalization for complex values.

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable

class _ComplexLayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True, learnable=True):
        super(_ComplexLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if learnable:
            W = Parameter
        else:
            W = Variable
        if self.affine:
            self.weight = W(torch.Tensor(num_features, 3))
            self.bias = W(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)

    def extra_repr(self):
        return '{}, eps={}, affine={}, learnable={}'.format(
            self.num_features, self.eps, self.affine, self.learnable
        )


class ComplexLayerNorm1d(_ComplexLayerNorm):

    def forward(self, input):
        assert (len(input.shape) == 3)
        assert (input.shape[-1] == 2)
        # self._check_input_dim(input)

        # calculate mean of real and imaginary part
        mean = input.mean(dim=1)

        # zero mean values
        input = input - mean[:, None, :]

        # Elements of the covariance matrix (biased for train)
        n = input.numel() / input.size(0)
        Crr = input[:, :, 0].var(dim=1, unbiased=False) + self.eps
        Cii = input[:, :, 1].var(dim=1, unbiased=False) + self.eps
        Cri = (input[:, :, 0].mul(input[:, :, 1])).mean(dim=1)

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[:, None] * input[:, :, 0] + Rri[:, None] * input[:, :, 1], \
                           Rii[:, None] * input[:, :, 1] + Rri[:, None] * input[:, :, 0]

        if self.affine:
            input_r, input_i = self.weight[None, :, 0] * input_r + self.weight[None, :, 2] * input_i + \
                               self.bias[None, :, 0], \
                               self.weight[None, :, 2] * input_r + self.weight[None, :, 1] * input_i + \
                               self.bias[None, :, 1]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return torch.stack([input_r, input_i], dim=-1)