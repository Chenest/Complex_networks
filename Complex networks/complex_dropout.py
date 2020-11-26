# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains several dropout functions for complex values.
# Cdropout1d, Cdropout2d

import torch
import torch.nn as nn

class Cdropout1d(nn.Module):
    def __init__(self, p, inplace=False):
        super(Cdropout1d, self).__init__()
        self.p = p
        assert 0 <= self.p <= 1
        self.inpalce = inplace

    def forward(self, input):
        # input:(batch_size, L, 2)
        if not self.training: return input
        device = 0 if input.get_device() else 1
        keep_prob = 1 - self.p
        if not keep_prob:
            if device:
                return torch.zeros_like(input).cuda()
            return torch.zeros_like(input)
        size = input.size()[:-1]
        mask = (torch.rand(size) < keep_prob).float().unsqueeze(-1).expand_as(input)
        if device:
            mask = mask.cuda()
        return mask * input / keep_prob

    def extra_repr(self):
        return 'p={}'.format(
            self.p
        )

class Cdropout2d(nn.Module):
    def __init__(self, p, inplace=False):
        super(Cdropout2d, self).__init__()
        self.p = p
        assert 0 <= self.p <= 1
        self.inpalce = inplace

    def forward(self, input):
        # input:(batch_size, L, 2)
        if not self.training: return input
        device = 1 if input.get_device() else 0
        keep_prob = 1 - self.p
        if not keep_prob:
            if device:
                return torch.zeros_like(input).cuda()
            return torch.zeros_like(input)
        size = input.size()[:-3]
        H = input.size()[-3]
        W = input.size()[-2]
        mask = (torch.rand(size) < keep_prob).float().unsqueeze(-1).repeat_interleave(H, dim=-1)\
            .unsqueeze(-1).repeat_interleave(W, dim=-1).unsqueeze(-1).expand_as(input)
        if device:
            mask = mask.cuda()
        return mask * input / keep_prob

    def extra_repr(self):
        return 'p={}'.format(
            self.p
        )

