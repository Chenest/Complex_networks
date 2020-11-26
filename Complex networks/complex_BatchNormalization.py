# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains BatchNormalization for complex values.

import torch
import torch.nn as nn
from torch.nn import Parameter


class NaiveComplexBatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.bn_r = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        assert (len(input.shape) == 3)
        assert (input.shape[-1] == 2)
        re, im = self.bn_r(input[:, :, 0]).unsqueeze(-1), self.bn_i(input[:, :, 1]).unsqueeze(-1)
        output = torch.cat([re, im], dim=-1)
        return output

    def extra_repr(self):
        return '{}, eps={}, momentum={}, affine={}, track_running_stats={}'.format(
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )


class NaiveComplexBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.bn_r = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        assert (len(input.shape) == 5)
        assert (input.shape[-1] == 2)
        re, im = self.bn_r(input[:, :, :, :, 0]).unsqueeze(-1), self.bn_i(input[:, :, :, :, 1]).unsqueeze(-1)
        output = torch.cat([re, im], dim=-1)
        return output

    def extra_repr(self):
        return '{}, eps={}, momentum={}, affine={}, track_running_stats={}'.format(
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )


class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)

    def extra_repr(self):
        return '{}, eps={}, momentum={}, affine={}, track_running_stats={}'.format(
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        assert (input.shape[-1] == 2)
        assert (len(input.shape) == 5)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean = input.mean([0, 2, 3])


            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            input = input - mean[None, :, None, None, :]

            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1. / n * input[:, :, :, :, 0].pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input[:, :, :, :, 1].pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input[:, :, :, :, 0].mul(input[:, :, :, :, 1])).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

            input = input - mean[None, :, None, None, :]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None, None] * input[:, :, :, :, 0] + Rri[None, :, None, None] * input[:, :, :, :, 1], \
                           Rii[None, :, None, None] * input[:, :, :, :, 1] + Rri[None, :, None, None] * input[:, :, :, :, 0]

        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 0, None, None], \
                               self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 1, None, None]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return torch.stack([input_r, input_i], dim=-1)


class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input):
        assert (len(input.shape) == 3)
        assert (input.shape[-1] == 2)
        # self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean = input.mean(dim=0)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            # zero mean values
            input = input - mean


            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input[:, :, 0].var(dim=0, unbiased=False) + self.eps
            Cii = input[:, :, 1].var(dim=0, unbiased=False) + self.eps
            Cri = (input[:, :, 0].mul(input[:, :, 1])).mean(dim=0)

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]
            # zero mean values
            input = input - mean

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :] * input[:, :, 0] + Rri[None, :] * input[:, :, 1], \
                           Rii[None, :] * input[:, :, 1] + Rri[None, :] * input[:, :, 0]

        if self.affine:
            input_r, input_i = self.weight[None, :, 0] * input_r + self.weight[None, :, 2] * input_i + \
                               self.bias[None, :, 0], \
                               self.weight[None, :, 2] * input_r + self.weight[None, :, 1] * input_i + \
                               self.bias[None, :, 1]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return torch.stack([input_r, input_i], dim=-1)

