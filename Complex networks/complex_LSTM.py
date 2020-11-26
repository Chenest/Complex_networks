# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains LSTM model for complex values.

import math
import torch
import torch.nn as nn
from complex_layers import ComplexLinear
from complex_activations import modSigmoid, modTanh
from complex_dropout import Cdropout1d
from complex_operation import mul


class _ComplexLSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(_ComplexLSTMcell, self).__init__()
        if (dropout < 0)|(dropout > 1):
            raise ValueError('dropout must be in [0, 1]')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = None
        self.dropout = None
        self.i2h = ComplexLinear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = ComplexLinear(hidden_size, 4 * hidden_size, bias=bias)
        if dropout != 0:
            self.dropout = Cdropout1d(dropout)
        self.sigmoid = modSigmoid()
        self.tanh1 = modTanh()
        self.tanh2 = modTanh()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.i2h.parameters():
            w.data.uniform_(-std, std)
        for w in self.h2h.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout != None
        h, c = hidden
        if (len(x.shape) != 3)|(len(h.shape) != 3)|(len(c.shape) != 3)\
                    |(x.shape[-1] != 2)|(h.shape[-1] != 2)|(c.shape[-1] != 2):
            raise ValueError('Input tensor must be size of [batch_size, input_size, 2]')
        h = h.view(h.size(0), -1, 2)
        c = c.view(c.size(0), -1, 2)
        x = x.view(x.size(0), -1, 2)

        i2h = self.i2h(x)
        h2h = self.h2h(h)
        preact = i2h + h2h

        gates = self.sigmoid(preact[:, :3 * self.hidden_size, :])
        g_t = self.tanh1(preact[:, 3 * self.hidden_size:, :])
        i_t = gates[:, :self.hidden_size, :]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size, :]
        o_t = gates[:, -self.hidden_size:, :]
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        h_t = mul(o_t, self.tanh2(c_t))

        if do_dropout:
            h_t = self.dropout(h_t)

        h_t = h_t.view(h_t.size(0), -1, 2)
        c_t = c_t.view(c_t.size(0), -1, 2)
        return h_t, c_t


class ComplexLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 bidirectional=False,
                 batch_first=True,
                 bias=True):
        super(ComplexLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = bidirectional + 1
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.dropout = dropout
        layers = []
        for i in range(num_layers):
            for j in range(self.direction):
                layers.append(_ComplexLSTMcell(input_size * self.direction, hidden_size, bias=bias, dropout=dropout))
            input_size = hidden_size
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def layer_forward(self, l, xs, h, reverse=False):
        h, c = h
        ys = []
        for i in range(xs.size(0)):
            if reverse:
                x = xs[xs.size(0)-1]
            else:
                x = xs[i]
            h, c = l(x, (h, c))
            ys.append(h)
        y = torch.stack(ys, 0)
        return y, (h, c)


    def forward(self, x, hiddens=None):

        max_batch_size = x.size(0) if self.batch_first else x.size(1)
        if hiddens is None:
            hiddens = []
            for idx in range(self.num_layers * self.direction):
                hiddens.append((torch.zeros(max_batch_size, self.hidden_size, 2, dtype=x.dtype, device=x.device),
                                torch.zeros(max_batch_size, self.hidden_size, 2, dtype=x.dtype, device=x.device)))

        if self.batch_first:
            x = x.permute(1, 0, 2, 3).contiguous()

        if self.direction > 1:
            x = torch.cat((x, x), 2)
        if type(hiddens) != list:
            # when the hidden feed is (direction * num_layer, batch, hidden)
            tmp = []
            for idx in range(hiddens[0].size(0)):
                tmp.append((hiddens[0].narrow(0, idx, 1),
                           (hiddens[1].narrow(0, idx, 1))))
            hiddens = tmp

        new_hs = []
        new_cs = []
        for l_idx in range(0, len(self.layers), self.direction):
            l, h = self.layers[l_idx], hiddens[l_idx]
            f_x, f_h = self.layer_forward(l, x, h)
            if self.direction > 1:
                l, h = self.layers[l_idx+1], hiddens[l_idx+1]
                r_x, r_h = self.layer_forward(l, x, h, reverse=True)

                x = torch.cat((f_x, r_x), 2)
                h = torch.stack((f_h[0], r_h[0]), 0)
                c = torch.stack((f_h[1], r_h[1]), 0)
            else:
                x = f_x
                h, c = f_h
            new_hs.append(h)
            new_cs.append(c)

        h = torch.cat(new_hs, 0)
        c = torch.cat(new_cs, 0)
        if self.batch_first:
            x = x.permute(1, 0, 2, 3)
        return x, (h, c)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
