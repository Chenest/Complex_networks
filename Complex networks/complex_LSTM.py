# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains LSTM model for complex values.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from complex_layers import ComplexLinear
from complex_activations import modSigmoid, modTanh
from complex_dropout import Cdropout1d
from complex_LayerNormalization import ComplexLayerNorm1d

class Complex_LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 bidirectional=0,
                 batch_first=True,
                 ln=False):
        super(Complex_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = bidirectional + 1
        self.batch_first = batch_first
        self.ln = ln
        layers = []
        for i in range(num_layers):
            for j in range(self.direction):
                layer = LSTMcell(input_size*self.direction, hidden_size, dropout=dropout, ln=ln)
                layers.append(layer)
            input_size = hidden_size
        self.layers = layers
        self.params = nn.ModuleList(layers)

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def layer_forward(self, l, xs, h, reverse=False):
        '''
        return:
            xs: (seq_len, batch, hidden)
            h: (1, batch, hidden)
        '''

        ys = []
        for i in range(xs.size(0)):
            if reverse:
                x = xs.narrow(0, (xs.size(0)-1)-i, 1)
            else:
                x = xs.narrow(0, i, 1)
            y, h = l(x, h)
            ys.append(y)
        y = torch.cat(ys, 0)
        return y, h

    def forward(self, x, hiddens=None):

        max_batch_size = x.size(0) if self.batch_first else x.size(1)
        if hiddens is None:
            hiddens = []
            for idx in range(self.num_layers * self.direction):
                hiddens.append((torch.zeros(1, max_batch_size, self.hidden_size, 2, dtype=x.dtype, device=x.device),
                                torch.zeros(1, max_batch_size, self.hidden_size, 2, dtype=x.dtype, device=x.device)))

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
                h = torch.cat((f_h[0], r_h[0]), 0)
                c = torch.cat((f_h[1], r_h[1]), 0)
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

class LSTMcell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch', ln=False, lnlearnable=True):
        super(LSTMcell, self).__init__()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        assert((dropout >= 0) & (dropout <= 1))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.ln = ln
        self.i2h = ComplexLinear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = ComplexLinear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        self.dropout_method = dropout_method
        self.sigmoid = modSigmoid()
        self.tanh1 = modTanh()
        self.tanh2 = modTanh()
        if dropout > 0:
            self.dropoutlayer = Cdropout1d(dropout)
        if ln:
            self.ln_h2h = ComplexLayerNorm1d(4 * hidden_size, learnable=lnlearnable)
            self.ln_i2h = ComplexLayerNorm1d(4 * hidden_size, learnable=lnlearnable)
            self.ln_cell = ComplexLayerNorm1d(hidden_size, learnable=lnlearnable)

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = Variable(torch.bernoulli(Tensor(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def mul(self, a, b):
        assert (a.shape == b.shape)
        size = a.size()
        a = a.reshape(-1, 2)
        b = b.reshape(-1, 2)
        output = torch.stack((torch.mul(a[:, 0], b[:, 0]) - torch.mul(a[:, 1], b[:, 1]),
                              torch.mul(a[:, 1], b[:, 0]) + torch.mul(a[:, 0], b[:, 1])), dim=-1)
        return output.reshape(size)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1, 2)
        c = c.view(c.size(1), -1, 2)
        x = x.view(x.size(1), -1, 2)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activations
        gates = self.sigmoid(preact[:, :3 * self.hidden_size, :])
        g_t = self.tanh1(preact[:, 3 * self.hidden_size:, :])
        i_t = gates[:, :self.hidden_size, :]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size, :]
        o_t = gates[:, -self.hidden_size:, :]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = self.dropoutlayer(g_t)

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(self.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)
        if self.ln:
            c_t = self.ln_cell(c_t)
        h_t = self.mul(o_t, self.tanh2(c_t))

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                h_t = self.dropoutlayer(h_t)
            if self.dropout_method == 'gal':
                    h_t.data.set_(self.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1, 2)
        c_t = c_t.view(1, c_t.size(0), -1, 2)
        return h_t, (h_t, c_t)

l1 = Complex_LSTM(32, 32, 3, 0.1, 1, True, True)
x = torch.rand(16, 8, 32, 2)
y, _ = l1(x)
print(y.shape)