# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains several activation functions for complex values.
# --modReLU, CReLU, zReLU, modLeakyReLU, CLeakyReLU, modSigmoid, CSigmoid
# --modTanh, CTanh

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable

class modReLU(nn.Module):
    def __init__(self, learnable=True):
        super(modReLU, self).__init__()
        self.b = Tensor(1).fill_(1)
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.b = W(self.b)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.b)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input):
        # input:(bs,~,2)
        # output:(bs,~,2)
        size = input.size()
        x = input.reshape(-1, 2)
        absz = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        output = torch.relu(1 + self.b.expand_as(absz) / absz) * x
        return output.reshape(size)

class modLeakyReLU(nn.Module):
    def __init__(self, scope=0.1, learnable=True):
        super(modLeakyReLU, self).__init__()
        self.b = Tensor(1).fill_(1)
        self.scope = scope
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.b = W(self.b)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.b)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input):
        # input:(bs,~,2)
        # output:(bs,~,2)
        size = input.size()
        x = input.reshape(-1, 2)
        absz = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        output = torch.leaky_relu(1 + self.b.expand_as(absz) / absz, self.scope).reshape(-1, 1) * x
        return output.reshape(size)

class modSigmoid(nn.Module):
    def __init__(self, learnable=True):
        super(modSigmoid, self).__init__()
        self.b = Tensor(1).fill_(1)
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.b = W(self.b)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.b)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input):
        # input:(bs,~,2)
        # output:(bs,~,2)
        size = input.size()
        x = input.reshape(-1, 2)
        absz = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        output = torch.sigmoid(1 + self.b.expand_as(absz) / absz).reshape(-1, 1) * x
        return output.reshape(size)

class modTanh(nn.Module):
    def __init__(self, learnable=True):
        super(modTanh, self).__init__()
        self.b = Tensor(1).fill_(1)
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.b = W(self.b)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.b)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input):
        # input:(bs,~,2)
        # output:(bs,~,2)
        size = input.size()
        x = input.reshape(-1, 2)
        absz = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        output = torch.tanh(1 + self.b.expand_as(absz) / absz).reshape(-1, 1) * x
        return output.reshape(size)

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, input):
        return F.relu(input)

class CLeakyReLU(nn.Module):
    def __init__(self, scope=0.1):
        super(CLeakyReLU, self).__init__()
        self.scope = scope
    def forward(self, input):
        return F.leaky_relu(input, self.scope)

class CSigmoid(nn.Module):
    def __init__(self):
        super(CSigmoid, self).__init__()

    def forward(self, input):
        return F.sigmoid(input)

class CTanh(nn.Module):
    def __init__(self):
        super(CTanh, self).__init__()

    def forward(self, input):
        return F.tanh(input)

class zReLU(nn.Module):
    def __int__(self):
        super(zReLU, self).__init__()

    def forward(self, input):
        size = input.size()
        x = input.reshape(-1, 2)
        output = x.where((x[:, 0] < 0)|(x[:, 1] > 0), torch.zeros_like(x))
        return output.reshape(size)
