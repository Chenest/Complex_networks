# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chenest
"""
# this profile contains several operation functions for complex values.
# -- +, -, *, /, inv(2*2), for one value or matrix

import numpy as np
import torch
import torch.nn
from matplotlib import pyplot as plt

def add(a, b):
    # add
    assert(a.shape[-1] == 2)
    assert(b.shape[-1] == 2)
    return a + b

def sub(a, b):
    # sub
    assert(a.shape[-1] == 2)
    assert(b.shape[-1] == 2)
    return a - b

def conjugate(a):
    # conjugate
    assert(a.shape[-1] == 2)
    a = a.unsqueeze(0).transpose(0, -1).squeeze(-1)
    c = torch.stack((a[0],
                     -a[1]), dim=-1)
    return c

def tranpose(a):
    # tranposation
    assert(a.shape[-1] == 2)
    assert(len(a.shape) > 2)
    return a.transpose(-2, -3)

def con_tran(a):
    # conjugate and tranposation
    assert(a.shape[-1] == 2)
    return tranpose(conjugate(a))

def mul(a, b):
    # dot product
    assert(a.shape[-1] == 2)
    assert(b.shape[-1] == 2)
    a = a.unsqueeze(0).transpose(0, -1).squeeze(-1)
    b = b.unsqueeze(0).transpose(0, -1).squeeze(-1)
    c = torch.stack((torch.mul(a[0], b[0]) - torch.mul(a[1], b[1]),
                   torch.mul(a[0], b[1]) + torch.mul(a[1], b[0])), dim=-1)
    return c

def Bmul(a, b):
    # dot product for batch
    assert(a.shape[-1] == 2)
    assert(b.shape[-1] == 2)
    assert(a.shape[0] == b.shape[0])
    a = a.unsqueeze(-2).transpose(0, -2).squeeze(0)
    b = b.unsqueeze(-2).transpose(0, -2).squeeze(0)
    return mul(a, b).unsqueeze(0).transpose(0, -2).squeeze(-2)

def div(a, b):
    # divide:a/b
    assert(a.shape[-1] == 2)
    assert(a.shape[-1] == 2)
    x = mul(b, conjugate(b))
    x = x.unsqueeze(0).transpose(0, -1).squeeze(-1)
    x[1] = x[0]
    x = x.unsqueeze(-1).transpose(0, -1).squeeze(0)
    y = mul(a, conjugate(b))
    return y / x

def Bdiv(a, b):
    # divide for batch
    assert(a.shape[-1] == 2)
    assert(b.shape[-1] == 2)
    assert(a.shape[0] == b.shape[0])
    a = a.unsqueeze(-2).transpose(0, -2).squeeze(0)
    b = b.unsqueeze(-2).transpose(0, -2).squeeze(0)
    return div(a, b).unsqueeze(0).transpose(0, -2).squeeze(-2)

def matmul(a, b):
    # cross product
    assert(a.shape[-1] == 2)
    assert(b.shape[-1] == 2)
    assert(a.shape[-2] == b.shape[-3])
    a = a.unsqueeze(0).transpose(0, -1).squeeze(-1)
    b = b.unsqueeze(0).transpose(0, -1).squeeze(-1)
    re = torch.matmul(a[0], b[0]) - torch.matmul(a[1], b[1])
    im = torch.matmul(a[1], b[0]) + torch.matmul(a[0], b[1])
    return torch.stack((re, im), dim=-1)

def det(a):
    # 2*2 determinant
    assert(a.shape[-1] == 2)
    assert(a.shape[-2] == 2)
    assert(a.shape[-3] == 2)
    a = a.unsqueeze(0).transpose(0, -3).squeeze(-3)
    a = a.unsqueeze(1).transpose(1, -2).squeeze(-2)
    out = mul(a[0, 0], a[1, 1]) - mul(a[0, 1], a[1, 0])
    return out

def inv(a):
    # n*n inverse
    # the output don't divide by det as the error will be expanded if divided.
    assert(a.shape[-1] == 2)
    assert(a.shape[-2] == a.shape[-3])
    assert(len(a.shape) < 5)
    N = a.shape[-2]
    a = a.unsqueeze(0).transpose(0, -1).squeeze(-1)
    re = a[0]
    im = a[1]
    x1 = torch.cat((re, -im), dim=-1)
    x2 = torch.cat((im, re), dim=-1)
    x = torch.cat((x1, x2), dim=-2)
    x = torch.inverse(x)
    x = x.unsqueeze(0).transpose(0, -1).squeeze(-1)
    x = x.unsqueeze(0).transpose(0, -1).squeeze(-1)
    re = x[0:N, 0:N]
    im = x[N:(2*N), 0:N]
    out = torch.stack((re, im), dim=-1)
    return out.unsqueeze(0).transpose(0, -2).squeeze(-2)


