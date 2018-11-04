#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class BatchGraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.weight)

    def forward(self, x, lap):
        #bacth_size * in_dim * out_dim
        expand_weight = self.weight.expand(x.shape[0], -1, -1) 
        
        support = torch.bmm(x, expand_weight)
        
        output = torch.bmm(lap, support.double())
        
        if self.bias is not None:
            return (output + self.bias.double()).float()
        else:
            return output.float()
