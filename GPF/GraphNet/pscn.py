#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class BatchPSCN(nn.Module):
    def __init__(self, dropout, neighbor_size, sequence_size, n_units=[128, 32, 32, 1]):
        super(BatchPSCN, self).__init__()
        assert len(n_units) == 4
        self.dropout = dropout
        # input is of shape bs x num_feature x l where l = w*k
        # after conv1, shape=(bs x ? x w)
        # after conv2 shape=(bs x ? x w/2)
        self.conv1 = nn.Conv1d(in_channels=n_units[0],
                    out_channels=n_units[1], kernel_size=neighbor_size,
                    stride=neighbor_size)
        k = 1
        self.conv2 = nn.Conv1d(in_channels=n_units[1],
                    out_channels=n_units[2], kernel_size=k, stride=1)
        self.fc = nn.Linear(in_features=n_units[2] * (sequence_size - k + 1),
                    out_features=n_units[3])

    def forward(self, batch_graph, node_feat, edge_feat):
        num_nebhors = batch_graph[0].mat.shape[0]
        #获取每个子图中邻节点所构成的邻接矩阵：shape = num_nebhors * num_nebhors
        recep = np.zeros((len(batch_graph), num_nebhors, num_nebhors))
        i = 0 
        for batch in batch_graph:
            recep[i,:] = (batch.mat).todense()
            i += 1
        #转变成张量
        recep = torch.from_numpy(recep)
        recep = recep.sum(dim=1)
        #x: batch_size * nebhors * latent_dim
        x = node_feat.view(len(batch_graph), num_nebhors, node_feat.shape[1])
        
        bs, l = recep.size()[:2]
        
        #print(bs,l)
        
        n = x.size()[1] # x is of shape bs x n x num_feature
        offset = torch.ger(torch.arange(0, bs).long(), torch.ones(l).long() * n)
        offset = Variable(offset, requires_grad=False)
        
        #print(recep.shape)
        #print(offset.shape)
        
        recep = (recep.long() + offset).view(-1)
        x = x.view(bs * n, -1)
        x = x.index_select(dim=0, index=recep)
        x = x.view(bs, l, -1) # x is of shape bs x l x num_feature, l=w*k
        x = x.transpose(1, 2) # x is of shape bs x num_feature x l, l=w*k
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.view(bs, -1)
        x = self.fc(x)
        #后面接其它网络层        
        return x
        #后面不接其它层，直接获取分类结果
        #return F.log_softmax(x, dim=-1)
