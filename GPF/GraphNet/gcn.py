#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from GraphNet.gcn_layers import BatchGraphConvolution
import numpy as np


class BatchGCN(nn.Module):
    def __init__(self, dropout, latent_dim=[128, 32, 32, 1]):
        super(BatchGCN, self).__init__()
        self.num_layer = len(latent_dim) - 1
        self.dropout = dropout
        self.latent_dim = latent_dim

        self.layer_stack = nn.ModuleList()

        for i in range(self.num_layer):
            self.layer_stack.append(
                    BatchGraphConvolution(latent_dim[i], latent_dim[i + 1])
                    )

    def forward(self, batch_graph, node_feat, edge_feat):
        num_nebhors = batch_graph[0].mat.shape[0]
        #获取每个子图中邻节点所构成的邻接矩阵：shape = num_nebhors * num_nebhors
        lap = np.zeros((len(batch_graph), num_nebhors, num_nebhors))
        i = 0 
        for batch in batch_graph:
            lap[i,:] = (batch.mat).todense()
            i += 1
        #转变成张量
        lap = torch.from_numpy(lap)
        
        #x: batch_size * nebhors * latent_dim
        x = node_feat.view(len(batch_graph), num_nebhors, node_feat.shape[1])
        
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)
            if i + 1 < self.num_layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        
        #每个子图的所有邻节点进行求和聚合
        x = x.sum(dim=1)
        
        #后面接其它网络层        
        return x
        #后面不接其它层，直接获取分类结果
        #return F.log_softmax(x, dim=-1)
