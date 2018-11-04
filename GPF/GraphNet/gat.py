#!/usr/bin/env python
# encoding: utf-8


from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphNet.gat_layers import BatchMultiHeadGraphAttention
import numpy as np


class BatchGAT(nn.Module):
    def __init__(self, dropout, n_units=[128, 32, 32, 1], n_heads=[8, 1],
            attn_dropout=0.0):
        
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        
        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            
            self.layer_stack.append(
                    
                    BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in,
                        f_out=n_units[i + 1], attn_dropout=attn_dropout)
                    )

    def forward(self, batch_graph, node_feat, edge_feat):
        num_nebhors = batch_graph[0].mat.shape[0]
        #获取每个子图中邻节点所构成的邻接矩阵：shape = num_nebhors * num_nebhors
        adj = np.zeros((len(batch_graph), num_nebhors, num_nebhors))
        i = 0 
        for batch in batch_graph:
            adj[i,:] = (batch.mat).todense()
            i += 1
        #转变成张量
        adj = torch.from_numpy(adj)
        
        #x: batch_size * nebhors * latent_dim
        x = node_feat.view(len(batch_graph), num_nebhors, node_feat.shape[1])
        
        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj) # bs x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
                
        x = x.sum(dim=1)
        #后面接其它网络层        
        return x
        #后面不接其它层，直接获取分类结果
        #return F.log_softmax(x, dim=-1)
