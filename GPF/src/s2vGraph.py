#!/usr/bin/env python
# encoding: utf-8

import networkx as nx
import numpy as np
import until

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label:预测状态
            node_tags: a list of integer node tags：结构化特征标签
            node_features: a numpy array of continuous node features
        '''
        if until.nxG_or_lap(g):
            self.g = g
        else:
            self.g = until.mat_to_nxG(g)
        self.mat = g
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(nx.degree(self.g)).values())

        if len(self.g.edges()) != 0:
            x, y = zip(*self.g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])