#!/usr/bin/env python
# encoding: utf-8

import scipy as spy
import networkx as nx
import numpy as np
from networkx.classes.graph import Graph

"""
稀疏矩阵转networkx图数据结构
"""
def mat_to_nxG(mat):
    g = nx.from_scipy_sparse_matrix(mat)
    return g
"""

networkx图数据结构转稀疏矩阵
"""
def nxG_to_mat(g):
    row=[]
    col=[]
    for edge in g.G.edges():
        row.append(int(edge[0]))
        col.append(int(edge[1]))
    value = np.ones(len(row))
    return spy.sparse.csc_matrix((value,(row,col)))

def nxG_or_lap(g):
    if isinstance(g, Graph):
        return True