#!/usr/bin/env python
# encoding: utf-8

import scipy.sparse as ssp
import random
import math
import numpy as np

"""
获取单个节点的正负样本：对节点进行采样
"""
def sample_single(g, test_ratio=0.1, max_train_num=None):
    nodes = g.G.nodes()
    
    num = len(nodes)
    split = int(math.ceil(num * (1 - test_ratio)))
    train = nodes[:split]
    test = nodes[split:]
    if max_train_num is not None:
        perm = np.random.permutation(len(train))[:max_train_num]
        train = train_pos[perm]
    #获取节点状态标签    
    train_status = {}
    test_status = {}
    for nd in train:
        train_status[nd] = g.G.node[nd]['status']
    for nd in test:
        test_status[nd] = g.G.node[nd]['status']   
    return train, train_status, test, test_status

"""
获取节点链接对的正负样本：正（有链接）、负（没有链接）
"""
def sample_pairs(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    
    return train_pos, train_neg, test_pos, test_neg