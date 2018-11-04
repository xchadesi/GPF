#!/usr/bin/env python
# encoding: utf-8

from tqdm import tqdm
import scipy.sparse as ssp
import numpy as np
import networkx as nx
import s2vGraph
import random

####################################################
###  获取每个节点的邻节点:                   
###  1.邻节点个数固定:指定固定的h和max_nodes_per_hop 
###  2.邻节点个数不固定：也就是获取每个节点的所有h层级邻节点
###                   指定h即可
####################################################

"""
输入：
    ind:需要采样邻节点的节点编号
    A：整个图的稀疏邻接矩阵
    h:邻节点层级
    max_nodes_per_hop:每层指定的最大邻节点个数
    node_information:整个图的【节点信息】：融合了节点特征和Embdding,或单个
输出：
    g:用矩阵表示的所有邻节点构成的子图
    labels:
    features:每个邻节点的特征信息
"""
def helper_extraction(ind, A, h, max_nodes_per_hop=None, node_information=None):
    dist = 0
    
    nodes = set([ind])
    visited = set([ind])
    fringe = set([ind])
    for dist in range(1, h+1):
        #print(fringe)
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
    
    # move target nodes to top
    nodes.remove(ind)
    nodes = [ind] + list(nodes) 
    
    if max_nodes_per_hop is not None:
        if max_nodes_per_hop < len(nodes):
            nodes = np.random.choice(nodes, max_nodes_per_hop, replace=False)
            #fringe = random.sample(fringe, max_nodes_per_hop)
        if max_nodes_per_hop > len(nodes):
            nodes = np.random.choice(nodes, max_nodes_per_hop, replace=True)
    subgraph = A[nodes, :][:, nodes]
    
    # remove link between target nodes
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    # apply node-labeling
    labels = node_label(subgraph)
    print(labels)
    # get node features
    features = None
    nodes = np.array(nodes)
    
    if node_information is not None:
        features = node_information[nodes]
    
    #邻节点所构成的稀疏矩阵 
    g = subgraph
    
    return g, labels.tolist(), features

#############################################
###            抽取单个节点的封闭子图        ###
#############################################
"""
输入：
    A：整个图的稀疏邻接矩阵
    train：训练集节点的编号列表
    train_status：训练集节点的状态（也就是需要预测的状态）
    test：测试集节点的编号列表
    test_status：测试集节点的状态（也就是需要预测的状态）
    h:邻节点层级
    max_nodes_per_hop:每层指定的最大邻节点个数
    node_information:整个图的【节点信息】：融合了节点特征和Embdding,或单个
输出：
    train_graphs：s2vGraph构成的列表，每个s2vGraph是抽取的训练集节点的子图
    test_graphs：s2vGraph构成的列表，每个s2vGraph是抽取的测试集节点的子图
    max_n_label['value']：每个子图都会获取一个结构特征列表，所有列表中的最大值，利用该最大值构建one-hot向量
"""
def singleSubgraphs(A, train, train_status, test, test_status, h=1, max_nodes_per_hop=None, node_information=None):
    train_graphs = []
    test_graphs = []
    max_n_label = {'value': 0}
    for inn in train:
        g, n_labels, n_features = helper_extraction(int(inn), A, h, max_nodes_per_hop, node_information)
        max_n_label['value'] = max(max(n_labels), max_n_label['value'])
        train_graphs.append(s2vGraph.S2VGraph(g, train_status[inn], n_labels, n_features))
    for innt in test:
        g, n_labels, n_features = helper_extraction(int(innt), A, h, max_nodes_per_hop, node_information)
        max_n_label['value'] = max(max(n_labels), max_n_label['value'])
        test_graphs.append(s2vGraph.S2VGraph(g, test_status[innt], n_labels, n_features))
        
    return train_graphs, test_graphs ,max_n_label['value']


#############################################
###            抽取节点对的封闭子图          ###
#############################################
"""
输入：
    A：整个图的稀疏邻接矩阵
    train_pos：训练集节点对的编号列表，正样本(节点对存在链接)
    train_neg：训练集节点对的编号列表，负样本(节点对不存在链接)
    test_pos：测试集节点对的编号列表，正样本(节点对存在链接)
    test_neg：测试集节点对的编号列表，负样本(节点对不存在链接)
    h:邻节点层级
    max_nodes_per_hop:每层指定的最大邻节点个数
    node_information:整个图的【节点信息】：融合了节点特征和Embdding,或单个
输出：
    train_graphs：s2vGraph构成的列表，每个s2vGraph是抽取的训练集节点对的子图
    test_graphs：s2vGraph构成的列表，每个s2vGraph是抽取的测试集节点对的子图
    max_n_label['value']：
"""
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        pdb.set_trace()
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    #extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label):
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(s2vGraph.S2VGraph(g, g_label, n_labels, n_features))
        return g_list
    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    
    return train_graphs, test_graphs, max_n_label['value']

"""
输入：
    ind:需要采样邻节点的节点对编号列表
    A：整个图的稀疏邻接矩阵
    h:邻节点层级
    max_nodes_per_hop:每层指定的最大邻节点个数
    node_information:整个图的【节点信息】：融合了节点特征和Embdding,或单个
输出：
    g:用矩阵表示的所有邻节点构成的子图
    labels:
    features:每个邻节点的特征信息
"""
def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    #print A
    dist = 0
    
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    #nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        #nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    
    if max_nodes_per_hop is not None:
        if max_nodes_per_hop < len(nodes):
            nodes = np.random.choice(nodes, max_nodes_per_hop, replace=False)
            #fringe = random.sample(fringe, max_nodes_per_hop)
        if max_nodes_per_hop > len(nodes):
            nodes = np.random.choice(nodes, max_nodes_per_hop, replace=True)
    
    subgraph = A[nodes, :][:, nodes]
    # remove link between target nodes
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    
    g = subgraph
    
    return g, labels.tolist(), features


"""
对fringe中的所有节点从整个图的稀疏矩阵中查找1级邻节点
"""
def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
        #print res
    return res

"""
功能：获取子图中节点的结构化特征标签
输入：
   subgraph:矩阵子图
输出：
   labels：节点的结构化特征：相对于关注的节点，子图中其它节点的相对位置结构特征
"""
def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+range(2, K), :][:, [0]+range(2, K)]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    
    return labels

"""
两种评价链接预测的方法，用作预选参数h
"""
def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    return auc