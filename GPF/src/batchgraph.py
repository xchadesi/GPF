#!/usr/bin/env python
# encoding: utf-8

import torch

"""
对抽取的batch_graph进行处理：多种特征的选择与融合
"""

#batch_graph:列表，放有多个s2vGraph实例
def PrepareFeatureLabel(batch_graph, cmd_args):
    
    labels = torch.LongTensor(len(batch_graph))
    n_nodes = 0

    if batch_graph[0].node_tags is not None and cmd_args.use_tag:
        node_tag_flag = True
        concat_tag = []
    else:
        node_tag_flag = False

    if batch_graph[0].node_features is not None:
        node_feat_flag = True
        concat_feat = []
    else:
        node_feat_flag = False

    #对batch_graph中所有子图的节点特征做融合
    for i in range(len(batch_graph)):
        labels[i] = batch_graph[i].label
        n_nodes += batch_graph[i].num_nodes
        if node_tag_flag == True:
            concat_tag += batch_graph[i].node_tags
        if node_feat_flag == True:
            tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
            concat_feat.append(tmp)

    if node_tag_flag == True:
        concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
        node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
        node_tag.scatter_(1, concat_tag, 1)

    if node_feat_flag == True:
        node_feat = torch.cat(concat_feat, 0)

    if node_feat_flag and node_tag_flag:
        # concatenate one-hot embedding of node tags (node labels) with continuous node features
        node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
    elif node_feat_flag == False and node_tag_flag == True:
        node_feat = node_tag
    elif node_feat_flag == True and node_tag_flag == False:
        pass
    else:
        node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

    if cmd_args.mode == 'gpu':
        node_feat = node_feat.cuda()
        labels = labels.cuda()

    return node_feat, labels
