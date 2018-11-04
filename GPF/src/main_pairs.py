#!/usr/bin/env python
# encoding: utf-8

import graph
import embeddings
import sample
import classifier
from classifier import loop_dataset
import subgraphs
import argparse
import torch
import torch.optim as optim
import networkx as nx
import until
import random
from tqdm import tqdm

#导入配置参数
from Parameters import parser, cmd_embed, cmd_opt

#参数转换
args = parser.parse_args()
args.cuda = not args.noCuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.maxNodesPerHop is not None:
    args.maxNodesPerHop = int(args.maxNodesPerHop)

#读取数据
g = graph.Graph()
g.read_edgelist(filename=args.dataName, weighted=args.weighted, directed=args.directed)

#获取全图节点的Embedding
embed_args = cmd_embed.parse_args() 
embeddings = embeddings.learn_embeddings(g, embed_args)
node_information = embeddings

#正负节点对采样
net = until.nxG_to_mat(g)
train_pos, train_neg, test_pos, test_neg = sample.sample_pairs(net, args.testRatio, max_train_num=args.maxTrainNum)

#抽取节点对的封闭子图
train_graphs, test_graphs, max_n_label = subgraphs.links2subgraphs(net, train_pos, train_neg, test_pos, test_neg, args.hop, args.maxNodesPerHop, node_information)
print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

#加载网络模型，并在classifier中配置相关参数
cmd_args = cmd_opt.parse_args()
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = node_information.shape[1]
cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]
model = classifier.Classifier(cmd_args)
optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

#训练和测试
train_idxes = list(range(len(train_graphs)))
best_loss = None
for epoch in range(args.num_epochs):
    random.shuffle(train_idxes)
    
    model.train()
    avg_loss = loop_dataset(train_graphs, model, train_idxes, cmd_args.batch_size, optimizer=optimizer)
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

    model.eval()
    test_loss = loop_dataset(test_graphs, model, list(range(len(test_graphs))), cmd_args.batch_size)
    print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

