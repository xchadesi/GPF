#!/usr/bin/env python
# encoding: utf-8

import argparse

"""
【设置运行参数】：
1、常规设置；2、读取数据设置；3、节点Eembdding设置；
4、采样设置；5、抽取子图设置；6、训练和测试参数设置；7、图网络模型设置；
"""

#############################################
###             主函数参数设置              ###
#############################################
parser = argparse.ArgumentParser(description='main() Parameters setting!')
#常规设置
parser.add_argument('--noCuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
#读取数据设置
parser.add_argument('--dataName', default='../data/karate.edgelist.txt',#'data/USAir.edgeList.txt', 
                    help='network path')
parser.add_argument('--labelName', default='../data/karate.label.txt',#'data/USAir.label.txt', 
                    help='node label file path')
parser.add_argument('--weighted', default=False, 
                    help='network weighted or not')
parser.add_argument('--directed', default=False, 
                    help='network directed or not')
#节点Eembdding设置
parser.add_argument('--embedMethod', default='node2vec', 
                    help='network embedding method')
#采样设置
parser.add_argument('--testRatio', type=float, default=0.3,
                    help='ratio of test links')
parser.add_argument('--maxTrainNum', type=int, default=None, 
                    help='set maximum number of train links (to fit into memory)')
#抽取子图设置
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--maxNodesPerHop', default=8, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')

#训练和测试参数设置
parser.add_argument('--num_epochs', default=50, 
                    help='num_epochs')
parser.add_argument('--learningRate', default=0.01, 
                    help='learningRate')

#############################################
###          Embedding模型参数设置          ###
#############################################
cmd_embed = argparse.ArgumentParser(description='Argparser for Embedding!')
cmd_embed.add_argument('--method', default='node2vec', choices=[
    'node2vec',
    'deepWalk',
    'line',
    'gcn',
    'grarep',
    'tadw',
    'lle',
    'hope',
    'lap',
    'gf',
    'sdne'
], help='The learning method')
cmd_embed.add_argument('--output',
                    help='Output representation file')
cmd_embed.add_argument('--number_walks', default=10, type=int,
                    help='Number of random walks to start at each node')
cmd_embed.add_argument('--directed', action='store_true',
                    help='Treat graph as directed.')
cmd_embed.add_argument('--walk_length', default=80, type=int,
                    help='Length of the random walk started at each node')
cmd_embed.add_argument('--workers', default=8, type=int,
                    help='Number of parallel processes.')
cmd_embed.add_argument('--representation_size', default=128, type=int,
                    help='Number of latent dimensions to learn for each node.')
cmd_embed.add_argument('--window_size', default=10, type=int,
                    help='Window size of skipgram model.')
cmd_embed.add_argument('--epochs', default=5, type=int,
                    help='The training epochs of LINE and GCN')
cmd_embed.add_argument('--p', default=1.0, type=float)
cmd_embed.add_argument('--q', default=1.0, type=float)
cmd_embed.add_argument('--label_file', default='',
                    help='The file of node label')
cmd_embed.add_argument('--feature_file', default='',
                    help='The file of node features')
cmd_embed.add_argument('--graph_format', default='adjlist', choices=['adjlist', 'edgelist'],
                    help='Input graph format')
cmd_embed.add_argument('--negative_ratio', default=5, type=int,
                    help='the negative ratio of LINE')
cmd_embed.add_argument('--weighted', action='store_true',
                    help='Treat graph as weighted')
cmd_embed.add_argument('--clf_ratio', default=0.5, type=float,
                    help='The ratio of training data in the classification')
cmd_embed.add_argument('--order', default=3, type=int,
                    help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
cmd_embed.add_argument('--no_auto_save', action='store_true',
                    help='no save the best embeddings when training LINE')
cmd_embed.add_argument('--dropout', default=0.5, type=float,
                    help='Dropout rate (1 - keep probability)')
cmd_embed.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight for L2 loss on embedding matrix')
cmd_embed.add_argument('--hidden', default=16, type=int,
                    help='Number of units in hidden layer 1')
cmd_embed.add_argument('--kstep', default=4, type=int,
                    help='Use k-step transition probability matrix')
cmd_embed.add_argument('--lamb', default=0.2, type=float,
                    help='lambda is a hyperparameter in TADW')
cmd_embed.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
cmd_embed.add_argument('--alpha', default=1e-6, type=float,
                    help='alhpa is a hyperparameter in SDNE')
cmd_embed.add_argument('--beta', default=5., type=float,
                    help='beta is a hyperparameter in SDNE')
cmd_embed.add_argument('--nu1', default=1e-5, type=float,
                    help='nu1 is a hyperparameter in SDNE')
cmd_embed.add_argument('--nu2', default=1e-4, type=float,
                    help='nu2 is a hyperparameter in SDNE')
cmd_embed.add_argument('--bs', default=200, type=int,
                    help='batch size of SDNE')
cmd_embed.add_argument('--encoder_list', default='[1000, 128]', type=str,
                    help='a list of numbers of the neuron at each encoder layer, the last number is the '
                         'dimension of the output node representation')

#############################################
###             图网络模型设置              ###
#############################################
cmd_opt = argparse.ArgumentParser(description='Argparser for graph_net')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='DGCNN', help='GCN/GAT/PSCN/DGCNN/mean_field/loopy_bp')
cmd_opt.add_argument('-batch_size', type=int, default=6, help='minibatch size')
cmd_opt.add_argument('-num_epochs', type=int, default=50, help='number of epochs')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-num_class', type=int, default=2, help='classes')
cmd_opt.add_argument('-dropout', type=float, default=0.2, help='whether add dropout after dense layer')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-attr_dim', default=0, help='attr_dim')
cmd_opt.add_argument('-latent_dim', type=str, default='128-32', help='dimension(s) of latent layers')
cmd_opt.add_argument('-out_dim', type=int, default=32, help='s2v output size')
cmd_opt.add_argument('-use_tag', type=bool, default=False, help='whether use structural tag')
#special for DGCNN 
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-hidden', type=int, default=128, help='dimension of regression')
#special for GAT
cmd_opt.add_argument('-attn_dropout', type=float, default=0.2, help='whether add dropout after dense layer')
cmd_opt.add_argument('-n_heads',  default=None, help='whether add dropout after dense layer')
#special for GCN 
cmd_opt.add_argument('-neighbor_size', type=int, default=8, help='whether add dropout after dense layer')
#special for PSCN 
cmd_opt.add_argument('-sequence_size', type=int, default=1, help='whether add dropout after dense layer')
