#!/usr/bin/env python
# encoding: utf-8

import sys
#指定路径
sys.path.append("../") 

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from GraphNet.DGCNN import DGCNN
from GraphNet.mlp_dropout import MLPClassifier
from GraphNet.gcn import BatchGCN
from GraphNet.gat import BatchGAT
from GraphNet.pscn import BatchPSCN
from batchgraph import PrepareFeatureLabel

class Classifier(nn.Module):
    def __init__(self, cmd_args):
        super(Classifier, self).__init__()
        
        self.cmd_args = cmd_args
        if not self.cmd_args.use_tag:
            cmd_args.feat_dim = 0
        
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        elif cmd_args.gm == 'GCN':
            model = BatchGCN
        elif cmd_args.gm == 'GAT':
            model = BatchGAT
        elif cmd_args.gm == 'PSCN':
            model = BatchPSCN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                            num_edge_feats=0,
                            k=cmd_args.sortpooling_k)
        elif cmd_args.gm == 'GCN':
            self.s2v = model(dropout=cmd_args.dropout,
                             latent_dim= [cmd_args.feat_dim+k for k in cmd_args.latent_dim])
        elif cmd_args.gm == 'GAT':
            self.s2v = model(dropout=cmd_args.dropout,
                             n_units=[cmd_args.feat_dim+k for k in cmd_args.latent_dim],
                             #n_heads=cmd_args.n_heads,
                             attn_dropout=cmd_args.attn_dropout)
        elif cmd_args.gm == 'PSCN':
            self.s2v = model(dropout=cmd_args.dropout,
                             n_units=[cmd_args.feat_dim+k for k in cmd_args.latent_dim],
                             neighbor_size=cmd_args.neighbor_size, 
                             sequence_size=cmd_args.sequence_size)    
        else:
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim,
                            num_edge_feats=0,
                            max_lv=cmd_args.max_lv)
        
        if cmd_args.gm == 'DGCNN':
            out_dim = cmd_args.out_dim
        else:
            out_dim = cmd_args.out_dim + cmd_args.feat_dim
        
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=False)
    


    def forward(self, batch_graph):
        node_feat, labels = PrepareFeatureLabel(batch_graph, self.cmd_args)
        embed = self.s2v(batch_graph, node_feat, None)
        
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        node_feat, labels = PrepareFeatureLabel(batch_graph, self.cmd_args)
        embed = self.s2v(batch_graph, node_feat, None)
        
        return embed, labels
        

def loop_dataset(g_list, classifier, sample_idxes, bsize, optimizer=None):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []
    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        #batch_graph是放有很多s2vGraph的列表
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = classifier(batch_graph)
        all_scores.append(logits[:, 1].detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))
    
    return avg_loss