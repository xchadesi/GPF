#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from NE import node2vec
from NE import line
from NE import tadw
from NE import gcn
from NE import lle
from NE import hope
from NE import lap
from NE import gf
#from NE import sdne
from NE import grarep


#学习节点向量
def learn_embeddings(g, args):
    #获取模型
    if args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    elif args.method == 'line':
        if args.label-file and not args.no_auto_save:
            model = line.LINE(g, epoch=args.epochs, rep_size=args.representation_size, order=args.order,
                              label_file=args.label_file, clf_ratio=args.clf_ratio)
        else:
            model = line.LINE(g, epoch=args.epochs,
                              repSize=args.representation_size, order=args.order)
    elif args.method == 'deepWalk':
        model = node2vec.Node2vec(graph=g, pathLength=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'tadw':
        # assert args.label_file != ''
        assert args.feature_file != ''
        g.read_node_label(args.label_file)
        g.read_node_features(args.feature_file)
        model = tadw.TADW(
            graph=g, dim=args.representation_size, lamb=args.lamb)
    elif args.method == 'gcn':
        assert args.label_file != ''
        assert args.feature_file != ''
        g.read_node_label(args.label_file)
        g.read_node_features(args.feature_file)
        model = gcn.GCN(graph=g, dropout=args.dropout,
                           weight_decay=args.weightDecay, hidden1=args.hidden,
                           epochs=args.epochs, clf_ratio=args.clf_ratio)
    elif args.method == 'grarep':
        model = GraRep(graph=g, Kstep=args.kstep, dim=args.representation_size)
    elif args.method == 'lle':
        model = lle.LLE(graph=g, d=args.representation_size)
    elif args.method == 'hope':
        model = hope.HOPE(graph=g, d=args.representation_size)
    elif args.method == 'sdne':
        encoder_layer_list = ast.literal_eval(args.encoder_list)
        model = sdne.SDNE(g, encoder_layer_list=encoderLayer_list,
                          alpha=args.alpha, beta=args.beta, nu1=args.nu1, nu2=args.nu2,
                          batch_size=args.bs, epoch=args.epochs, learningRate=args.lr)
    elif args.method == 'lap':
        model = lap.LaplacianEigenmaps(g, rep_size=args.representation_size)
    elif args.method == 'gf':
        model = gf.GraphFactorization(g, rep_size=args.representation_size,
                                      epoch=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay)
        
    #按节点编号排序
    def sortedDictValues(adict): 
        keys = adict.keys()
        keys = [int(k) for k in keys]
        keys.sort()
        #print keys
        return np.array([adict[str(key)] for key in keys])
    model.save_embeddings("embedding.txt")
    #输出节点向量
    return sortedDictValues(model.vectors)

#测试函数可用性
if __name__ == '__main__':
    import graph
    g = graph.Graph()
    g.read_edgelist(filename='data/karate.edgelist.txt', weighted=False, directed=False)
    args = ArgumentParser(noAutoSave='', method='node2vec') 
    print(learn_embeddings(args))
    
    
