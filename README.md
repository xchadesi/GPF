# GPF
## 一、GPF（Graph Processing Flow）：利用图神经网络处理问题的一般化流程

1、图节点预表示：利用NE框架，直接获得全图每个节点的Embedding;<br>
2、正负样本采样：（1）单节点样本；（2）节点对样本；<br>
3、抽取封闭子图：可做类化处理，建立一种通用图数据结构;<br>
4、子图特征融合：预表示、节点特征、全局特征、边特征;<br>
5、网络配置：可以是图输入、图输出的网络；也可以是图输入，分类/聚类结果输出的网络;<br>
6、训练和测试;<br>

![https://github.com/xchadesi/GPF/blob/master/gpf.JPG](https://github.com/xchadesi/GPF/blob/master/gpf.JPG)

## 二、主要文件：
1、graph.py：读入图数据;<br>
2、embeddings.py：预表示学习;<br>
3、sample.py：采样;<br>
4、subgraphs.py/s2vGraph.py：抽取子图;<br>
5、batchgraph.py：子图特征融合;<br>
6、classifier.py：网络配置;<br>
7、parameters.py/until.py:参数配置/帮助文件;<br>

## 三、使用
1、在parameters.py中配置相关参数（可默认）；<br>
2、在example/文件夹中运行相应的案例文件--包括链接预测、节点状态预测；<br>

以链接预测为例：<br>

### 1、导入配置参数
```from parameters import parser, cmd_embed, cmd_opt```

### 2、参数转换
```
args = parser.parse_args()
args.cuda = not args.noCuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.maxNodesPerHop is not None:
    args.maxNodesPerHop = int(args.maxNodesPerHop)
```

### 3、读取数据
```
g = graph.Graph()
g.read_edgelist(filename=args.dataName, weighted=args.weighted, directed=args.directed)
g.read_node_status(filename=args.labelName)
```

### 4、获取全图节点的Embedding
```
embed_args = cmd_embed.parse_args() 
embeddings = embeddings.learn_embeddings(g, embed_args)
node_information = embeddings
#print node_information 
```

### 5、正负节点采样
```
train, train_status, test, test_status = sample.sample_single(g, args.testRatio, max_train_num=args.maxTrainNum)
```

### 6、抽取节点对的封闭子图
```
net = until.nxG_to_mat(g)
#print net
train_graphs, test_graphs, max_n_label = subgraphs.singleSubgraphs(net, train, train_status, test, test_status, args.hop, args.maxNodesPerHop, node_information)
print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))
```

### 7、加载网络模型，并在classifier中配置相关参数
```
cmd_args = cmd_opt.parse_args()
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = node_information.shape[1]
cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]
model = classifier.Classifier(cmd_args)
optimizer = optim.Adam(model.parameters(), lr=args.learningRate)
```

### 8、训练和测试
```
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
```

### 9、运行结果
```
average test of epoch 0: loss 0.62392 acc 0.71462 auc 0.72314
loss: 0.51711 acc: 0.80000: 100%|███████████████████████████████████| 76/76 [00:07<00:00, 10.09batch/s]
average training of epoch 1: loss 0.54414 acc 0.76895 auc 0.77751
loss: 0.37699 acc: 0.79167: 100%|█████████████████████████████████████| 9/9 [00:00<00:00, 34.07batch/s]
average test of epoch 1: loss 0.51981 acc 0.78538 auc 0.79709
loss: 0.43700 acc: 0.84000: 100%|███████████████████████████████████| 76/76 [00:07<00:00,  9.64batch/s]
average training of epoch 2: loss 0.49896 acc 0.79184 auc 0.82246
loss: 0.63594 acc: 0.66667: 100%|█████████████████████████████████████| 9/9 [00:00<00:00, 28.62batch/s]
average test of epoch 2: loss 0.48979 acc 0.79481 auc 0.83416
loss: 0.57502 acc: 0.76000: 100%|███████████████████████████████████| 76/76 [00:07<00:00,  9.70batch/s]
average training of epoch 3: loss 0.50005 acc 0.77447 auc 0.79622
loss: 0.38903 acc: 0.75000: 100%|█████████████████████████████████████| 9/9 [00:00<00:00, 34.03batch/s]
average test of epoch 3: loss 0.41463 acc 0.81132 auc 0.86523
loss: 0.54336 acc: 0.76000: 100%|███████████████████████████████████| 76/76 [00:07<00:00,  9.57batch/s]
average training of epoch 4: loss 0.44815 acc 0.81711 auc 0.84530
loss: 0.44784 acc: 0.70833: 100%|█████████████████████████████████████| 9/9 [00:00<00:00, 28.62batch/s]
average test of epoch 4: loss 0.48319 acc 0.81368 auc 0.84454
loss: 0.36999 acc: 0.88000: 100%|███████████████████████████████████| 76/76 [00:07<00:00, 10.17batch/s]
average training of epoch 5: loss 0.39647 acc 0.84184 auc 0.89236
loss: 0.15548 acc: 0.95833: 100%|█████████████████████████████████████| 9/9 [00:00<00:00, 28.62batch/s]
average test of epoch 5: loss 0.30881 acc 0.89623 auc 0.95132

```

## 四、引用
[图节点表示学习框架:    https://github.com/thunlp/OpenNE/](https://github.com/thunlp/OpenNE/)<br>
[节点状态预测-DeepInf:    https://github.com/xptree/DeepInf/](https://github.com/xptree/DeepInf/)<br>
[链接预测-SEAL:    https://github.com/muhanzhang/SEAL/](https://github.com/muhanzhang/SEAL/)<br>
