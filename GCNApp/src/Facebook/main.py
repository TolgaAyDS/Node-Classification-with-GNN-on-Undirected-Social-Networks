import urllib.request
import time
import numpy as np
import pandas as pd
import torch
import os
import numpy as np
import json

import dgl
from dgl.data import DGLDataset
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import tensorflow as tf

from sklearn.metrics import classification_report


def import_data(path):
    df=pd.read_csv(path)
    return df

def import_json_to_matrix(path):
    dictionary=json.load(open(path))
    dictionary = {str(k): [str(val) for val in v] for k, v in dictionary.items()}
    matrix = np.zeros((len(dictionary), len(dictionary)), dtype=int)
    for key in dictionary.keys():
        for value in range(len(dictionary[key])):
            matrix[int(key)][int(dictionary[key][value])] = 1

    return matrix

#
# features_dict = json.load(open(features_path))
# features_dict = {str(k): [str(val) for val in v] for k, v in features_dict.items()}
#
# features_matrix = np.zeros((len(features_dict), len(features_dict)), dtype=int)
#
# for key in features_dict.keys():
#     for value in range(len(features_dict[key])):
#         features_matrix[int(key)][int(features_dict[key][value])] = 1


# node_features_fb=torch.tensor(features_matrix).float()
# edges_src_fb = torch.from_numpy(edges['id_1'].to_numpy())
# edges_dst_fb = torch.from_numpy(edges['id_2'].to_numpy())
# node_labels_fb = torch.from_numpy(targets['page_type'].astype('category').cat.codes.to_numpy()).long()

# graph = dgl.graph((edges_src_fb, edges_dst_fb), num_nodes=targets.shape[0])
# graph.ndata['feat'] = node_features_fb
# graph.ndata['label'] = node_labels_fb

# n_nodes = targets.shape[0]
# n_train = int(n_nodes * 0.6)
# n_val = int(n_nodes * 0.2)
# train_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
# val_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
# test_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
# train_mask_fb[:n_train] = True
# val_mask_fb[n_train:n_train + n_val] = True
# test_mask_fb[n_train + n_val:] = True
# graph.ndata['train_mask'] = train_mask_fb
# graph.ndata['val_mask'] = val_mask_fb
# graph.ndata['test_mask'] = test_mask_fb

# print(graph)

# gcn_msg = fn.copy_u(u='h', out='m')
# gcn_reduce = fn.mean(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(22470, 160)
        self.layer2 = GCNLayer(160, 4)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


def evaluate(model, g, features, labels, mask):
    global logits
    model.eval()
    with torch.no_grad():
        #logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
        report = classification_report(indices, labels, target_names=targets['page_type'].drop_duplicates())
        return acc, report


if __name__ == "__main__":

    # Importing datasets
    edges_path = '../../data/Facebook/facebook_edges.csv'
    targets_path = '../../data/Facebook/facebook_target.csv'
    features_path = '../../data/Facebook/facebook_features.json'

    edges = import_data(edges_path)
    targets = import_data(targets_path)
    features_matrix= import_json_to_matrix(features_path)

    # To tensor
    node_features_fb = torch.tensor(features_matrix).float()
    edges_src_fb = torch.from_numpy(edges['id_1'].to_numpy())
    edges_dst_fb = torch.from_numpy(edges['id_2'].to_numpy())
    node_labels_fb = torch.from_numpy(targets['page_type'].astype('category').cat.codes.to_numpy()).long()

    # Creating graph
    graph = dgl.graph((edges_src_fb, edges_dst_fb), num_nodes=targets.shape[0])
    graph.ndata['feat'] = node_features_fb
    graph.ndata['label'] = node_labels_fb

    # Configuring graph
    n_nodes = targets.shape[0]
    n_train = int(n_nodes * 0.6)
    n_val = int(n_nodes * 0.2)
    train_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask_fb[:n_train] = True
    val_mask_fb[n_train:n_train + n_val] = True
    test_mask_fb[n_train + n_val:] = True
    graph.ndata['train_mask'] = train_mask_fb
    graph.ndata['val_mask'] = val_mask_fb
    graph.ndata['test_mask'] = test_mask_fb

    print(graph)

    # Massage passing
    gcn_msg = fn.copy_u(u='h', out='m')
    gcn_reduce = fn.mean(msg='m', out='h')

    # Creating network
    net = Net()
    print(net)

    # Add edges between each node and itself to preserve old node representations
    graph.add_edges(graph.nodes(), graph.nodes())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    dur = []

    # Running Validation
    for epoch in range(100):
        if epoch >=3:
            t0 = time.time()

        net.train()
        logits = net(graph, node_features_fb)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask_fb], node_labels_fb[train_mask_fb])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        acc,report = evaluate(net, graph, node_features_fb, node_labels_fb, val_mask_fb)
        print("Epoch {:05d} | Loss {:.4f} | Validation Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))

        print('GCN Classification Report: \n {}'.format(report))


    # Complete training and forecast

    graph = dgl.graph((edges_src_fb, edges_dst_fb), num_nodes=targets.shape[0])
    graph.ndata['feat'] = node_features_fb
    graph.ndata['label'] = node_labels_fb

    n_nodes = targets.shape[0]
    n_train = int(n_nodes * 0.8)
    #n_val = int(n_nodes * 0.1)
    train_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
    #val_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask_fb = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask_fb[:n_train] = True
    #val_mask_fb[n_train:n_train + n_val] = True
    test_mask_fb[n_train:] = True
    graph.ndata['train_mask'] = train_mask_fb
    #graph.ndata['val_mask'] = val_mask_fb
    graph.ndata['test_mask'] = test_mask_fb

    gcn_msg = fn.copy_u(u='h', out='m')
    gcn_reduce = fn.mean(msg='m', out='h')

    net = Net()
    print(net)

    # Add edges between each node and itself to preserve old node representations
    graph.add_edges(graph.nodes(), graph.nodes())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    dur = []

    for epoch in range(100):
        if epoch >=3:
            t0 = time.time()

        net.train()
        logits = net(graph, node_features_fb)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask_fb], node_labels_fb[train_mask_fb])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        acc,report = evaluate(net, graph, node_features_fb, node_labels_fb, test_mask_fb)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))

        print('GCN Classification Report: \n {}'.format(report))
