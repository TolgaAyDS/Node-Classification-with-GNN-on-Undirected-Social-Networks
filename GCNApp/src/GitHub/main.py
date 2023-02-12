import time
import pandas as pd
import torch
import numpy as np
import json

import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
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

def creating_graph(validation=True):

    # To tensor
    node_features = torch.tensor(features_matrix).float()
    edges_src = torch.from_numpy(edges['id_1'].to_numpy())
    edges_dst = torch.from_numpy(edges['id_2'].to_numpy())
    node_labels = torch.from_numpy(targets['ml_target'].to_numpy())

    # Creating graph
    graph = dgl.graph((edges_src, edges_dst), num_nodes=targets.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels

    if validation:
        # Configuring graph
        n_nodes = targets.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

        # Add edges between each node and itself to preserve old node representations
        graph.add_edges(graph.nodes(), graph.nodes())

        return graph, node_features, node_labels, train_mask, val_mask

    else:
        # Configuring graph
        n_nodes = targets.shape[0]
        n_train = int(n_nodes * 0.8)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        test_mask[n_train:] = True

        graph.ndata['train_mask'] = train_mask
        graph.ndata['test_mask'] = test_mask

        # Add edges between each node and itself to preserve old node representations
        graph.add_edges(graph.nodes(), graph.nodes())

        return graph, node_features, node_labels, train_mask, test_mask



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
        self.layer1 = GCNLayer(37700, 5000)
        self.layer2 = GCNLayer(5000, 2)
        #self.layer3 = GCNLayer(300, 2)

    def forward(self, g, features):
        x0 = F.relu(self.layer1(g, features))
        #x1 = F.relu(self.layer2(g, x0))
        x2 = self.layer2(g, x0)
        return x2


def evaluate(model, g, features, labels, mask, logits):
    model.eval()
    with torch.no_grad():
        #logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        #correct = torch.sum(indices == labels)
        acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
        report = classification_report(indices, labels)
        return acc, report


def run(epoch_num=100,validation=True):

    # Creating the graph
    if validation:
        graph,node_features,node_labels,train_mask,val_mask=creating_graph()
    else:
        graph,node_features,node_labels,train_mask,test_mask=creating_graph(validation=False)

    for epoch in range(epoch_num):
        if epoch >=3:
            t0 = time.time()

        net.train()
        logits = net(graph, node_features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], node_labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        if validation:
            acc,report = evaluate(net, graph, node_features, node_labels, val_mask, logits)
        else:
            acc, report = evaluate(net, graph, node_features, node_labels, test_mask, logits)

        print("Epoch {:05d} | Loss {:.4f} | Validation Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))
        print('GCN Classification Report: \n {}'.format(report))


if __name__ == "__main__":

    # Importing datasets
    edges_path = '../../data/Github/git_edges.csv'
    targets_path = '../../data/GitHub/git_target.csv'
    features_path = '../../data/GitHub/git.json'

    edges = import_data(edges_path)
    targets = import_data(targets_path)
    features_matrix= import_json_to_matrix(features_path)

    # Massage passing
    gcn_msg = fn.copy_u(u='h', out='m')
    gcn_reduce = fn.mean(msg='m', out='h')

    # Creating network
    net = Net()
    print(net)

    # Defining Optimizer and stop parameter
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    dur = []

    # Running
    run(validation=False)
