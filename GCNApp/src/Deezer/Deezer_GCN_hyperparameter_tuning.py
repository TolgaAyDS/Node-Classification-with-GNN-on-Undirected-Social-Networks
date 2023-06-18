import pandas as pd
import torch
import numpy as np
import json

import dgl
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F

import time
from itertools import product

start_time = time.time()

def import_data(path):
    df = pd.read_csv(path)
    return df

def import_json_to_matrix(path):
    dictionary = json.load(open(path))
    dictionary = {str(k): [str(val) for val in v] for k, v in dictionary.items()}
    matrix = np.zeros((len(dictionary), 33000), dtype=int)
    for key in dictionary.keys():
        for value in range(len(dictionary[key])):
            matrix[int(key)][int(dictionary[key][value])] = 1
    matrix = matrix[:, np.any(matrix, axis=0)]
    return matrix

def creating_graph(validation=True):
    # To tensor
    node_features = torch.tensor(features_matrix).float()
    edges_src = torch.from_numpy(edges['node_1'].to_numpy())
    edges_dst = torch.from_numpy(edges['node_2'].to_numpy())
    node_labels = torch.from_numpy(targets['target'].to_numpy())

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

        return graph, node_features, node_labels, train_mask, val_mask, test_mask

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

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.activation(h)
        h = self.dropout1(h)
        h = self.conv2(g, h)
        h = self.activation(h)
        h = self.dropout2(h)
        return h

def train(g, model, features, labels, train_mask, val_mask, test_mask, epoch_number, weight_decay, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0

    for e in range(epoch_number):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
            e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

    return best_test_acc

def run(params, validation=True):
    # Creating the graph
    if validation:
        graph, node_features, node_labels, train_mask, val_mask, test_mask = creating_graph()
    else:
        graph, node_features, node_labels, train_mask, test_mask = creating_graph(validation=False)

    in_feats = graph.ndata['feat'].shape[1]
    num_classes = 2
    results = []

    for param in params:
        hidden_units, learning_rate, num_epochs, dropout_rate, weight_decay, activation = param

        model = GCN(in_feats, hidden_units, num_classes, dropout_rate)
        if activation == 'relu':
            model.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            model.activation = nn.LeakyReLU()

        print("Hyperparameters of the GCN model:")
        print(f"Input feature dimension: {model.conv1._in_feats}")
        print(f"Hidden feature dimension: {model.conv1._out_feats}")
        print(f"Number of classes: {model.conv2._out_feats}")
        print(f"Number of layers: {2}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Activation function: {activation}")

        print("Architecture of the GCN model:")
        print(model)

        print("Trainable parameters of the GCN model:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

        best_test_acc = train(graph, model, node_features, node_labels, train_mask, val_mask, test_mask, num_epochs, weight_decay, learning_rate)
        results.append((hidden_units, learning_rate, num_epochs, dropout_rate, weight_decay, activation, best_test_acc))

    return results

if __name__ == "__main__":
    # Importing datasets
    edges_path = '../../data/deezer_europe/deezer_europe_edges.csv'
    targets_path = '../../data/deezer_europe/deezer_europe_target.csv'
    features_path = '../../data/deezer_europe/deezer_europe_features.json'

    edges = import_data(edges_path)
    targets = import_data(targets_path)
    features_matrix = import_json_to_matrix(features_path)

    # Hyperparameter grid
    param_grid = {
        'hidden_units': [50, 5000],
        'learning_rate': [0.001, 0.1],
        'num_epochs': [20, 100],
        'dropout_rate': [0.1, 0.6],
        'weight_decay': [0.001, 0.1],
        'activation': ['relu', 'leakyrelu']
    }

    # Generate all combinations of hyperparameters
    hyperparams = list(product(param_grid['hidden_units'], param_grid['learning_rate'], param_grid['num_epochs'], param_grid['dropout_rate'], param_grid['weight_decay'], param_grid['activation']))

    # Running grid search
    results = run(hyperparams)

    # Print the results
    print("\nGrid Search Results:")
    table_columns = ['Hidden Units', 'Learning Rate', 'Num Epochs', 'Dropout Rate', 'Weight Decay', 'Activation', 'Best Test Accuracy']
    table_data = [(*params, acc) for params, acc in zip(hyperparams, results)]
    table = pd.DataFrame(table_data, columns=table_columns)
    print(table)

    # Runtime
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
