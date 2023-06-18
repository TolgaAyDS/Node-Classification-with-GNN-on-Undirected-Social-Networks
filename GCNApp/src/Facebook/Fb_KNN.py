# Importing libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

import time
start_time = time.time()

def import_data(path):
    df = pd.read_csv(path)
    return df

def import_json_to_matrix(path):
    dictionary = json.load(open(path))
    dictionary = {str(k): [str(val) for val in v] for k, v in dictionary.items()}
    matrix = np.zeros((len(dictionary), len(dictionary)), dtype=int)
    for key in dictionary.keys():
        for value in range(len(dictionary[key])):
            matrix[int(key)][int(dictionary[key][value])] = 1

    return matrix

def run():
    # Importing datasets
    edges_path = '../../data/Facebook/facebook_edges.csv'
    targets_path = '../../data/Facebook/facebook_target.csv'
    features_path = '../../data/Facebook/facebook_features.json'

    edges = import_data(edges_path)
    targets = import_data(targets_path)
    features_matrix = import_json_to_matrix(features_path)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_matrix, targets['page_type'], test_size=0.2, random_state=42)

    # Creating and training the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    run()

    # Runtime
    print("time elapsed: {:.2f}s".format(time.time() - start_time))