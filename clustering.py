import pickle

import numpy as np
from functions import view_cluster
from sklearn.cluster import KMeans


if __name__=='__main__':
    pickle_reduced_path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/OpenFoodFacts/reduced_features.pkl"

    # Load reduced features from pickle file
    with open(pickle_reduced_path, "rb") as input_file:
        data_reduced = pickle.load(input_file)
    
    X = np.array([i for i in data_reduced.values()])

    # get a list of the filenames
    filenames = np.array(list(data_reduced.keys()))

    n_clusters = 8

    # cluster feature vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    kmeans.fit(X)


    # holds the cluster id and the images { id: [images] }
    groups = {}
    for file, cluster in zip(filenames, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    for cluster in groups:
        print(cluster, len(groups[cluster]))

    for cluster in groups:
        print("Cluster: ", cluster)
        view_cluster(groups, cluster, 10)