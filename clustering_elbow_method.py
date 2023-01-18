import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

if __name__=='__main__':
    pickle_reduced_path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/OpenFoodFacts/reduced_features.pkl"

    # Load reduced features from pickle file
    with open(pickle_reduced_path, "rb") as input_file:
        data_reduced = pickle.load(input_file)
    
    X = np.array([i for i in data_reduced.values()])

    distortions = []
    K = range(1, 40)
    for k in tqdm(K):
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(K, distortions, 'x-')
    ax.set_xlabel('k')
    ax.set_ylabel('Distortion')
    ax.set_title('The Elbow Method showing the optimal k')
    fig.show()