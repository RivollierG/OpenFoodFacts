import os
import pickle

import numpy as np
from sklearn.decomposition import PCA


if __name__=='__main__':
    n_components = 100

    path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/images"
    # change the working directory to the path where the images are located
    os.chdir(path)

    # this list holds all the image filename
    images = [i for i in os.listdir(path) if i.endswith('.jpg')]

    pickle_path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/OpenFoodFacts/features.pkl"
    pickle_reduced_path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/OpenFoodFacts/reduced_features.pkl"

    # Load features from pickle
    with open(pickle_path, "rb") as input_file:
        data = pickle.load(input_file)

    # get a list of the filenames
    filenames = np.array(list(data.keys()))

    # get a list of just the features
    feat = np.array(list(data.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1, 4096)

    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(feat)
    x = pca.transform(feat)

    data_reduced = {}
    for i, image in enumerate(images):
        data_reduced[image] = x[i]

    # Save reduced features dictionary in pickle format
    with open(pickle_reduced_path, 'wb') as file:
        pickle.dump(data_reduced, file)

    print(f"The first {n_components} components kept preserve about {pca.explained_variance_ratio_.cumsum()}% of the information held by the whole dataset.")