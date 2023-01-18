import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img
from sklearn.neighbors import NearestNeighbors

if __name__=='__main__':
    path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/images"
    # change the working directory to the path where the images are located
    os.chdir(path)

    # this list holds all the image filename
    images = [i for i in os.listdir(path) if i.endswith('.jpg')]

    pickle_reduced_path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/OpenFoodFacts/reduced_features.pkl"

    # Load reduced features from pickle file
    with open(pickle_reduced_path, "rb") as input_file:
        data_reduced = pickle.load(input_file)

    X = np.array([i for i in data_reduced.values()])

    neigh = NearestNeighbors(n_neighbors=10, n_jobs=-1)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)

    #key = '0041331124669'
    nb_neighbours = 5

    for key in np.random.choice(range(len(images)), size=5, replace=False):
        index = 0
        fig = plt.figure(figsize=(25, 25))
        im_neighbours = [images[key]] + [images[i] for i in indices[key][1:nb_neighbours+1]]
        for im in im_neighbours:
            plt.subplot(6, 6, index+1)
            img = load_img(im)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')
            if index==0:
                plt.title('Ref')
            index += 1