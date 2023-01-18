import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

from tqdm import tqdm


def extract_features(file, model):
    """
    Function to extract features from images using
    a deep learning model whose last layer was removed

    Parameters
    ----------
    file: str
        Image name
    model: sklearn model
        Model without last layer
    
    Return
    ------
    features: list
        Features extracted from the image
    """
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


def view_cluster(groups, cluster, nb_im):
    """
    Function that lets you view a cluster (based on identifier)

    Parameters
    ----------
    groups: dictionary
        Object containing the image names grouped by cluster
        (key: cluster id, value: list of image names)
    cluster: int
        Identifier of the cluster you want to visualise
    nb_im: int
        Number of images to display per cluster

    Return
    ------
    None
    """
    plt.figure(figsize = (25, 25))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > nb_im:
        print(f"Clipping cluster size from {len(files)} to {nb_im}")
        files = files[:nb_im-1]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')