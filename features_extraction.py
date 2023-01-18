import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import pickle

from functions import extract_features
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm

if __name__=='__main__':
    path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/images"
    # change the working directory to the path where the images are located
    os.chdir(path)

    # this list holds all the image filename
    images = [i for i in os.listdir(path) if i.endswith('.jpg')]

    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    data = {}
    pickle_path = r"/home/kevin.bouchaud@Digital-Grenoble.local/code/Data_For_Good/OpenFoodFacts/features.pkl"

    for image in tqdm(images):
        feat = extract_features(image, model)
        data[image] = feat

    # Save features dictionary in pickle format
    with open(pickle_path, 'wb') as file:
        pickle.dump(data, file)