import os
import glob
import argparse
import matplotlib
import numpy as np

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
# from utils import predict_, load_images, display_images
from matplotlib import pyplot as plt


# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model_path = "./nyu.h5"
model = load_model(model_path, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(model_path))

def depth_images(image_files, batch_size):
    """
    Func: Returns Depth Maps
    """
    dpmap = np.empty([len(image_files), 224,224,1], dtype=np.float) # returns an empty of given shape without any entries
    for i in range(len(image_files)//batch_size):
        inputs = load_images(image_files)
        dpmap[i*batch_size:(i+1)*batch_size,...] = predict_(model, inputs)
        return dpmap