'''
Final proj - Team #7
part-related attribute(4)
'''
import os
import h5py
import image
import operator
import tensorflow as tf
import argparse

import matplotlib.pyplot as plt
import time, pickle, pandas
import numpy as np
import keras
from PIL import Image
import glob

from keras.applications.vgg16 import VGG16
from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D ,Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
from keras import applications
from keras import utils
import keras.backend as K
from preprocessing import *
from VGG16_DeepFashion_part import *

parser = argparse.ArgumentParser(description='Multi-layer Cepstrum')
parser.add_argument('infile', type=str, help='input wav file')

if __name__ == "__main__":
    GPU_usage()

    args = parser.parse_args()
    model = load_model('./models/model_part_v3.h5')

    if os.path.isfile(args.infile):
        print('===Image predict===')
        img = Image.open(args.infile)
        img = img.resize((img_width, img_height))
        img = np.array(img)
        img = np.resize(img,(1, img_width, img_height, 3))
        pred = model.predict(img)
        print('file name : ',args.infile)
        predResult(pred)
    else:
        print(args.infile,': File not found!')
    