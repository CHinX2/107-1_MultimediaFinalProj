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

'''
很重要!!!!!
GPU記憶體用量設定
'''
def GPU_usage():
    # 使用GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 設定GPU記憶體的使用比例
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.tensorflow_backend.set_session(sess)


def VGG16_part() :

    if not os.path.isfile('./data/train_data_1000.npy'): # also can use train_data_20000.npy
        print('Data preprocessing...')
        if not os.path.isfile('./data/img_attr_part.csv'):
            print('CSV label loading...')
            LoadCSV()
        print('Image data loading...')
        data, label = DataPreprocessing(1000) #這裡要跑超久, 可以先減少資料量
        print('finish')
    else:
        print('Training data loading...')
        # load training data from preprocessing.py
        # 因為input data size要改成 224 * 224, 不能直接用助教給的
        data = np.load('./data/train_data_1000.npy')
        label = np.load('./data/train_label_1000.npy')
        print('finish')

    print('Start training...')

    print(data.shape)
    print(label.shape)

    # 90% train ; 10% test
    train_size = int(0.9*data.shape[0]) 

    train_data = data[0:train_size,:,:,:]
    train_label = label[0:train_size,:]   
    test_data = data[train_size:,:,:,:]
    test_label = label[train_size:,:]

    print('train_data shape:',train_data.shape)
    print('test_data shape:',test_data.shape)

    # 使用keras VGG-16架構，並自行接上Fully-connect layer (Dense) 來進行分類
    # 最後一層Fully-connect layer參數量則為分類數量
    # input data size = 224 * 224
    vgg16 = VGG16(include_top=True, weights='imagenet') # True : VGG 3 layer
    x = vgg16.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dense(1000, activation='relu')(x) #看要不要加
    x = Drop(0.5)(x)
    predict = Dense(216, activation='sigmoid')(x)

    # Define input and output of the model
    model = Model(inputs=vgg16.input, outputs=predict)

    if os.path.isfile('./models/model_weight_part_v3.h5'):
        model.load_weights('./models/model_part_weight_v3.h5')

    # Using Adam optimizer with lower learning rate
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #sgd = optimizers.SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['accuracy'])

    # Training the model for 5 epochs
    model.fit(
            x=train_data ,
            y=train_label,
            batch_size=32,
            epochs=10,
            validation_split=0.1,
            verbose=1
    )

    # 儲存訓練完成後的model權重，即完成model訓練
    # 當重新訓練後要更改檔名，否則會覆蓋掉之前train好的weights
    model.save_weights('./models/model_weight_part_v5.h5')
    model.save('./models/model_part_v5.h5')

    # testing model 需準備好test_data及test_label
    loss, accuracy = model.evaluate(x=test_data, y=test_label, batch_size=32, verbose=1)
    # if no label -> model.predict(test_data)
    print("Testing: accuracy = %f  ;  loss = %f" % (accuracy, loss))

    return model


if __name__ == '__main__':
    GPU_usage()

    model = VGG16_part()
