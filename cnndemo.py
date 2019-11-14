# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:44:05 2019

@author: AZEST-2019-07
"""

import os 
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, jaccard_score
from keras.datasets import mnist
from tensorflow.keras.callbacks import *
import sys
sys.path.append('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles')
from mnistprep import img_train,label_train,img_test,label_test

batch_size = 16
num_classes = 21
epochs = 1
#data_augmentation = True
#num_predictions = 20
#save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = 'keras_cifar10_trained_model.h5'

def load():
    x_train, y_train, x_test, y_test = img_train,label_train,img_test,label_test
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    indices = np.random.permutation(np.arange(x_train.shape[0]))
    
    
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = load()


def ioc_loss(y_true,y_pred):
    def jaccard_distance(y_true, y_pred, smooth=100):
        """ Calculates mean of Jaccard distance as a loss function """
        intersection = K.sum(K.abs(y_true * y_pred))
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd =  (1 - jac) * smooth
        return tf.reduce_mean(jd)
    return K.mean(-1*y_true*K.log(y_pred+1/(2^20))) + K.log(jaccard_distance(y_true,y_pred)+1/(2^20))

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def nnBlock(input_shape):
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(168))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    return model

def save_model(model):    
    json_string = model.to_json()
    open('model.json', 'w').write(json_string)
    
def load_model2():
    model = load_model('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\mymodel.h5')
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def train(x_train,y_train,x_test,y_test):

    tensorboard = TensorBoard(log_dir="C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\logs\\tb1")
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
#    x_train = np.reshape(x_train,(x_train.shape+(1,)))
#    x_test = np.reshape(x_test,(x_test.shape+(1,)))
    
    print(' test... \n')
    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    
    model = nnBlock((28,28,1))
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights2.hdf5')
    print(model.summary())
        
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
       
    model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(filepath="C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5", 
                                   monitor = 'val_accuracy',
                                   verbose=1, 
                                   save_best_only=True)
    
    history = model.fit(x_train, y_train,batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,callbacks=[tensorboard,checkpointer])
    save_model(model)
    model.save_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    model.save('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\mymodel.h5')

train(x_train,y_train,x_test,y_test)








