# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
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
from tensorflow.keras.callbacks import *
import cv2

#Reference libraries 
#sklearn.model_selection -> Gridsearch, train_test_split
#sklearn.preprocessing -> StandardScalar, One hot encoder, .. 
#sklearn.linear_model -> lasso, linear regression, stochastic gradient descent,
#sklearn.ensemble -> RandomForestClassifier, RandomForestRegressor,
#sklearn.metrics -> max_error, mean_absolute_error, f1_score, jaccard_score

input_size = (512, 512, 1)
n_filters = 64
cf_size = (3, 3)   # convolution filter size
mpf_size = (2, 2)  # max pooling size
s_size = (2, 2)    # stride size

#loads data and splits it into test and train
def load():
    (x_train, y_train), (x_test, y_test) = None,None,None,None
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = load()

# self defined loss
def ioc_loss(y_true,y_pred):                                                   
    
    # jaccard loss based on IoC
    def jaccard_distance(y_true, y_pred, smooth=100):                          
        """ Calculates mean of Jaccard distance as a loss function """
        intersection = K.sum(K.abs(y_true * y_pred))
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd =  (1 - jac) * smooth
        return tf.reduce_mean(jd)
    #returns cross entropy loss + log(jaccard loss)
    return K.mean(-1*y_true*K.log(y_pred+K.epsilon())) + K.log(jaccard_distance(y_true,y_pred)+K.epsilon())

# self defined F1 metric
def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    def recall(y_true, y_pred):
        
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# U-Net model 
def unet(input_size,n_filters,cfsize,mpfsize,ssize):
    
    inp = Input(input_size)
    C1 = Conv2D(n_filters,cfsize,strides=1,padding='same',activation='relu',name='C01')(inp)
    C1 = Conv2D(n_filters,cfsize,strides=1,padding='same',activation='relu',name='C1')(C1)
    M1 = MaxPooling2D(mpfsize,name='M1')(C1)
    
    C2 = Conv2D(n_filters*2,cfsize,strides=1,padding='same',activation='relu',name='C02')(M1)
    C2 = Conv2D(n_filters*2,cfsize,strides=1,padding='same',activation='relu',name='C2')(C2)
    M2 = MaxPooling2D(mpfsize,name='M2')(C2)
    
    C3 = Conv2D(n_filters*4,cfsize,strides=1,padding='same',activation='relu',name='C03')(M2)
    C3 = Conv2D(n_filters*4,cfsize,strides=1,padding='same',activation='relu',name='C3')(C3)
    M3 = MaxPooling2D(mpfsize,name='M3')(C3)
    
    C4 = Conv2D(n_filters*8,cfsize,strides=1,padding='same',activation='relu',name='C04')(M3)
    C4 = Conv2D(n_filters*8,cfsize,strides=1,padding='same',activation='relu',name='C4')(C4)
    M4 = MaxPooling2D(mpfsize,name='M4')(C4)
    D4 = Dropout(0.2,name='D4')(M4)
    
    C5 = Conv2D(n_filters*16,cfsize,strides=1,padding='same',activation='relu',name='C05')(D4)
    C5 = Conv2D(n_filters*16,cfsize,strides=1,padding='same',activation='relu',name='C5')(C5)
    D5 = GaussianDropout(0.4,name='D5')(C5)
    
    #..up convolution stage
    
    U1 = Conv2DTranspose(n_filters*8,cfsize,strides = ssize, padding='same',activation = 'relu',name='U1')(D5)
    A1 = concatenate([C4,U1],axis=3,name='A1')
    C6 = Conv2D(n_filters*8,cfsize,strides=1,padding='same',activation='relu',name='C06')(A1)
    C6 = Conv2D(n_filters*8,cfsize,strides=1,padding='same',activation='relu',name='C6')(C6)
    
    U2 = Conv2DTranspose(n_filters*4,cfsize,strides = ssize, padding='same',activation = 'relu',name='U2')(C6)
    A2 = concatenate([C3,U2],axis=3,name='A2')
    C7 = Conv2D(n_filters*4,cfsize,strides=1,padding='same',activation='relu',name='C07')(A2)
    C7 = Conv2D(n_filters*4,cfsize,strides=1,padding='same',activation='relu',name='C7')(C7)
    
    U3 = Conv2DTranspose(n_filters*2,cfsize,strides = ssize, padding='same',activation = 'relu',name='U3')(C7)
    A3 = concatenate([C2,U3],axis=3,name='A3')
    C8 = Conv2D(n_filters*2,cfsize,strides=1,padding='same',activation='relu',name='C08')(A3)
    C8 = Conv2D(n_filters*2,cfsize,strides=1,padding='same',activation='relu',name='C8')(C8)
   
    U4 = Conv2DTranspose(n_filters,cfsize,strides = ssize, padding='same',activation = 'relu',name='U4')(C8)
    A4 = concatenate([C1,U4],axis=3,name='A4')
    C9 = Conv2D(n_filters,cfsize,strides=1,padding='valid',activation='relu',name='C09')(A4)
    C9 = Conv2D(n_filters,cfsize,strides=1,padding='valid',activation='relu',name='C9')(C9)
    D9 = GaussianDropout(0.2,name='D9')(C9)
    C9 = Conv2D(2,cfsize,strides=1,padding='valid',activation='relu',name='C_9')(D9)
    D9 = GaussianDropout(0.25,name='D_9')(C9)
    
    C10 = Conv2D(1,(1,1),activation='sigmoid',name='C10')(D9)
    
    
    
    model = Model(inputs = inp, outputs=C10)
    model.summary()
    return model

def save_model(model):    
    json_string = model.to_json()
    open('model.json', 'w').write(json_string)
    
def load_model():
    model = model_from_json(open('model.json').read())
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss=man_loss,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


tensorboard = TensorBoard(log_dir='C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\logs\\tb1')


model = unet(input_size,n_filters,cf_size,mpf_size,s_size)
print(model.summary())
    
opt = keras.optimizers.Adam(learning_rate=0.0001)
   
model.compile(loss=ioc_loss,
                  optimizer=opt,
                  metrics=['accuracy',f1])

checkpointer = ModelCheckpoint(filepath="C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5", 
                               monitor = 'val_accuracy',
                               verbose=1, 
                               save_best_only=True)

history = model.fit(x_train, y_train,batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,callbacks=[tensorboard,checkpointer])

model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
model.save('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\shapes_cnn.h5')





