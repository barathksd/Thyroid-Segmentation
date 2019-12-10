# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:30:52 2019

@author: AZEST-2019-07
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


file_path = 'D:\VOCdevkit\VOC2012\ImageSets\\Segmentation'  # has train.txt, trainval.txt, val.txt
seg_path = 'D:\VOCdevkit\VOC2012\\SegmentationClass'  # has annotated images
img_path = 'D:\VOCdevkit\VOC2012\\JPEGImages'   # has raw images

colordict = dict({'[0 0 0]':0,'[  0   0 128]':1,'[  0 128   0]':2,'[  0 128 128]':3,'[128   0   0]':4,'[128   0 128]':5,'[128 128   0]':6,'[128 128 128]':7
                  ,'[ 0  0 64]':8,'[  0   0 192]':9,'[  0 128  64]':10,'[  0 128 192]':11,'[128   0  64]':12,'[128   0 192]':13,'[128 128  64]':14
                  ,'[128 128 192]':15,'[ 0 64  0]':16,'[  0  64 128]':17,'[  0 192   0]':18,'[  0 192 128]':19,'[128  64   0]':20,'[192 224 224]':21})

train = np.array(open(file_path+'\\train.txt').read().splitlines())
trainval = np.array(open(file_path+'\\trainval.txt').read().splitlines())
test = np.array(open(file_path+'\\val.txt').read().splitlines())

imgd = dict((i,cv2.imread(img_path+'\\'+i+'.jpg')) for i in train)

def convseg(seg_path,colordict):
    
    for path,subdir,files in os.walk(seg_path):
        for file in files:
            img = cv2.imread(seg_path+'\\'+file)
            i2 = np.zeros([img.shape[0],img.shape[1]])
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    #print(i,j,str(img[i,j]),img[i,j])
                    i2[i,j] = colordict[str(img[i,j])]
            cv2.imwrite('D:\VOCdevkit\VOC2012\\output\\'+file,i2)
            print('*' )

convseg(seg_path,colordict)


def train(x_train,y_train,x_test, y_test,model):

    tensorboard = TensorBoard(log_dir="C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\logs\\tb1")
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = np.reshape(x_train,(x_train.shape+(1,)))
    x_test = np.reshape(x_test,(x_test.shape+(1,)))
    
    print(' test... \n')
    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    
    model = nnBlock((28,28,1))
    print(model.summary())
        
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
       
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
    save_model(model)
    model.save_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    model.save('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\mymodel.h5')