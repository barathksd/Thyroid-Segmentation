# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:36:05 2019

@author: AZEST-2019-07
"""

import cv2
import os
import numpy as np
import sys
import sklearn
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
from sklearn.cluster import KMeans


#import cnndemo


def load_model():
    model = model_from_json(open('model.json').read())
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def fusion_image(img1, img2):
    vis1 = cv2.resize(img1, (14,28))
    vis2 = cv2.resize(img2, (14,28))
    return np.array(cv2.hconcat([vis1, vis2]))

def add_point(img, sigma=1.5):
  col = np.random.randint(13,15)
  row = np.random.randint(22,25)
  rad = 1
  for x in range(row-rad, row+rad+1):
    for y in range(col-rad, col+rad+1):
      img[x][y] = np.exp(-((row-x)**2+(col-y)**2)/(2*sigma**2)) * 255
  return img


#cv2.imshow('img',cv2.resize(fusion_image(x_train[0], x_train[5]), (140, 140)))
#cv2.imshow('img2',cv2.resize(add_point(fusion_image(x_train[0], x_train[5])), (140, 140)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

train_label = [np.where(y_train==n)[0] for n in range(10)]
test_label = [np.where(y_test==n)[0] for n in range(10)]

# I is the number of extra train data
# J is the number of extra test data
I = 60000
classes = 20

x_extra_train = np.ndarray((I,28,28))
y_extra_train = np.ndarray((I))

for i in range(I):
    n = np.random.randint(10)
    plc_o = np.random.choice(train_label[n])
    plc_u = np.random.choice(train_label[5])
    x_extra_train[i] = np.uint8(add_point(fusion_image(x_train[plc_o], x_train[plc_u])))
    y_extra_train[i] = 10+n

J = 10000
x_extra_test = np.ndarray((J,28,28))
y_extra_test = np.ndarray((J))

for j in range(J):
    n = np.random.randint(10)
    plc_o = np.random.choice(test_label[n])
    plc_u = np.random.choice(test_label[5])
    x_extra_test[j] = np.uint8(add_point(fusion_image(x_test[plc_o], x_test[plc_u])))
    y_extra_test[j] = 10+n

num_dict = {}
for i in range(10):
    num_dict[i] = i
    num_dict[10+i] = i+0.5

#check = np.random.randint(I)
#cv2.imshow('img3',np.uint8(cv2.resize(x_extra_train[check],(140,140))))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def image_augument(img):
    re_size = np.random.choice([s for s in range(14,30,2)])
    left = int(np.random.choice([0, (28-re_size)/2, 28-re_size]))
    top = int(np.random.choice([0, (28-re_size)/2, 28-re_size]))
    re_img = np.zeros((28,28))
    re_img[top:top+re_size, left:left+re_size] = cv2.resize(img, (re_size, re_size))
    return re_img


#check = np.random.randint(I)
#cv2.imshow('img4',np.uint8(cv2.resize(image_augument(x_extra_train[check]),(140,140))))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

aug_x_train = np.zeros((60000,28,28))
aug_x_extra_train = np.zeros((I,28,28))

for i in range(60000):
    aug_x_train[i] = image_augument(x_train[i])
for i in range(I):
    aug_x_extra_train[i] = image_augument(x_extra_train[i])

aug_x_test = np.zeros((10000,28,28))
aug_x_extra_test = np.zeros((J,28,28))

for j in range(10000):
    aug_x_test[j] = image_augument(x_test[j])
for j in range(J):
    aug_x_extra_test[j] = image_augument(x_extra_test[j])
  
train_order = np.random.permutation([i for i in range(60000+I)])
img_train = np.concatenate([aug_x_train, aug_x_extra_train])[train_order].reshape(60000+I,28,28,1)[:100000]
label_train = np.concatenate([y_train, y_extra_train])[train_order].reshape(60000+I)[:100000]

test_order = np.random.permutation([j for j in range(10000+J)])
img_test = np.concatenate([aug_x_test, aug_x_extra_test])[test_order].reshape(10000+J,28,28,1)[:16000]
label_test = np.concatenate([y_test, y_extra_test])[test_order].reshape(10000+J)[:16000]


def newimg():
    isides = {}
    ilabels = {}
    path = 'C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles'
    for path,subdir,files in os.walk(path):
        for file in files:
            if 'index' in file:
                fp = path + '\\' + file
                index = int(file.replace('index','').replace('.jpg',''))
                isides[index] = cv2.cvtColor(cv2.imread(fp),cv2.COLOR_BGR2GRAY)
                ilabels[index] = 21
        break
    for i in [1,30,62]:
        ilabels[i] = 0
    for i in [5,38,70]:
        ilabels[i] = 1
    for i in [9,46,78]:
        ilabels[i] = 2
    for i in [13,54]:
        ilabels[i] = 3
    
    keys = np.array(list(ilabels.keys()))
    keys = np.repeat(keys,250)
    keys = np.random.permutation(keys)
    trset = np.array([isides[k] for k in keys])
    trlab = np.array([ilabels[k] for k in keys])
    
    return trset,trlab

trset,trlab = newimg()
trset = trset.reshape(16250,28,28,1)

img_train = np.concatenate((img_train,trset),axis=0)
label_train = np.concatenate((label_train,trlab))

img_test = np.concatenate((img_test,trset[:6000]),axis=0)
label_test = np.concatenate((label_test,trlab[:6000]))

