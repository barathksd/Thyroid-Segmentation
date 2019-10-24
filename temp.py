# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend
import sklearn
import cv2

#sklearn.model_selection Gridsearch, .. 
#sklearn.preprocessing StandardScalar, One hot encoder, ..

file='C:\Users\AZEST-2019-07\Desktop\python_files'
