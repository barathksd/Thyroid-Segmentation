# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:11:51 2019

@author: AZEST-2019-07
"""

import numpy as np
import cv2
import statsmodels.api as sm
import matplotlib.pyplot as plt
from keras.models import load_model

#WIDTH = 30
HEIGHT = 10
num_dict = {}
for i in range(10):
    num_dict[i] = i
    num_dict[10+i] = i+0.5
    
    
img = cv2.imread('D:\\Ito data\\AI2\\02\\Image003 -nodule.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = cv2.inRange(img, 200, 250) / 255
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = img2
LAG = int(img.shape[0] / 4)

def acf(vector, nlags, r=False):
    acfs = np.zeros(len(vector))
    if not r:
        ts = np.concatenate([vector, vector[:nlags]])
        for i in range(len(vector)):
            acfs[i] = ts[i]*ts[i+nlags]
        return acfs
    elif r:
        ts = np.concatenate([vector[-nlags:], vector])
        for i in range(len(vector)):
            acfs[i] = ts[i+nlags]*ts[i]
        return acfs

autocorr = np.zeros((img.shape[1], LAG, img.shape[0]))


for lag in range(LAG):
    for col in range(img.shape[1]):
        autocorr[col][lag] = acf(img[:,col],nlags=lag,r=False)


v_autocorr = np.sum(autocorr, axis=2)
lag_weight = np.resize(np.array([w for w in range(1,LAG+1)]), (LAG, 1))
lg2 = np.ones((LAG,1))
max_col = np.argmax(np.sum(v_autocorr,axis = 1))
print(max_col)
#max_col = np.argmax(np.resize(np.dot(v_autocorr,lg2), img.shape[1]))

"""
                               
concent_weight = np.resize(v_autocorr[max_col], (LAG, 1))
peak = np.resize(np.dot(autocorr[max_col].T, concent_weight), img.shape[0])

row_weight = np.array([w for w in range(0,img.shape[0])])
top_row = np.argmax(peak * ((img.shape[0]-1)-row_weight))


r_autocorr = np.zeros((img.shape[1], LAG, img.shape[0]))
for lag in range(LAG):
    for col in range(img.shape[1]):
        r_autocorr[col][lag] = acf(img[:,col],nlags=lag,r=True)

r_peak = np.resize(np.dot(r_autocorr[max_col].T, concent_weight), img.shape[0])
bottom_row = np.argmax(r_peak * row_weight)

for delta_row in range(0, img.shape[0]-top_row):
    if img[top_row+delta_row][max_col] == 0:
        break
for delta_col in range(0, img.shape[1]-max_col):
    if np.sum(img[top_row-delta_row:top_row+delta_row, max_col+delta_col]) == 0:
        break

left_zone = img[max(top_row-HEIGHT,0):min(top_row+HEIGHT,img.shape[0]),max_col-delta_col-2*HEIGHT:max_col-delta_col]
right_zone = img[max(top_row-HEIGHT,0):min(top_row+HEIGHT,img.shape[0]),max_col+delta_col:max_col+delta_col+2*HEIGHT]

left_inp = np.resize(cv2.resize(left_zone,(28, 28), cv2.INTER_CUBIC), (1,28,28,1))
right_inp = np.resize(cv2.resize(right_zone,(28, 28), cv2.INTER_CUBIC), (1,28,28,1))

#model = load_model("mnist_cnn.hdf5")
model = load_model("decimal_mnist.h5")

left_res = model.predict(left_inp, batch_size=1)
right_res = model.predict(right_inp, batch_size=1)

print('left:{}\nright:{}'.format(left_res[0][0], right_res[0][0]))

if left_res[0][0] > right_res[0][0]:


    for next_top_row in range(top_row+HEIGHT,img.shape[0]):
        if np.sum(img[next_top_row, max_col-delta_col-2*HEIGHT:max_col-delta_col]) != 0:
            break
    for next_bottom_row in range(next_top_row,img.shape[0]):
        if np.sum(img[next_bottom_row, max_col-delta_col-2*HEIGHT:max_col-delta_col]) == 0:
            break
    
    next_center_row = int((next_top_row+next_bottom_row) / 2)
    zone = img[max(next_center_row-HEIGHT,0):min(next_center_row+HEIGHT,img.shape[0]),max_col-delta_col-2*HEIGHT:max_col-delta_col]
    inp = np.resize(cv2.resize(zone,(28, 28), cv2.INTER_CUBIC), (1,28,28,1))
    res = model.predict(inp, batch_size=1)
    num = np.argmax(res[0])
    dist = next_center_row - top_row

    print('next number is {}\nprobability is {}\ndistance is {} pixels'.format(num_dict[num], res[0][num], dist))

    
    pos = 'left'
    cv2.imshow('Left Zone', left_zone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
   
    cv2.imshow('Next Zone', zone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:

    for next_top_row in range(top_row+HEIGHT,img.shape[0]):
        if np.sum(img[next_top_row, max_col+delta_col:max_col+delta_col+2*HEIGHT]) != 0:
            break
    for next_bottom_row in range(next_top_row,img.shape[0]):
        if np.sum(img[next_bottom_row, max_col+delta_col:max_col+delta_col+2*HEIGHT]) == 0:
            break
    
    next_center_row = int((next_top_row+next_bottom_row) / 2)
    zone = img[max(next_center_row-HEIGHT,0):min(next_center_row+HEIGHT,img.shape[0]),max_col+delta_col:max_col+delta_col+2*HEIGHT]
    inp = np.resize(cv2.resize(zone,(28, 28), cv2.INTER_CUBIC), (1,28,28,1))
    res = model.predict(inp, batch_size=1)
    num = np.argmax(res[0])
    dist = next_center_row - top_row

    print('next number is {}\nprobability is {}\ndistance is {} pixels'.format(num_dict[num], res[0][num], dist))

    pos = 'right'
    cv2.imshow('Right Zone', right_zone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



"""
"""
plt.figure(figsize=(20, 10), dpi=50)
plt.imshow(autocorr.T, interpolation='nearest', vmin=0, cmap='jet', aspect=3.0)
plt.ylim(0, LAG)
plt.colorbar()
plt.show()
"""

"""
cv2.imshow('GSimg', img[max(top_row-HEIGHT,0):min(bottom_row+HEIGHT,img.shape[0]),max_col-WIDTH:max_col+WIDTH])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""