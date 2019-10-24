# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:28:45 2019

@author: AZEST-2019-07
"""


import skimage.io as io
import sys
import os
import cv2
import json
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import tqdm

dir_path = 'D:\\Ito data\\AI'

#loads data from folder and saves it in dict, key is patientID, value is images in list
def load(fpath):    
    imgd = {}
    for path,subdir,files in os.walk(fpath):
        name = os.path.basename(path)
        imglist = []
        for file in files:
            full_path = path+ '\\' + file
            imgdata = pydicom.read_file(full_path)
            img = imgdata.pixel_array
            #img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            imglist.append(img)
        imgd[name] = imglist
    return imgd
imgd = load(dir_path)

img = cv2.imread('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\images\\ultrasound\\demo2.jpg')
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

clr = np.random.rand(10,3)*255
maskcolor = dict((i+1,clr[i]) for i in range(10))


# resizes image while maintaining aspect ratio
def img_resize(img,final_shape):
    
    m,n = img.shape
    mx = max(m,n)
    ratio = final_shape/mx
    #print(ratio)
    rimg = cv2.resize(img,dsize=None,fx=ratio,fy=ratio)
    ax = 0
    i1 = np.uint8(np.zeros((final_shape-min(rimg.shape),final_shape)))
    #print(i1.shape)
    if m>n:
        i1 = i1.T
        ax = 1
    rimg = np.uint8(np.concatenate([i1,rimg],axis=ax))
    
    assert rimg.shape == (final_shape,final_shape)
    return rimg
    #print(rimg.shape)

points = []

def mousepoint(event,x,y,flags,param): 
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(imgd['01'][0],(x,y),1,(255,255,255),-1)
        points.append((x,y))
        print(x,y,'  --------')


editimg = None
def segment(img):
    global points,maskcolor,editimg
    editimg = np.uint8(np.zeros((img.shape[0],img.shape[1])))
    while(1):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',mousepoint)
        points = []
        while(1):
            cv2.imshow('image',img)
            if cv2.waitKey(20) & 0xFF == 27:
                #cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        if points != []:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1,1,2))
            
            mask = input('Enter the value ')
            clr = maskcolor[int(mask)]
            x = int(clr[0])
            y = int(clr[1])
            z = int(clr[2])
            
            cv2.polylines(img,[pts],True,color = (x,y,z))
            cv2.polylines(editimg,[pts],True,color = int(mask))
            cv2.fillConvexPoly(img, points=np.array(points), color=(x,y,z))
            cv2.fillConvexPoly(editimg, points=np.array(points), color=int(mask))
            
            
            print('Area ',cv2.contourArea(pts,oriented = False))
            print('Length ',cv2.arcLength(pts,closed=True))
            cv2.imshow('image',img)
            if cv2.waitKey(20) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
        else:
            cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\demo.jpg',img)
            cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\demo2.jpg',editimg)
            cv2.destroyAllWindows()
            break


segment(imgd['01'][0])

cv2.imshow('image2',editimg*30)
cv2.waitKey(0)
cv2.destroyAllWindows()


    