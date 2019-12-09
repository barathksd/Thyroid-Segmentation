# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:14:37 2019

@author: AZEST-2019-07
"""

import cv2
import numpy as np
import sys
import os
import cv2
import json
import pydicom
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import model_from_json, load_model
import tqdm

base = 'D:\\Ito data\\'

dicom_path = base + 'AI'
jpg_path = base + 'AI2'
annotated_path = base + 'annotated'
overlap_path = base + 'overlap'
final_dim = 6

sample = None
imgdata = None

def loadimg(fpath,ftype):
    
    global sample
    imgdict = {}
    if ftype == 'dicom':
        for path,subdir,files in os.walk(fpath):
            name = os.path.basename(path)
            imglist = []
            for file in files:
                full_path = path+ '\\' + file
                if int(file.replace('Image',''))%2 != 0:
                    imgdata = pydicom.read_file(full_path)
                    if sample == None:
                        sample = imgdata
                    img = imgdata.pixel_array
                    imglist.append(img[40:-40,:,:])
                    
            if len(imglist) != 0:
                imgdict[name] = imglist
                
    elif ftype == 'jpg':
        for path,subdir,files in os.walk(fpath):
            name = os.path.basename(path)
            imglist = []
            for file in files:
                full_path = path+ '\\' + file
                
                if '.jpg' in full_path and 'red' in full_path:
                    img = cv2.imread(full_path)
                    imglist.append(img[40:-40,:,:])
                
            if len(imglist) != 0:
                imgdict[name] = imglist
                
    elif ftype == 'annotation':
        print('annotation')
        m = '01'
        for path,subdir,files in os.walk(fpath):
            for file in files:
                full_path = path+'\\'+file
                
                if m != file[:2]:
                    imgdict[m] = imglist
                    m = file[:2]
                    imglist = []
                    
                imglist.append(cv2.imread(full_path))
            imgdict['07'] = imglist
            
    return imgdict

imgd = loadimg(dicom_path,'dicom')



def load_model2():
    model = load_model('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\mymodel.h5')
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

# Extract the pixels of the scale and measure the distance if there is a number beside it
def getdist(img,col,top,bottom,flist):
   # cnn model trained on decimal mnist
    model = load_model2()

     # maps model output to numbers
    num_dict = {}
    for i in range(10):
        num_dict[i] = i
        num_dict[10+i] = i+0.5

    num_dict[20] = -1
    
    colslice = img[max(top-10,0):min(flist[0]+30,bottom),col:col+1]
    mask = cv2.inRange(img,100,255)
    img[mask == 0] = 0
    img = img[max(top-10,0):min(flist[0]+30,bottom),:]
    print(colslice.shape)
    row,column = colslice.shape
    peak = []
    pr = 0  #row which has the point
    corr = 0
    for r in range(row):    
        if colslice[r,0] >= 140:
            pr = r
            corr += 1
        elif pr!=0:
            peak.append(pr-round(corr/2))
            corr = 0
            pr = 0
    
    #print(peak,i1.shape)
    d = 0
    lor = ''    # left side(l) or right side(r)
    dist = []
    l0 = 0
    r0 = 0
    
    for pos in peak:
        lside = img[pos-14:pos+14,col-29:col-1]
        rside = img[pos-14:pos+14,col+1:col+29]
        if lside.shape == (28,28) and rside.shape == (28,28):
            l = num_dict[np.argmax(model.predict(lside.reshape(1,28,28,1)/255))]
        
            r = num_dict[np.argmax(model.predict(rside.reshape(1,28,28,1)/255))]
           
            if l == 0 and lor=='':
                lor = 'l'
                d = pos
            elif r == 0 and lor=='':
                lor = 'r'
                d = pos
            elif lor=='l' and (l == 1 or l == 2 or l == 0.5):
                #print((pos-d)/(l-l0),m,l,l0,pos)
                dist.append((pos-d)/(l-l0))
                d = pos
                l0 = l
            elif lor=='r' and (r == 1 or r == 2 or r == 0.5):
                #print((pos-d)/(r-r0),m,r,r0,pos)
                dist.append((pos-d)/(r-r0))
                d = pos
                r0 = r
       
#        cv2.imshow('l',lside)
#        cv2.imshow('r',rside)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    return peak,np.average(dist)


# resizes image based on distance while maintaining aspect ratio
def img_resize(img,top,bottom,left,right,d_avg,final_shape):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img[top:bottom,left:right]
    m,n = img.shape
    
    m2,n2 = np.int32(np.round([m*120/d_avg,n*120/d_avg]))
    print(m,n,d_avg,final_shape,m2,n2)
    img = cv2.resize(img,(n2,m2))
    img = cv2.copyMakeBorder(img, int((final_shape-m2)/2), int((final_shape-m2+1)/2), int((final_shape-n2)/2), int((final_shape-n2+1)/2), cv2.BORDER_CONSTANT) 
    
    assert img.shape == (final_shape,final_shape)
    
    return img


# extract the outline of the annotated image
def create_image(imgd,img,color,k,index,c=5):
    top,bottom,left,right = cut(imgd[k][min(int(index/2),1,c)])          
    img = img[top:bottom,left:right]
    img2 = np.zeros(img.shape)
    r,c,d = img.shape
    #print(r,c,k,index,index/2)
    for i in range(r):
        for j in range(c):
            if (img[i,j]>[0,30,100]).sum()>=3 and (img[i,j]<[80,160,256]).sum()>=3:
                img2[i,j] = np.array(color)
    cv2.imwrite('D:\\Ito data\\annotated\\'+k+str(index)+'.png',img2)
#    cv2.imshow('img',img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    return img2

# extract the outline of all images by iteration
def create_map(imgd,imgj):
    for k,v in imgj.items():
        for i in range(len(v)):
            if i%2==0:
                color = [0,255,0]
            else:
                color = [0,255,255]
            create_image(imgd,v[i],color,k,i)



# add the thyroid and nodule images of a single patient
def add(i1,i2,k,index):
    r0,c0,d0 = i1.shape
    r1,c1,d1 = i2.shape
    
    r,c = min(r0,r1),min(c0,c1)
    
    i1 = i1[0:r,0:c,:]
    i2 = i2[0:r,0:c,:]
    
    i3 = cv2.add(i1,i2)
    cv2.imwrite('D:\\Ito data\\overlap\\'+k+str(index)+'.png',i3)



# augment the images with their mirror-image 
def flip(fpath,spath):
    for path,subdir,files in os.walk(fpath):
        for file in files:
            full_path = path+ '\\' + file
            flipimg = cv2.flip(cv2.imread(full_path), 1)
            cv2.imwrite(spath+'\\'+file+'.jpg',flipimg)


# mix the annotated output and input image
def fusion(img,img2,path,name):
    img3 = cv2.add(7*np.uint8(img/12),5*np.uint8(img2/12))
    img3[img2==0] = img[img2==0]
    cv2.imwrite(path+'\\'+name,img3)


# Convert color -> classes and classes -> one-hot vector
def one_hot(overlap_path,fd=6):   # 1-thyroid, 2-papillary, 3-benign, 4-cyst, 5-solid lesions 0-other
    #thyroid green 
    #outline yellow
    #papillary red 
    #benign blue 252,3,22
    #cyst violet 239,27,218
    #solid-lesion pale white 180,207,216
    
    for path, subdir, files in os.walk(overlap_path):
        fimg_list = []
        for file in files:
            full_path = path + '\\' + file
            img = cv2.imread(full_path)     
            #reshape(img)
            r,c,d = img.shape
            
            fimg = np.uint8(np.zeros([r,c]))
            # color -> classes
            for i in range(r):
                for j in range(c):
                    b,g,r = img[i,j]
                        
                    if b<60 and g<60 and r<60:       # black other 0
                        fimg[i,j] = 0
                    
                    elif g>2*r and g > 2*b:          # green thyroid 1
                        fimg[i,j] = 1
                        
                    elif r>2*g and r>2*b:            # red papillary 2
                        fimg[i,j] = 2 
                        
                    elif b>200 and b>3*g and b>3*r:  # blue benign 3 
                        fimg[i,j] = 3
                        
                    elif b>180 and r>180 and g<80:   # violet cyst 4
                        fimg[i,j] = 4
                        
                    elif r>180 and g>180 and b>150:  # pale white solid-lesion 5
                        fimg[i,j] = 5
                        
                    elif r>200 and g>200 and b<60:   # yellow outline 1
                        fimg[i,j] = 1
            
            #color -> one-hot vector
            fimg = to_categorical(fimg,fd)
            fimg_list.append(fimg)
            return fimg_list






