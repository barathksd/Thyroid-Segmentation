# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:06:33 2019

@author: AZEST-2019-07
"""
import importlib
import os 
import sys
import numpy as np
import cv2

code_path = 'C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\thyroid segmentation'
if np.sum([code_path in x for x in sys.path])==0:
    sys.path.append(code_path)
importlib.import_module('PatientClass')
from PatientClass import *
from Preprocess import *

path = 'D:\\Ito data\\AI2\\01\\Image003.jpg'
img = cv2.imread(path)[40:-40,:]
cv2.imwrite('D:\\Ito data\\034.png',img)

p1 = Patient([img],0,'01')
p1.cut(cut(p1.orgimage[0]))
t,b,l,r = p1.top,p1.bottom,p1.left,p1.right

scol,flist = extscale(p1.orgimage[0],p1.top,p1.bottom,p1.left)

peak,dist = getdist(cv2.cvtColor(p1.orgimage[0],cv2.COLOR_BGR2GRAY),scol,p1.top,p1.bottom,flist)
p1.dist(scol,dist)

p1.orgimage.append(img_resize(img,p1.top,p1.bottom,p1.left,p1.right,p1.scale,512))
Patient.display(p1.orgimage[1])







#img = cv2.imread('D:\\Ito data\\AI2\\07\\Image002 -thyroid_red.jpg')[40:-40,:][t:b,l:r]
#img2 = np.zeros((b-t,r-l,3))
#r,c,d = img2.shape
#print(img2.shape,img.shape,t,b,l,r)
#for i in range(r):
#    for j in range(c):
#        if (img[i,j]>[0,30,100]).sum()>=3 and (img[i,j]<[80,160,256]).sum()>=3:
#            color = [0,255,0]
#            img2[i,j] = np.array([0,255,255])
#cv2.imwrite('D:\\Ito data\\annotated\\071.png',img2)
#
#add(np.float64(cv2.imread('D:\\Ito data\\annotated\\070.png')),img2,'07',0)



#img2 = cv2.imread('D:\\Ito data\\overlap\\070.png')
#img3 = cv2.add(7*np.uint8(img/12),5*np.uint8(img2/12))
#img3[img2==0] = img[img2==0]
#cv2.imwrite('D:\\Ito data\\023.png',img3)


















#img = cv2.imread('D:\\Ito data\\014.png')
#img2 = cv2.imread('D:\\Ito data\\overlap\\010.png')
#
#img3 = cv2.add(7*np.uint8(img/12),5*np.uint8(img2/12))
#img3[img2==0] = img[img2==0]
#cv2.imwrite('D:\\Ito data\\013.png',img3)