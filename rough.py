

import numpy as np
import cv2
import sys
#sys.path.append('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles')
import dataprep
import matplotlib.pyplot as plt

f1 = 'D:\\Ito data\\AI2\\07\\Image005.jpg'
f2 = 'C:\\Users\\AZEST-2019-07\\Desktop\\kyoto-japan.jpg'

def disp(i,img=None,img1=None):
    cv2.imshow('i',i)
    if not img is None:
        cv2.imshow('img',img)
    if not img1 is None:
        cv2.imshow('img1',img1)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def quality(img):
    
    img = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    ddepth = cv2.CV_8U
    laplacian = cv2.Laplacian(img, ddepth, ksize=3) 
    disp(laplacian)
    return laplacian

img = cv2.imread(f1)[40:-80,20:-20,:]
t,b,l,r = dataprep.cut(img)
img = img[t:b,l:r]
imgb = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

q = quality(imgb)
cl = 5000/max(q,2000) - 0.9
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
c0 = clahe.apply(np.uint8(img[:,:,0]))
c1 = clahe.apply(np.uint8(img[:,:,0]))
c2 = clahe.apply(np.uint8(img[:,:,0]))
img[:,:,0],img[:,:,1],img[:,:,2] = c0,c1,c2


disp(img,c)

ddepth = cv2.CV_8U
l1 = cv2.Laplacian(img, ddepth, ksize=3) 
l2 = cv2.Laplacian(c, ddepth, ksize=3) 
print(l1.var(),l2.var())

'''

def histo(img):
    img = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.hist(img.ravel(),256,[0,256])
    #plt.show()
    return hist

#hist = histo(img)
    
'''
