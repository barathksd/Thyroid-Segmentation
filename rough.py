

import numpy as np
import cv2
import sys
#sys.path.append('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles')
import dataprep
from dataprep import cut,disp
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
import gc
import os

f1 = 'C:\\Users\\AZEST-2019-07\\Downloads\\Image002.jpg'
fxml = 'C:\\Users\\AZEST-2019-07\\Downloads\\annotations.xml'


def transfer(dbase):

    for path,subdir,files in os.walk(dbase):
        if len(files) ==0:
            print('nofile ',path,subdir)
        else:
            for file in files:
                if 'Image' in file:
                    ipath = path + '/'+ file
                    spath = os.path.dirname(path).replace('Data','Images') + '/' + os.path.basename(path)+'_'+file
                    #print(ipath,spath)
                    img = cv2.imread(ipath)[40:-40,20:-20,:]
                    cv2.imwrite(spath,img)
                #break

img0 = cv2.imread(f1)[40:-35,20:-15]
#disp(img)
img = np.uint8(cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY))

c = cv2.Canny(img,750,800)
#mouse(c)


kernel = np.ones((50,65),np.float32)/500
dst = cv2.filter2D(c,-1,kernel,borderType = cv2.BORDER_WRAP)
disp(img0)
ct = np.int32(np.mean(np.where(dst==np.max(dst)),axis=-1))
gc.collect()

t,b,l,r = 0,0,0,0
s = 0
while(s<4):
    s = 0
    if  dst[ct[0]-t,ct[1]] >= 85:
        t += 1
    else:
        s += 1
    if  dst.shape[0]>ct[0]+b and dst[ct[0]+b,ct[1]] >= 88:
        b += 1
    else:
        s += 1
    if  dst[ct[0],ct[1]-l] >= 88:
        l += 1
    else:
        s += 1
    if  dst.shape[1]>ct[1]+r and dst[ct[0],ct[1]+r] >= 88:
        r += 1
    else:
        s += 1

print(t,b,l,r)
sc = img0[ct[0]-t:ct[0]+b,ct[1]-l:ct[1]+r].copy()
#disp(sc)

m,n,d = sc.shape
sh = np.zeros((sc.shape[0],sc.shape[1]))

for i in range(m):
    for j in range(n):
        b,g,r = sc[i,j]
        if (not (r<100 or g<100)) and ((2*b < g and 2*b < r) or (r<200 and r<b-30 and r<g-10)):
            sh[i,j] = 255

#print(sc.sum(axis=-1)
# sc[:,int(sc.shape[1]/2)] = [200,100,50]
# sc[:,int(sc.shape[1]/2)-6] = [200,100,50]
# sc[10,:] = [200,100,50]
gc.collect()

dmin = 100
cmax = int(sc.shape[1]/2)-5
sc0 = cv2.cvtColor(sc,cv2.COLOR_BGR2GRAY)
for i in range(int(sc.shape[1]/2)-5,int(sc.shape[1]/2)+6):
    col = len(sc0[:,i])
    
    p = 0
    n = 0
    j = 10
    while j<col:
        #print(i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin)
        
        if (sc0[j,i] > 120) and (sc0[j-2,i] < 36) :
            n += 1
            if n == 2:
                p = j
            elif n==3:
                if j-p < dmin:
                    dmin = j-p
                    cmax = i
                print('break ',i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin,' ',j-p)
                break
            j += 3
        j += 1



if np.max(sh.sum(axis=0)) > np.max(sh.sum(axis=1)):
    cd = np.argmax(sh.sum(axis=0))
    print('Vertical',cd)
    sc[:,cd] = [50,200,50]
    sc[:,cmax] = [200,100,50]
    disp(sc,[sh])
else:
    print('Horizontal',np.argmax(sh.sum(axis=1)))

#hist = histo(img)
    
