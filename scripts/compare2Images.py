# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:33:24 2016

@author: ThinkPad
"""
import os
import cv2
from numpy import zeros, array
import matplotlib.pylab as plt

path_pitch = ""
path_can = ""

dirs_pitch = os.listdir(path_pitch)
canvas = cv2.imread(path_can, 0)

Found = 0
coordinate = zeros((len(dirs_pitch), 2))

result = []
coor = []
n = 1
for item in dirs_pitch:                 
    img = cv2.imread(path_pitch+item, 0)
    img_content = array(img.flatten(), dtype='int')                
    for i1 in range(canvas.shape[0]-img.shape[0]+1):
        for i2 in range(canvas.shape[1]-img.shape[1]+1):
            if n%10000 == 0:
                print "Current x,y coordinate:", n
            img_temp = array(canvas[i1:i1+img.shape[0], i2:i2+img.shape[1]].flatten(), dtype='int')
            MSE = sum((img_temp-img_content)**2)/img_temp.size
            result.append( MSE )
            coor.append( [i1,i2] )
            n += 1

result = array(result)
minNUM = result.argmin()
min_coor = coor[minNUM]  

DEVI = img_content - array(canvas[min_coor[0]:min_coor[0]+img.shape[0], min_coor[1]:min_coor[1]+img.shape[1]].flatten(), dtype='int')
print DEVI

plt.figure(1)
plt.imshow(img)
plt.figure(2)
plt.imshow(canvas[min_coor[0]:min_coor[0]+img.shape[0], min_coor[1]:min_coor[1]+img.shape[1]])
plt.figure(3)
plt.imshow(img - canvas[min_coor[0]:min_coor[0]+img.shape[0], min_coor[1]:min_coor[1]+img.shape[1]])
plt.figure(4)
plt.plot(result)