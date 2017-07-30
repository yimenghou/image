# -*- coding: utf-8 -*-
"""
Created on Thu May 05 12:29:57 2016

@author: Yimeng
"""

import cv2
import numpy as np
import matplotlib.pylab as plt
from functions.zeropadding import zeropadding
from functions.flt import *

path = 'C:\\dataspace\\singleimg\\3.bmp'
img = cv2.imread(path, 0)
img = np.float32(img)
rows,cols = img.shape
factor = 0.2
        
hist, bins = np.histogram(img, 16, [0,256])
maxhist = hist.argmax()
mass_greyscale = int((maxhist+0.5)*16)
bg_color = 0.4

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if 0.2 * mass_greyscale < img[i,j] < 1.2 * mass_greyscale:
            img[i,j] = max(img.max(0))*bg_color

img1 = zeropadding(img, (cols-1)*np.absolute(factor), (cols-1)*np.absolute(factor), int(max(img.max(0))*bg_color))#int((maxhist+0.5)*16) )
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        img1[i,j] += np.random.randint(-4,5)

row,col = img1.shape
pts_base = np.float32([[0,0],[row-1,0],[0,col-1]])
pts = np.float32([[0, -(col-1)*factor],[row-1,0],[0,col-1]])
M = cv2.getAffineTransform(pts_base, pts)
dst = cv2.warpAffine(img1, M,(col,row))

resize_factor = img.shape[0]/float(dst.shape[0])
down_img = cv2.resize(dst, (0,0), fx=resize_factor, fy = resize_factor, interpolation=cv2.INTER_AREA)
#plt.figure()
#plt.plot(hist, '-x')
plt.figure()
plt.subplot(131),plt.imshow(img1),plt.title('Input')
plt.subplot(132),plt.imshow(dst),plt.title('Output')
plt.subplot(133),plt.imshow(down_img),plt.title('Output')
plt.show()
       