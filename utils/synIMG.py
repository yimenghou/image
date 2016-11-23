# -*- coding: utf-8 -*-
"""
Created on Fri May 06 09:29:18 2016

@author: ThinkPad
"""

import cv2
import os
import numpy as np
import matplotlib.pylab as plt
from prep import zeropadding
from flt import *

def synIMG(inputIMG, outputIMG, factor, bg_color):
    #factor = [-0.1, 0.1, 0.2, 0.3, 0.4]
    #bg_color = 0~1    
    
    img = cv2.imread(inputIMG, 0)
    
    if img.size != 784:
        return None
    
    img = np.float32(img)
    rows, cols = img.shape
            
    hist, bins = np.histogram(img, 16, [0,256])
    maxhist = hist.argmax()
    mass_greyscale = int((maxhist+0.5)*16)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0.2 * mass_greyscale < img[i,j] < 1.2 * mass_greyscale:
                img[i,j] = max(img.max(0))*bg_color
                
    img1 = zeropadding(img, (cols-1)*np.absolute(factor), (cols-1)*np.absolute(factor), int(max(img.max(0))*bg_color))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            img1[i,j] += np.random.randint(-4,5)
    
    row,col = img1.shape
    pts_base = np.float32([[0,0],[row-1,0],[0,col-1]])
    pts = np.float32([[0, -(col-1)*factor],[row-1,0],[0,col-1]])
    M = cv2.getAffineTransform(pts_base, pts)
    dst = cv2.warpAffine(img1, M,(col,row))
    
    resize_factor = img.shape[0]/float(dst.shape[0])
    down_img = cv2.resize(dst, (0,0), fx=resize_factor, fy = resize_factor, interpolation=cv2.INTER_NEAREST)

    for i in range(down_img.shape[0]):
        for j in range(down_img.shape[1]):
            if img[i,j] > 255:
                img[i,j] = 255
            elif img[i,j] < 0:
                img[i,j] = 0
    
    cv2.imwrite(outputIMG, down_img)
    
    return np.uint8(down_img)

# main functions goes here   
outpath = 'C:\\dataspace\\output_examples\\' 
inpath = 'C:\\dataspace\\input_examples\\' 
items = os.listdir(inpath)
container = []
c_cout = 0
for item in items:
    for x in np.arange(-0.1, 0.3, 0.1):
        for y in np.arange(0.2, 0.9, 0.15):
            print item, x, y
            img_name = str(c_cout)+'syn'+'.jpg'
            outputIMG = synIMG(inpath+item, outpath+img_name, x, y)
            container.append(outputIMG)
            c_cout += 1
            


