# -*- coding: utf-8 -*-
"""
Created on Tue May 03 11:51:00 2016

@author: ThinkPad
"""

#
import cv2
import os
from numpy import random

path = 'C:\\dataspace\\harbour\\canvas\\'
scaleFactor = 1.0
dirs = os.listdir(path)
rand_idx = random.randint(len(dirs))
img = cv2.imread(path+dirs[14], 0)

hog_canvas = cv2.HOGDescriptor((28, 28), (8, 8), (4,4), (4,4), 9)
hog_patch = cv2.HOGDescriptor((28, 28), (8, 8), (4,4), (4,4), 9)
     
hist_patch1 = hog_patch.compute(img[:28, :28])
hist_patch2 = hog_patch.compute(img[28:56, :28])
hist_patch3 = hog_patch.compute(img[:28, 28:56])
hist_canvas = hog_canvas.compute(img)