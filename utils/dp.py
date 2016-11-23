# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 13:55:06 2016

@author: ThinkPad
"""
import scipy
import numpy as np
import copy
import cv2
import matplotlib.pylab as plt
import time

#img = cv2.imread("D:\\paper\\image anlysis\\Lab2_DynamicProgramming\\view1.jpg", 0)
importdata = scipy.io.loadmat("D:\\paper\\image anlysis\\Lab2_DynamicProgramming\\R.mat")
img = np.array(importdata["R"], dtype='float')
#img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
# dynamic programming

def dp(InputCost):
    
    weight_smooth = 2
    edge = np.ones(InputCost.shape[1])
    paths = np.ones(InputCost.shape[0])  
    backtracing = np.ones(InputCost.shape)
    CAM = copy.deepcopy(InputCost)
    
    for i in range(1, InputCost.shape[1]):
        for j in range(InputCost.shape[0]):
            for k in range(InputCost.shape[0]):
                paths[k] = weight_smooth*np.absolute(j-k) + CAM[k, i-1] + CAM[j, i]
                
            CAM[j,i] = paths.min()  
            backtracing[j,i] = paths.argmin()
    
    
    edge[-1] = CAM[:,-1].argmin()
    
    for i in range(InputCost.shape[1]-2, -1, -1):
        edge[i] = backtracing[int(edge[i+1]), i+1]
        
    return edge
    
    
costMTX = np.array([[2,2,4,3,1],
                    [4,1,1,2,1],
                    [1,2,5,6,0],
                    [2,2,2,4,2]])

               
edge,cam = dp(img)



