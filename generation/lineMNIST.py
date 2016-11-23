# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:09:36 2016

@author: houyimeng
"""

# create a row/column of digits

from numpy import zeros,reshape,array,random
from MNISTDataset import MNISTDataset
from functions.preprocessing import normalize

def lineMNIST(numDigits, edgesize = 0, DigitLane = 10, Lane = 'h', norm_flag = 0):

    num_row = int((numDigits-1)/DigitLane)+1     
    rowpixel = num_row*28+ (num_row-1)*edgesize
    truelabelmatrix = zeros((numDigits)) 
    
    if numDigits > DigitLane:
        colpixel = DigitLane*28 +edgesize*(DigitLane - 1)
    else:
        colpixel = numDigits*28 + (numDigits-1)*edgesize
        
    dataset = MNISTDataset('C:\\dataspace\\MNIST')
    content_a = zeros((10000, 784))
    content_b = zeros((10000, 10))
    
    for ii in range(10000):
        label_temp, data_temp = dataset.getTestingItem( ii )
        content_a[ii,:] = data_temp.flatten()
        content_b[ii,:] = label_temp.flatten()
        
    if norm_flag != 0:        
        content_a = normalize(content_a)
        
    idxvector = random.choice(10000, numDigits, replace=False)
    
    if Lane == 'h':
        canvas = zeros((rowpixel, colpixel))
        matrixsize = array([rowpixel, colpixel])
    elif Lane == 'v':
        canvas = zeros((colpixel, rowpixel))
        matrixsize = array([colpixel, rowpixel]) 
        
    adder = 0                
    for j in idxvector:

        truelabelmatrix[adder] = content_b[j].argmax()
        temp = reshape(content_a[j], (28,28))
        
        if Lane == 'h':
            canvas[(28+edgesize)*int(adder/DigitLane):(28+edgesize)*int(adder/DigitLane)+28 , \
                   (28+edgesize)*(adder%DigitLane):(28+edgesize)*(adder%DigitLane)+28] = temp
        elif Lane == 'v':
            canvas[(28+edgesize)*(adder%DigitLane):(28+edgesize)*int(adder%DigitLane)+28 , \
                   (28+edgesize)*int(adder/DigitLane):(28+edgesize)*int(adder/DigitLane)+28] = temp    
        adder += 1
        
    return canvas, array(truelabelmatrix, dtype='int'), matrixsize