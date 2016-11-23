# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:09:47 2016

@author: houyimeng
"""

# create a row/column of digits

from numpy import zeros,reshape,array,random
from loadImgData import loadImgData
from normalize import normalize

def lineIMAGE(numDigits, source, patcharm=28, edgesize=0, DigitLane = 5, Lane = 'h', norm_flag = 0):

    num_row = int((numDigits-1)/DigitLane)+1     
    rowpixel = num_row*patcharm + (num_row-1)*edgesize
    truelabelmatrix = zeros((numDigits)) 
    
    if numDigits > DigitLane:
        colpixel = DigitLane*patcharm +edgesize*(DigitLane - 1)
    else:
        colpixel = numDigits*patcharm + (numDigits-1)*edgesize
        
    itemTr, labelTr, itemTe, labelTe = loadImgData(source)
    idxvector = random.choice(itemTe.shape[0], numDigits, replace=False)
    
    if Lane == 'h':
        canvas = zeros((rowpixel, colpixel))
        matrixsize = array([rowpixel, colpixel])
    elif Lane == 'v':
        canvas = zeros((colpixel, rowpixel))
        matrixsize = array([colpixel, rowpixel]) 
   
    if norm_flag != 0:
       itemTe = normalize(itemTe)
     
    adder = 0                
    for j in idxvector:

        img = itemTe[j,:]
        label = reshape( labelTe[j,:], (10,1))

        truelabelmatrix[adder] = label.argmax()
        temp = reshape(img, (patcharm, patcharm))
        
        if Lane == 'h':
            canvas[(patcharm+edgesize)*int(adder/DigitLane):(patcharm+edgesize)*int(adder/DigitLane)+patcharm , \
                   (patcharm+edgesize)*(adder%DigitLane):(patcharm+edgesize)*(adder%DigitLane)+patcharm] = temp
        elif Lane == 'v':
            canvas[(patcharm+edgesize)*(adder%DigitLane):(patcharm+edgesize)*int(adder%DigitLane)+patcharm , \
                   (patcharm+edgesize)*int(adder/DigitLane):(patcharm+edgesize)*int(adder/DigitLane)+patcharm] = temp    
        adder += 1
        
    return canvas, array(truelabelmatrix, dtype='int'), matrixsize
