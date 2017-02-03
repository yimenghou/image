# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 16:12:26 2016

@author: yimeng
"""

import numpy as np
from numpy import ones, zeros, ceil, array
from preprocessing import zeropadding

def massflt(inputdata, flt_arm = 1,  pad_val = 0):
    #small region mass-number filter    
    
    threshold = (flt_arm*2+1)**2-7
    inputmatrix = zeropadding(inputdata, flt_arm, flt_arm, constant_val = pad_val)
    m, n = inputmatrix.shape
    outdata = -1*np.ones(inputdata.shape)
        
    for row in range(flt_arm, m-flt_arm):
        for col in range(flt_arm, n-flt_arm):
            data_window = inputmatrix[row-flt_arm:row+flt_arm+1, col-flt_arm:col+flt_arm+1].flatten()
            
            temp = data_window.tolist().count(-1)
            if temp <= threshold:
                outdata[row-flt_arm, col-flt_arm] = inputmatrix[row-flt_arm, col-flt_arm]
            else:
                continue
                
    return outdata
    
def meanflt(inputdata, flt_arm = 1):
    # LPF
    
    inputmatrix = zeropadding(inputdata, flt_arm, flt_arm, -1)
    outmatrix = np.ones(inputdata.shape)
    m, n = inputmatrix.shape
    outmatrix = np.ones(inputdata.shape)
        
    for row in range(flt_arm, m-flt_arm):
        for col in range(flt_arm, n-flt_arm):
            data_window = inputmatrix[row-flt_arm:row+flt_arm+1, col-flt_arm:col+flt_arm+1].flatten()            
            outmatrix[row-flt_arm, col-flt_arm] = sum(data_window)/data_window.size

    return outmatrix   

def histflt(inputdata, labelhat, img_size):
    
    bin_size = 28  
    # each row in input data is an axis
    # each col in input data is an example
    bin_counter_x = np.zeros( int(np.ceil(img_size[0]/float(bin_size)) ) )
    bin_counter_y = np.zeros( int(np.ceil(img_size[1]/float(bin_size)) ) )
    
    for i in range(inputdata.shape[1]):
        coor_x = inputdata[0,i]
        coor_y = inputdata[1,i]
        hist_idx_x = int(coor_x/bin_size)
        hist_idx_y = int(coor_y/bin_size)
        bin_counter_x[hist_idx_x] += 1
        bin_counter_y[hist_idx_y] += 1 
    
    if max(bin_counter_x) > max(bin_counter_y):
        main_vector_direction = 0
        main_vector_idx = bin_counter_x.argmax()
    else:
        main_vector_direction = 1
        main_vector_idx = bin_counter_y.argmax()
            
    # filter out    
    coor_kept = []
    label_kept = []
    for j in range(inputdata.shape[1]):

        if main_vector_direction == 0:
            coor_x = inputdata[0, j]
            if  bin_size*main_vector_idx <= coor_x <= bin_size*(main_vector_idx+1):
                coor_kept.append( np.array([inputdata[0,j], inputdata[1,j]]) )
                label_kept.append ( labelhat[j] )
            else:
                continue
                
        elif main_vector_direction == 1:
            coor_y = inputdata[1, j]
            if  bin_size*main_vector_idx <= coor_y <= bin_size*(main_vector_idx+1):
                coor_kept.append( np.array([inputdata[0,j], inputdata[1,j]]) )
                label_kept.append( labelhat[j] )
            else:
                continue 
    
    outputdata = np.zeros((2, len(coor_kept)))
    for k in range(len(coor_kept)):
        outputdata[:,k] = coor_kept[k]
                    
    return outputdata, label_kept, main_vector_direction
                
    
        
    















    
        