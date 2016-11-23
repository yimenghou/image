# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 10:15:52 2016

@author: yimeng
"""

import math, os, cv2, shutil
import numpy as np

def dp(InputCost):
    # dynamic programming 

    weight_smooth = 2
    edge = np.ones(InputCost.shape[1])
    paths = np.ones(InputCost.shape[0])  
    backtracing = np.ones(InputCost.shape)
    CAM = copy.deepcopy(InputCost) # cost accumulate matrix
    
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
   
def openning(inputmatrix, wind_size = (3,3)):
    # mathematical morphology: openning
    
    kernel = np.ones(wind_size, np.uint8)
    erosion = cv2.erode(inputmatrix, kernel)
    dilation = cv2.dilate(erosion, kernel)
    return dilation
    
def closing(inputmatrix, wind_size = (3,3)):
    # mathematical morphology: closing
    
    kernel = np.ones(wind_size, np.uint8)
    dilation = cv2.dilate(inputmatrix, kernel)
    erosion = cv2.erode(dilation, kernel)
    return erosion
    
def normalize_std(data):
    # standard normalize
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0) 
    data_std[data_std == 0] = 1   
    return (data - data_mean)/data_std, data_mean, data_std   
    
def relativeDistance(inputdata):
    temp_data = np.vstack((inputdata[0], inputdata[1]))

    dist = 0
    for i in range(len(inputdata[0])):
        for j in range(len(inputdata[0])):
            if i!= j:
                dist += np.sqrt((temp_data[0,i] - temp_data[0,j])**2 + (temp_data[1,i] - temp_data[1,j])**2)
                
    return dist
    
def zeropadding(matrix, padsize1, padsize2, constant_val = 0):
    leftPad,rightPad,topPad,bottomPad = padsize1, padsize1, padsize2, padsize2
    pads = ((leftPad,rightPad),(topPad,bottomPad))
    return np.pad(matrix, pads, 'constant', constant_values = constant_val)

def NonMaximumSuppression(labelmap,scoremap,suppresssize):
    finallabel = np.zeros((0,4))
    mapshape = np.array([scoremap.shape[0:2]])
    while (scoremap>0).any():
        index = scoremap.argmax()
        index = np.array([index/mapshape[0,1],index%mapshape[0,1]])
        singlelabel = np.array([[labelmap[index[0],index[1]],scoremap[index[0],index[1]],index[0],index[1]]])
        finallabel = np.concatenate((finallabel,singlelabel),0)
        suppressedLoc = np.array([0,0,mapshape[0,0],mapshape[0,1]])
        if index[0]-suppresssize>0:
            suppressedLoc[0] = index[0]-suppresssize
        if index[0]+suppresssize<suppressedLoc[2]:
            suppressedLoc[2] = index[0]+suppresssize
        if index[1]-suppresssize>0:
            suppressedLoc[1] = index[1]-suppresssize
        if index[1]+suppresssize<suppressedLoc[3]:
            suppressedLoc[3] = index[1]+suppresssize
        scoremap[suppressedLoc[0]:suppressedLoc[2],suppressedLoc[1]:suppressedLoc[3]]=0
    
    return finallabel

def cooc(inputmtx, angle, greylevel=16, norm_flag=1):
    # calculate co-occurence matrix

    a,b = inputmtx.shape
    inputmatrix = np.zeros(inputmtx.shape)
    cocurrentmtx = np.zeros((256/greylevel, 256/greylevel))
    
    for x in range(inputmtx.shape[0]):
        for y in range(inputmtx.shape[1]):            
            inputmatrix[x,y] = round(inputmtx[x,y]/greylevel )   

    if angle == 0:
        for i in range(a):
            for j in range(b-1):
                leftitem = inputmatrix[i,j]
                rightitem = inputmatrix[i,j+1]
                cocurrentmtx[leftitem-1,rightitem-1] += 1
                cocurrentmtx[rightitem-1,leftitem-1] += 1
    elif angle == 45:
        for i in range(a-1):
            for j in range(1,b):
                upperrightitem = inputmatrix[i,j]
                lowerleftitem = inputmatrix[i+1,j-1]
                cocurrentmtx[upperrightitem-1,lowerleftitem-1] += 1
                cocurrentmtx[lowerleftitem-1,upperrightitem-1] += 1
    elif angle == 90:
        for i in range(a-1):
            for j in range(b):
                upperitem = inputmatrix[i,j]
                loweritem = inputmatrix[i+1,j]
                cocurrentmtx[upperitem-1,loweritem-1] += 1
                cocurrentmtx[loweritem-1,upperitem-1] += 1 
    elif angle == 135:
        for i in range(a-1):
            for j in range(b-1):
                upperleftitem = inputmatrix[i,j]
                lowerrightitem = inputmatrix[i+1,j+1]
                cocurrentmtx[upperleftitem-1,lowerrightitem-1] += 1
                cocurrentmtx[lowerrightitem-1,upperleftitem-1] += 1  

    if norm_flag == 1:
        return cocurrentmtx/float(sum(sum(cocurrentmtx)))
    else:
        return cocurrentmtx

def featureDescriptor(coocmtx):
    # feature descriptors: asm, con, ent, corr

    a,b = coocmtx.shape
    mu_a = coocmtx.mean(1)
    mu_b = coocmtx.mean(0)
    sigma_a = coocmtx.std(1)
    sigma_b = coocmtx.std(0)    
    con_val, ent_val, corr_val, home_val = 0,0,0,0
    
    asm_val = sum(sum(coocmtx*coocmtx))
    for i in range(a):
        for j in range(b):
            con_val += coocmtx[i,j]*((i-j)**2)            
            if coocmtx[i,j] !=0:
                ent_val += coocmtx[i,j]*np.log2(coocmtx[i,j])
            corr_val += i*j*coocmtx[i,j]
            home_val += coocmtx[i,j]/float(1+abs(i-j))
       
    corr_val = (corr_val-np.dot(mu_a,mu_b))/np.dot(sigma_a,sigma_b)
    ent_val = -ent_val
    
    return asm_val, con_val, ent_val, corr_val, home_val

def twos_comp(val, bits):
    # compute the 2's compliment of int value val

    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)    
        # compute negative value
    return val                         # return positive value as is

def converter(string_list):
    # convert bytes from FPGA into int
    
    string_list = string_list[::-1]
    number_list = []        
    buf = ""
    
    cout = 0
    for i in string_list:

        current_buf = bin( int(i, 16) )[2:].zfill(8)
        buf += current_buf        
        if (cout+1)%4 == 0:

            buf = twos_comp(int(buf,2), 32)
            number_list.append( buf )
            buf = ""
               
        cout += 1
            
    return number_list[::-1]

# def normalize_fpga(input_data):
#     # converted float into uint8     
    
#     data_max = input_data.max()
#     output_data = np.uint8(input_data/data_max*255)
#     return output_data

def normalize_fpga0(input_data):

    a,b = 0,255
    input_data_max = input_data.max()
    input_data_min = input_data.min()

    output_data = (b-a)*(input_data-input_data_min)/(input_data_max-input_data_min)

    return np.uint8(output_data)

def makePatch(input_canvas, window_sz, stride, resize_fact, save_flg = True):
    # decomposite large canvas into several image patchs
    # the function returns a 5d data matrix
    
    # baseSavePath = os.path.join('C:\Users\westwell\Desktop\harbourProject\img', str(window_sz[0]))
    baseSavePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), str(window_sz[0]))

    if save_flg: 
        try:
            shutil.rmtree(baseSavePath)
        except:
            pass

        os.mkdir(baseSavePath)

    canvas_ds = cv2.resize( input_canvas, (0,0), fx=resize_fact[0], fy=resize_fact[1] )
    canvas_sz = canvas_ds.shape

    # indexing
    idh = [i for i in range(0, canvas_sz[0]-window_sz[0]+1, stride[0])]
    idv = [j for j in range(0, canvas_sz[1]-window_sz[1]+1, stride[1])]

    # slicing
    image_5d = np.zeros( (window_sz[0], window_sz[1], 3, len(idh), len(idv) ), dtype='uint8' )

    if save_flg:

        for i in enumerate(idh):
            for j in enumerate(idv):

                temp_img = canvas_ds[i[1]:i[1] + window_sz[0], j[1]:j[1] + window_sz[1], :]
                image_5d[:,:,:,i[0],j[0]] = temp_img

                img_name = os.path.join(baseSavePath,str(i[0])+'-'+str(j[0])+".bmp")
                cv2.imwrite(img_name, temp_img)
    else:

        for i in enumerate(idh):
            for j in enumerate(idv):
                temp_img = canvas_ds[i[1]:i[1] + window_sz[0], j[1]:j[1] + window_sz[1], :]
                image_5d[:,:,:,i[0],j[0]] = temp_img

    return image_5d

def dimension_pad(inputmatrix):
    # pad current inputmatrix into a multiple integer of 128

    pad_len = 128-inputmatrix.shape[1]%128

    if pad_len == 128:
        return inputmatrix
    else:
        outputmatrix = np.hstack(( inputmatrix, np.zeros((inputmatrix.shape[0], pad_len), dtype=int) ))
        return outputmatrix

