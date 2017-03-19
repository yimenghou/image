# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 14:48:22 2016

@author: ThinkPad
"""
from numpy import eye, zeros, hstack

# calculate the (hamming) code distance between observed outputs and target outputs

def distCal(inputArray):
    
    sz = inputArray.size
    table = hstack(( eye(sz), zeros((sz, 1)) ))
    ans = zeros(sz+1)
    
    for dist in range(sz+1):        
        sqrt_temp = map(lambda p,q: (p-q)**2, table[:,dist] , inputArray)
        ans[dist] = reduce(lambda p,q: p+q, sqrt_temp)
    
    if ans.argmin() == sz or ans.argmin() == sz-1:
        return -1, min(ans)
    else:
        return ans.argmin(), min(ans)
        
def distCal2(inputArray):
    
    sz = inputArray.size
    table = hstack(( eye(sz), zeros((sz, 1)) ))
    ans = zeros(sz+1)
    
    for dist in range(sz+1):        
        sqrt_temp = map(lambda p,q: (p-q)**2, table[:,dist] , inputArray)
        ans[dist] = reduce(lambda p,q: p+q, sqrt_temp)
    
    if ans.argmin() == sz or ans.argmin() == sz-1:
        return -1
    else:
        return ans.argmin()
        
def distCal_bin(inputArray):
    
    sz = inputArray.size
    table = hstack(( eye(sz), zeros((sz, 1)) ))
    ans = zeros(sz+1)
    
    for dist in range(sz+1):        
        sqrt_temp = map(lambda p,q: (p-q)**2, table[:,dist] , inputArray)
        ans[dist] = reduce(lambda p,q: p+q, sqrt_temp)
    
    if ans.argmin() == 2 or ans.argmin() == 0:
        return -1
    else:
        return ans.argmin()

def distCal_normal(inputArray):
    
    sz = inputArray.size
    table = eye(sz)
    ans = zeros(sz)
    
    for dist in range(sz):        
        sqrt_temp = map(lambda p,q: (p-q)**2, table[:,dist] , inputArray)
        ans[dist] = reduce(lambda p,q: p+q, sqrt_temp)
    
    return ans.argmin()
        
        
    