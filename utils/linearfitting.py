# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:22:52 2016

@author: ThinkPad
"""
from scipy.optimize import curve_fit

def fit_func(x, a, b):
    return a*x + b

def linearfitting(centerSET):
    # centerSET is an 2-by-N 2d array
    
    # curve fitting        
    params = curve_fit(fit_func, centerSET[0,:], centerSET[1,:])
    [slope, bias] = params[0]

    return slope, bias        