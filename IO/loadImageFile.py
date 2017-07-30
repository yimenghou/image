# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:35:23 2016

@author: yimeng
"""

import numpy as np
import cv2, os

class getImg(object):
    
    def __init__(self, config_table):
        
        self.config = config_table
        self._basePath = self.config["path"]        
        
        if not os.path.exists(self._basePath):
            raise IOError("Path: %s to specified dataset does not exist"%(self._basePath))
        
        self.IMGfileHandler() 

        if self.config['hog']:
            self.dataset = self.hog_computer()     
                        
    def IMGfileHandler(self):
        # method to load image file
            
        self._classNum = len(os.listdir(self._basePath))
        _itemlist = os.listdir(self._basePath)
        
        print ">>> Loading harbour Dataset in progress ..."  
        
        database0 = []
        labelbase0 = []
        _sample_count = 0
        
        for i in _itemlist:
            
            print "Now loading examples from Class %s"%i
                
            _classPath = os.path.join(self._basePath, str(i))

            for sample in os.listdir(_classPath): 
                _sample_count += 1
                               
                if _sample_count%1000 == 0:
                    print "> current loading index: "+str(_sample_count) 
            
                _img = cv2.imread(os.path.join(_classPath, sample))   

                _img_processed = self.IMG_prep_handler(_img)                
                database0.append( _img_processed.flatten() )
                labelbase0.append( int(i) )

        print ">>> Load complete, Number of total train/test item: "+str(len(database0))  
        
        self.dataset = np.zeros((len(database0), _img_processed.size), dtype = np.uint8)
        for i in range(len(database0)):
            self.dataset[i,:] = database0[i]
            
        labelbase0 = np.array(labelbase0)
        self.labelset = np.zeros((len(database0), self._classNum), dtype = np.int)

        for j in range(len(database0)):       
            self.labelset[j, labelbase0[j]] = 1  
        
        # shuffle dataset
        numSum = self.dataset.shape[0]        
        shuffle_idx = np.random.permutation(numSum)
        self.dataset = self.dataset[shuffle_idx,:]
        self.labelset = self.labelset[shuffle_idx,:]
        
    def IMG_prep_handler(self, _prep_img):
        # apply preprocessing to each example
        
        _prep_img = np.uint8(_prep_img)
        _prep_img_sz = _prep_img.shape
        
        if self.config["greyscale"]:
            _prep_img = cv2.cvtColor( _prep_img, cv2.COLOR_RGB2GRAY )

        if self.config["rescale"]:
            _prep_img = cv2.resize( _prep_img, (0,0), fx=self.config["factor"], fy=self.config["factor"])

        if self.config["crop"]:
            _crop_rg = self.config["region"]
            _prep_img = _prep_img[int(_prep_img_sz[0]*_crop_rg[2]):int(_prep_img_sz[0]*_crop_rg[3]), \
                                  int(_prep_img_sz[1]*_crop_rg[0]):int(_prep_img_sz[1]*_crop_rg[1])]

        if self.config["threshold"]:
            threshold_temp = cv2.threshold(_prep_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _prep_img = threshold_temp[1].flatten()
        
        return _prep_img

    # def hog_computer(self):
    #     # Note: this part only support data with feature dimension of 28*28 = 784
        
    #     N_num = self.dataset.shape[0]       
    #     hog_dataset = np.zeros((N_num, 1296))          
        
    #     self.dataset = np.uint8(self.dataset)
    #     self.hog_instance = cv2.HOGDescriptor((28,28), (8,8), (4,4), (4,4), 9)
    #     for i in range(N_num):
    #         hog_dataset[i,:] = self.hog_instance.compute( np.reshape(self.dataset[i,:],(28,28) )).flatten()
        
    #     return hog_dataset          
        