# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:35:23 2016

@author: yimeng
"""

import numpy as np
import cv2, os, cPickle
from mnist import mnist


class getImg(object):
    
    def __init__(self, config_table):
        
        self.config = config_table
        self._basePath = self.config["path"]        
        self.datasetType = self.config["type"]
        
        if not os.path.exists(self._basePath):
            raise IOError("Path: %s to specified dataset does not exist"%(self._basePath))
                    
        if self.datasetType == "MNIST":
            self.MNISTHandler()
        elif self.datasetType == 'CONTAINER':
            self.IMGfileHandler()
        elif self.datasetType == "CIFAR":
            self.CIFAR10Handler()
        else:
            raise Exception("Unexpected datasets")

    def databaseHandler(self):  
        # handle other datasets
        raise NotImplementedError

    def MNISTHandler(self):  
        # MNIST dataset has 70,000 examples in total
        print ">>> Loading MNIST in progress ..."
        
        self.dataset = np.zeros((70000, 784))
        self.labelset = np.zeros((70000, 10), dtype=np.int)        
        
        MNISTtr = mnist("train")
        MNISTte = mnist("test")
        
        for i in range(70000):
            
            if i%10000 == 0:
                print "> current loading index: "+str(i)                         
            
            if i < 60000:
                _label, _img = MNISTtr.GetImage(i)

            else:
                _label, _img = MNISTte.GetImage(i-60000)
            
            _prep_img = self.IMG_prep_handler(_img)
            
            self.dataset[i,:] = _prep_img.flatten()
            label_temp = np.zeros(10)
            label_temp[_label] = 1
            self.labelset[i,:] = label_temp        

    def CIFAR10Handler(self, basepath):
        # cifar-10 has 60000 examples in total
        print ">>> Loading CIFAR-10 in progress ..."      
        
        if self.config['hog']:
            self.dataset = np.zeros((60000, 7056))
        else:
            self.dataset = np.zeros((60000, 3072))
            
        self.labelset = np.zeros((60000, 10), dtype=np.int)  

        path1 = "data_batch_1"
        path2 = "data_batch_2"
        path3 = "data_batch_3"
        path4 = "data_batch_4"
        path5 = "data_batch_5"
        path6 = "test_batch"
        
        try:
            data_dict1 = self.unpickle(os.path.join(basepath, path1))
            data_dict2 = self.unpickle(os.path.join(basepath, path2))
            data_dict3 = self.unpickle(os.path.join(basepath, path3))
            data_dict4 = self.unpickle(os.path.join(basepath, path4))
            data_dict5 = self.unpickle(os.path.join(basepath, path5))
            data_dict6 = self.unpickle(os.path.join(basepath, path6))
        except IOError:
            print "Path: %s to CIFAR data does not exist"%path1[:-12]
        
        dataset_temp = np.vstack((data_dict1['data'],data_dict2['data'],\
                                  data_dict3['data'],data_dict4['data'],\
                                  data_dict5['data'],data_dict6['data'] ))
                                  
        labelset_list = data_dict1['labels']+data_dict2['labels']+ \
                        data_dict3['labels']+data_dict4['labels']+ \
                        data_dict5['labels']+data_dict6['labels']
                        
        for i in range(len(labelset_list)):
            
            if i%10000 == 0:
                print "> current loading index: "+str(i)                         
            
            _img = dataset_temp[i,:]
            
            _prep_img = self.IMG_prep_handler(_img)
            
            self.dataset[i,:] = _prep_img.flatten()            
                        
            label_temp = np.zeros(10, dtype=np.int)
            label_temp[ labelset_list[i] ] = 1
            self.labelset[i,:] = label_temp
                       
    def IMGfileHandler(self):
            
        self._classNum = len(os.listdir(self._basePath))
        _itemlist = os.listdir(self._basePath)
        
        print ">>> Loading IMG Dataset in progress ..."  
        
        database0 = []
        labelbase0 = []
        _sample_count = 0
        
        for i in _itemlist:
            
            try:
                convert_folder_name = int(i)
                print "Now loading examples from Class",str(convert_folder_name)
                
            except ValueError:
                print "Expect folder name to be numbers instead of characters"
                
            _classPath = self._basePath + "\\" + str(i) + '\\'

            for sample in os.listdir(_classPath): 
                _sample_count += 1
                               
                if _sample_count%1000 == 0:
                    print "> current loading index: "+str(_sample_count) 
                
                try:
                    _img = cv2.imread(_classPath+sample)
                except:
                    continue

                _img_processed = self.IMG_prep_handler(_img) 
                
                database0.append( _img_processed.flatten() )
                labelbase0.append( int(i)%10 )

        print ">>> Load complete, Number of total train/test item: "+str(len(database0))  
        print ""
        
        database1 = np.zeros((len(database0), _img_processed.size))
        for i in range(len(database0)):
            database1[i,:] = database0[i]
            
        labelbase0 = np.array(labelbase0)
        labelbase = np.zeros((len(database0), self._classNum))

        for j in range(len(database0)):       
            labelbase[j, labelbase0[j]] = 1

        data_con = np.hstack((database1, labelbase))
        
        # shuffle dataset
        np.random.shuffle(data_con)
        self.dataset = data_con[:, :database1.shape[1]]
        self.labelset = data_con[:, database1.shape[1]:]     
        
        # shuffle dataset
        numSum = self.dataset.shape[0]        
        shuffle_idx = np.random.permutation(numSum)
        self.dataset = self.dataset[shuffle_idx]
        self.labelset = self.labelset[shuffle_idx]
        
    def IMG_prep_handler(self, _prep_img):
        # apply preprocessing to each example
        img_height, img_width, _ = _prep_img.shape
        _prep_img = np.uint8(_prep_img)
        
        if self.config["greyscale"]:
            _prep_img = cv2.cvtColor( _prep_img, cv2.COLOR_RGB2GRAY )
        elif self.config["rescale"]:
            _prep_img = cv2.resize( _prep_img, (0,0), fx=self.config["factor"], fy=self.config["factor"])
        elif self.config["crop"]:
            _crop_rg = self.config["region"]
            _prep_img = _prep_img[int(img_height*_crop_rg[2]):int(img_height*_crop_rg[3]), int(img_width*_crop_rg[0]):int(img_width*_crop_rg[1])]
        elif self.config["threshold"]:
            threshold_temp = cv2.threshold(_prep_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _prep_img = threshold_temp[1].flatten()
        else:
            _prep_img = _prep_img
        
        return _prep_img_out
        
    def unpickle(self, file_path):   
        with open(file_path, 'rb') as fhandle:
            dict = cPickle.load(fhandle)    
        return dict
        
    @property
    def content(self):
        if self.config['hog']:
            return normalizer(self.dataset)
        else:
            return self.dataset
    @property
    def label(self):
        return self.labelset
    @property
    def name(self):
        return self.datasetType
    
def twos_comp(val, bits):
    '''
    compute the 2's compliment of int value val
    '''
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)    
        # compute negative value
    return val                         # return positive value as is

def converter(string_list):
    '''
    convert bytes from FPGA into int
    '''
    
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

def normalizer(input_data):
    '''
    this normalizer is mainly used for converted float into uint8     
    '''
    
    data_max = input_data.max()
    output_data = np.uint8(input_data/data_max*255)
    return output_data