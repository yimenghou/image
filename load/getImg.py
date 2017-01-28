
import numpy as np
import cv2
import os
import cPickle
from mnist import mnist


class getImg(object):
    
    def __init__(self, config_table):
        
        self.config = config_table
        self._basePath = self.config["path"]        
        self.datasetType = self.config["type"]

        self.hog_instance = cv2.HOGDescriptor((28,28), (8,8), (4,4), (4,4), 9)
        
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

    def MNISTHandler(self):       
        print ">>> Loading MNIST in progress ..."
        
        if self.config["hog"]:
            self.dataset = np.zeros((70000, 1296))
        else:
            self.dataset = np.zeros((70000, 784))
        self.labelset = np.zeros((70000, 10))        
        
        MNISTtr = mnist("train")
        MNISTte = mnist("test")
        
        for i in range(70000):
            
            if i < 60000:
                _label, _img = MNISTtr.GetImage(i)

            else:
                _label, _img = MNISTte.GetImage(i-60000)
            
            _prep_img = self.IMG_prep_handler(_img)
            
            self.dataset[i,:] = _prep_img.flatten()
            label_temp = np.zeros(10)
            label_temp[_label] = 1
            self.labelset[i,:] = label_temp        

    def CIFAR10Handler(self):
        print ">>> Loading CIFAR-10 in progress ..."      
 
        self.dataset = np.zeros((60000, 3072))
        self.labelset = np.zeros((60000, 10))  
        
        
        path1 = "E:\\dataset\\cifar10\\cifar-10-batches-py\\data_batch_1"
        path2 = "E:\\dataset\\cifar10\\cifar-10-batches-py\\data_batch_2"
        path3 = "E:\\dataset\\cifar10\\cifar-10-batches-py\\data_batch_3"
        path4 = "E:\\dataset\\cifar10\\cifar-10-batches-py\\data_batch_4"
        path5 = "E:\\dataset\\cifar10\\cifar-10-batches-py\\data_batch_5"
        path6 = "E:\\dataset\\cifar10\\cifar-10-batches-py\\test_batch"
        
        try:
            data_dict1 = self.unpickle(path1)
            data_dict2 = self.unpickle(path2)
            data_dict3 = self.unpickle(path3)
            data_dict4 = self.unpickle(path4)
            data_dict5 = self.unpickle(path5)
            data_dict6 = self.unpickle(path6)
        except IOError:
            print "Path: %s to CIFAR data does not exist"%path1[:-12]
        
        self.dataset = np.vstack((data_dict1['data'],data_dict2['data'],\
                                  data_dict3['data'],data_dict4['data'],\
                                  data_dict5['data'],data_dict6['data'] ))
                                  
        labelset_list = data_dict1['labels']+data_dict2['labels']+ \
                        data_dict3['labels']+data_dict4['labels']+ \
                        data_dict5['labels']+data_dict6['labels']
                        
        for i in range(len(labelset_list)):
            label_temp = np.zeros(10)
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
                
                if self.config["type"] == "MNIST" or self.config["type"] == "CONTAINER":
                    if _img.size% 784 != 0:
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
        print labelbase.shape
        for j in range(len(database0)):       
            labelbase[j, labelbase0[j]] = 1

        data_con = np.hstack((database1, labelbase))
        
        np.random.shuffle(data_con)
        self.dataset = data_con[:, :database1.shape[1]]
        self.labelset = data_con[:, database1.shape[1]:]     
        
        self.numSum = self.dataset.shape[0]        
        shuffle_idx = np.random.permutation(self.numSum)
        self.dataset = self.dataset[shuffle_idx]
        self.labelset = self.labelset[shuffle_idx]
        
    def IMG_prep_handler(self, _prep_img):

        if self.config["type"] == 'MNIST' or self.config["type"] == 'CONTAINTER':
            feature_arm = 28
        else:
            feature_arm = 32
        _prep_img = np.uint8(_prep_img)
        
        if self.config["greyscale"]:
            _prep_img = cv2.cvtColor( _prep_img, cv2.COLOR_RGB2GRAY )
            
            if self.config["threshold"]:
                threshold_temp = cv2.threshold(_prep_img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                _prep_img = threshold_temp[1].flatten()
                
        elif self.config["rescale"]:
            _prep_img = cv2.resize( _prep_img, (0,0), fx=self.config["factor"], fy=self.config["factor"])
        elif self.config["crop"]:
            _crop_rg = self.config["region"]
            _prep_img = _prep_img[int(feature_arm*_crop_rg[2]):int(feature_arm*_crop_rg[3]), int(feature_arm*_crop_rg[0]):int(feature_arm*_crop_rg[1])]
        else:
            _prep_img = _prep_img
            
        if self.config["hog"]:
            if len(_prep_img.shape) == 1:
                _prep_img_out = self.hog_instance.compute( np.reshape(_prep_img, (feature_arm, feature_arm)) )
            else:
                _prep_img_out = self.hog_instance.compute( _prep_img )
        else:
            _prep_img_out = _prep_img
        
        return _prep_img_out
        
    def unpickle(self, file_path):
    
        with open(file_path, 'rb') as fhandle:
            dict = cPickle.load(fhandle)
    
        return dict
        
    @property
    def content(self):
        return self.dataset
    @property
    def label(self):
        return self.labelset
    @property
    def name(self):
        return self.datasetType
    

def twos_comp(val, bits):
    """compute the 2's compliment of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is

def converter(string_list):
    
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
    data_max = input_data.max()
    output_data = np.uint8(input_data/data_max*255)
    return output_data




    