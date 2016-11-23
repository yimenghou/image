import numpy as np
import cv2
import os, time

from utils.prep import *
from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2grey
from skimage.exposure import histogram

# method for loading training datasets, used for validation and training
class getTrainingImage(object):
    
    def __init__(self, config_table):
        
        self.config = config_table
        self.basePath = self.config["dataset"]["path_trainImage"]   
        self.baseSavePath = os.path.dirname(os.path.realpath(__file__))   
        self.resize_scale = self.config["dataset"]["resize"]  
        
        if not os.path.exists(self.basePath):
            raise IOError("Path: %s to specified dataset does not exist"%(self.basePath))

        # make a new directory for saving training features
        self.featurePath = os.path.join(self.baseSavePath, "Feature")
        try:
            os.mkdir(self.featurePath)
        except:
            pass

        # if we already have training features, load it directly; if not, preprocessing the training images
        if os.listdir( self.featurePath ) == []:
            self.IMGfileHandler()  
        else:
            try:
                print ">>> Load dataset from existing feature files"
                self.dataset =  np.loadtxt( os.path.join(self.featurePath, "training_dataset"), dtype='uint8')
                self.labelset = np.loadtxt( os.path.join(self.featurePath, "training_labelset"), dtype='int')
            #     self.mean_vector = np.loadtxt( os.path.join(self.featurePath, "mean"))
            #     self.var_vector = np.loadtxt( os.path.join(self.featurePath, "var"))
            except:
                print ">>> Load existing feature files failed, start loading dataset from image files"
                self.IMGfileHandler()               
                        
    def IMGfileHandler(self):
        # this method loads datasets from basepath and shuffle them
            
        self._classNum = len(os.listdir(self.basePath))
        _itemlist = os.listdir(self.basePath)
        
        print ">>> Loading and Preprocessing Training Dataset from Path: %s"%self.basePath
        
        database0 = []
        labelbase0 = []
        _sample_count = 0
        _tot_n_examples = 0

        # calculate the number of total examples
        for item in _itemlist:
            _tot_n_examples += len(os.listdir( os.path.join(self.basePath, item) ))
        
        # start loading examples
        for i in _itemlist:
                
            _classPath = os.path.join(self.basePath, str(i))

            for sample in enumerate(os.listdir(_classPath)): 
         
                if _sample_count%100 == 0:
                    print ""
                    print "> Current loading index: %d of %d from class %d, totally %d of %d"\
                            %(sample[0], len(os.listdir(_classPath)), int(i), _sample_count, _tot_n_examples)

                # _img = imread(os.path.join(_classPath, sample[1]))
                _img = cv2.imread(os.path.join(_classPath, sample[1]), 0)
                # _img_processed = self.color_hist_handler( _img )
                _img_processed = self.fourierFeature_handler( _img )
                database0.append( _img_processed.flatten() )  
                labelbase0.append( int(i) )
                _sample_count += 1
                print ".",   

        raw_dataset = np.zeros((len(database0), _img_processed.size), dtype=float)
        for i in range(len(database0)):
            raw_dataset[i,:] = database0[i]
            
        labelbase0 = np.array(labelbase0)
        self.labelset = np.zeros((len(database0), self._classNum), dtype = np.int)

        for j in range(len(database0)):       
            self.labelset[j, labelbase0[j]] = 1  
        
        # shuffle examples with a fixed seed
        numSum = raw_dataset.shape[0]       
        np.random.seed(0) 
        shuffle_idx = np.random.permutation(numSum)
        raw_dataset = raw_dataset[shuffle_idx,:]
        self.labelset = self.labelset[shuffle_idx,:]

        # norm_temp = normalize_std(raw_dataset) # standard normalize
        # self.mean_vector = norm_temp[1]
        # self.var_vector = norm_temp[2]
        self.dataset = normalize_fpga0(raw_dataset) # fpga normalize

        print ""
        print "Saving datasets and labelsets .."
        # save the features for future use
        np.savetxt(os.path.join(self.featurePath, 'training_dataset'), self.dataset)
        np.savetxt(os.path.join(self.featurePath, 'training_labelset'), self.labelset) 
        # np.savetxt(os.path.join(self.featurePath, 'mean'), self.mean_vector)
        # np.savetxt(os.path.join(self.featurePath, 'var'), self.var_vector)        

    def color_hist_handler(self, img):
        # applied to every patch

        img = cv2.resize(img, (0,0), fx = self.resize_scale[0], fy = self.resize_scale[1])
        grey_img = rgb2grey(img)
        hsv_img = rgb2hsv(img)

        # color feature
        rgb_val = [img[:,:,i].mean() for i in range(3)] 
        hsv_val = [hsv_img[:,:,i].mean() for i in range(3)]

        #greyscale histogram feature
        hist = histogram(grey_img, 2) # 16
        hist_val = hist[0]

        # GLCM feature
        prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'correlation']

        descriptor_val = np.zeros(120)
        for i in enumerate(prop_list):
            glcm = greycomatrix(np.uint8(grey_img*255), [1,2,3,4,5,6], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
            temp = greycoprops(glcm, prop=i[1])

            descriptor_val[i[0]*24:(i[0]+1)*24] = np.hstack((temp))

        # concatenate above features
        featureBag = np.hstack((np.array(rgb_val), np.array(hsv_val), hist_val, descriptor_val) )
        # featureBag = np.hstack((np.asarray(rgb_val), np.asarray(hsv_val), hist_val))

        return featureBag

    def fourierFeature_handler(self, img):
        # applied to every patch

        img = cv2.resize(img, (0,0), fx = self.resize_scale[0], fy = self.resize_scale[1])
        fft_img = np.fft.fft2(img)
        featureBag = np.abs(fft_img)
        # featureBag = np.hstack((np.real(fft_img), np.imag(fft_img)))

        return featureBag