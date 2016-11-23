import numpy as np
import cv2, time, os
from prep import *

from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2grey
from skimage.exposure import histogram

# method for loading testing dataset
class getTestingImage(object):
    
    def __init__(self, config_table, extern_base_path = []):       
 
        self.config = config_table
        self.resize_scale = self.config['dataset']['resize']

        if extern_base_path == []:
            self.basePath = self.config["path_testCanvas"]     
        else:
            self.basePath = extern_base_path   
        
        if not os.path.exists(self.basePath):
            raise IOError("Path: %s to specified dataset does not exist"%(self.basePath))

        self.dataset = self.IMGfileHandler() 
                        
    def IMGfileHandler(self):
        
        print ">>> Loading/preprocessing testing Dataset in progress .."  
        print "Path: %s"%self.basePath
        
        canvas = imread(self.basePath)
        # the testing canvas is always been donwsampled.
        self.canvas_ds = cv2.resize(canvas,(0,0), fx=self.resize_scale[0], fy=self.resize_scale[1])
        # canvas_gray = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
        
        self.boundSize = self.config["dataset"]["boundSize"]
        self.boundStride = self.config["dataset"]["boundStride"]

        # this function decomposite the big canvas into several small patches        
        patchMatrix = makePatch( canvas, self.boundSize, self.boundStride, resize_fact = self.resize_scale )     
        processedPatchMatrix = np.zeros((patchMatrix.shape[3], patchMatrix.shape[4], self.config['init']['buffer_size'] ), dtype=float)

        # convert a 5d data matrix into 3d matrix, apply preprocessing methods to every patch of canvas
        _sample_count = 1
        for i in range(patchMatrix.shape[3]):
            for j in range(patchMatrix.shape[4]):

                if _sample_count%100 == 0:
                    print ""
                    print "> Current processing index:", _sample_count, "totally:", patchMatrix.shape[3]*patchMatrix.shape[4]
  
                # to try new preprocessing methods, just replace the below function with others
                processedPatchMatrix[i,j,:] = self.fourierFeature_handler( patchMatrix[:,:,:,i,j] )
                _sample_count += 1
                print ".",
        print ""
        return processedPatchMatrix

    def color_hist_handler(self, img):
        # applied to every patch

        grey_img = rgb2grey(img)
        hsv_img = rgb2hsv(img)

        #color feature
        rgb_val = [img[:,:,i].mean() for i in range(3)] 
        hsv_val = [hsv_img[:,:,i].mean() for i in range(3)]

        #greyscale histogram feature
        hist = histogram(grey_img, 2) # 122
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

        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fft_img = np.fft.fft2(grey_img)
        featureBag = np.abs(fft_img)
        # featureBag = np.hstack((np.real(fft_img), np.imag(fft_img)))

        return featureBag.flatten()
