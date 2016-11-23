
"""
Performs the ELM convolution
"""

import numpy as np
from flt import histflt2
from kmean import kmean

def find(lst, tar):
    return [i for i, x in enumerate(lst) if x == tar]

class Clusterer(object):
    """ Contains all the Image pre-processing code """

    def __init__(self, config):
        """ Creates a new instance of the preprocessor """

        self.boundry_thresh = config['boundary_thresh']
        self.Outsize = config['outsize']

    def run(self, labelmatrix, confidence):
        
        """ Run the clustering algorithm """
        clust = np.where(labelmatrix != -1)

        if len(clust[0]) == 0:
            return "No numbers detected..."
        else:
            data_coor = np.vstack((clust[0], clust[1]))  
            
            #data_coor, main_vec_idx = histflt(data_coor, labelmatrix.shape)
            #data_coor = histflt2(data_coor, labelmatrix)                    
            #print "center Axis = ", main_vec_idx
            
            print ""            
            center, numDigits = self.findCentroids(data_coor)
            labelhat_confidence = self.confidenceMethod(data_coor, labelmatrix, confidence, center, numDigits)
            
            print "Detected strings by confidence:", labelhat_confidence
            labelhat_weighting = self.weightingMethod(data_coor, labelmatrix, center, numDigits)
            print "Detected strings by weighting :", labelhat_weighting
            
            if len(clust[0]) < 7:
                labelhat_kmean = []
                print "Detected strings by kmean     :", labelhat_kmean 
            else:
                labelhat_kmean = self.kmeanMethod(data_coor, labelmatrix, numDigits = 7) 
                print "Detected strings by kmean     :", labelhat_kmean       
            
        
    def confidenceMethod(self, data_coor, labelmatrix, confidence, center, numDigits):
        
        labelhat = np.zeros(numDigits)
        centroids = np.zeros((2, numDigits))        
        n_cout = 0
        
        for i in set(center):
            
            clusters_idx = find(center, i)
                                            
            k_val = []

            for k in clusters_idx:
                k_val.append( confidence[data_coor[0, k], data_coor[1, k]] )
            
            k_val = np.array(k_val)
            
            centroids[0, n_cout] = data_coor[0, clusters_idx[k_val.argmin()]]
            centroids[1, n_cout] = data_coor[1, clusters_idx[k_val.argmin()] ]
            
            labelhat[n_cout] = labelmatrix[centroids[0, n_cout], centroids[1, n_cout]]                 
        
            n_cout += 1
                            
        centroidindex = np.argsort(centroids[0,:])
        labelhat_sort = np.array(labelhat[centroidindex], dtype='int')
        
        '''     
        center_cout = 1
        for j in centroidindex:

            print "Prediction", center_cout ,", Coordinates=", centroids[:,j], "Label ->", int(labelhat[j])
            center_cout += 1
        '''     
        
        return labelhat_sort 
            
    def weightingMethod(self, data_coor, labelmatrix, center, numDigits):
        
        labelhat = np.zeros(numDigits)
        centroids = np.zeros((2, numDigits))        
        n_cout = 0
        
        for i in set(center):
            
            clusters_idx = find(center, i)
                            
            centroid_x, centroid_y = 0,0
            
            for j in clusters_idx:
                centroid_x += data_coor[0,j]
                centroid_y += data_coor[1,j]
                
            centroids[:, n_cout] = np.array([centroid_x/len(clusters_idx), centroid_y/len(clusters_idx)])
            countnum = np.zeros(self.Outsize)

            for k in clusters_idx:
                distfactor = np.absolute(data_coor[0, k]-centroids[0, i-1]) + np.absolute(data_coor[1, k]-centroids[1, i-1]) + 0.01 # euclidean distance                    
                countnum[labelmatrix[data_coor[0, k], data_coor[1, k]]] += 1/distfactor # equalization #1
                    
            labelhat[n_cout] = countnum.argmax()
            
            n_cout += 1
                                        
        centroidindex = np.argsort(centroids[0,:])
        labelhat_sort = np.array(labelhat[centroidindex], dtype='int')
        
        '''     
        center_cout = 1        
        for j in centroidindex:

            print "Prediction", center_cout ,", Coordinates=", centroids[:,j], "Label ->", int(labelhat[j])
            center_cout += 1
        '''     
        
        return labelhat_sort  
            
    def kmeanMethod(self, data_coor, labelmatrix, numDigits):
        
        if data_coor == []:
            return np.array([])
            
        Con, Iter = True, 0     
        
        while Con and Iter < (numDigits * 5):
            
            clusters, centroids = kmean(data_coor.T, numDigits)
            Con = False
            
            for i in range(len(centroids)):
                for j in range(len(centroids)):
                    
                    # bound detection 
                    if i != j and np.absolute(centroids[i][0] - centroids[j][0]) <= self.boundry_thresh*2 :
                        Con = True

            Iter += 1
        if Iter < numDigits * 5:        
            print "Clustering Succeed!"
        else:
            print "Clustering failed, latest entry picked"
        
        centroids = np.array(centroids)
        labelhat = np.zeros(numDigits)
        countnum = np.zeros((self.Outsize, numDigits))

        # decicsion making   
        if clusters != []:
            for i in range(len(clusters)):
                for j in range(len(clusters[i])):
                    temp = clusters[i][j]
                    distfactor = np.absolute(temp[0]-centroids[i,0]) + np.absolute(temp[1]-centroids[i,1]) + 0.01 # euclidean distance
                    countnum [labelmatrix[temp[0],temp[1]],i] += 1/distfactor # equalization

                labelhat[i] = countnum[:,i].argmax()
        else:
            print "No data generated"
            labelhat = -1
              
        centroidindex = np.argsort(centroids[:,0])            
        labelhat_sort = np.array(labelhat[centroidindex], dtype='int')
        
        '''
        center_cout = 1
        for j in centroidindex:

            print "Prediction", center_cout ,", Coordinates=", centroids[:,j], "Label ->", int(labelhat[j]) 
            center_cout += 1
        '''     
        
        return labelhat_sort 
        
    def findCentroids(self, data_coor):
                
        numDigits = 0
        mean_center = []        
        center = np.ones(data_coor.shape[1])
        num_center0 = np.zeros(data_coor.shape[1]) # to be cut    
        
        for i in range(data_coor.shape[1]):

            coor = np.array([data_coor[0,i], data_coor[1,i]]) 
            
            if i  == 0:
                mean_center.append( coor )
                center[i] = numDigits
                num_center0[numDigits] += 1
                numDigits += 1

            elif i > 0 :                   
                dist = np.zeros(numDigits)        
                for j in range(numDigits):
                    mean_center_tmp = mean_center[j]
                    tmp_coor_x = mean_center_tmp[0]
                    tmp_coor_y = mean_center_tmp[1]
                    
                    dist[j] = np.sqrt((coor[0] - tmp_coor_x)**2 + (coor[1] - tmp_coor_y)**2)
                     
                if min(dist) <= self.boundry_thresh:
                    
                    min_idx = dist.argmin() 
                    center[i] =  center[min_idx]
                    
                    #update mean of each center
                    mean_tmp_new = ( mean_center[min_idx] * num_center0[min_idx] + coor )/(num_center0[min_idx]+1)
                    mean_center[min_idx] = mean_tmp_new
                    num_center0[min_idx] += 1
                    
                else:
                    center[i] = numDigits
                    mean_center.append( coor )
                    num_center0[numDigits] += 1
                    numDigits += 1 
                    
        return center, numDigits
    
    