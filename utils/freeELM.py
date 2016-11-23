# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:32:47 2016

@author: Yimeng Hou
"""

'''
This class creates a feed-forward network consists of ELM implementation
'''


'''
To use this class:
open another empty script,

#initialize a freeELM class

FrameworkLayout = [numELM1layer, numELM2layer, numELM3layer, ...]
Note that: numELM1layer = 4**n , n = 1,2,3,... 
           numELMmlayer = even number
           numELMlastlayer = 1

as an example,
FrameworkLayout = [4, 2, 1]

oneELM = freeELM(FrameworkLayout)

# Train it
oneELM.trainLoop()

# Test it
accuracy = oneELM.testLoop()
'''


from ELM.ELM import ELM
import time
from ELM.MNISTDataset import MNISTDataset
from numpy import zeros, reshape, array
from segImage import segImage
from distCal import distCal


def accumulateSum(inputarray, num):
    tot0 = 0
    for i in range(num):
        tot0 += inputarray[i]
    return tot0
 
# feed-forward ELM network
   
class freeELM(object):
       
    def __init__(self, layout):
        
        self.timeRun = 0 # time
        self._numELM = layout
        self._numLayer = len(self._numELM)
        self._dataset = MNISTDataset("MNIST")
        self._complexity = 'lite'
        self._weightType = 'dec'
        self._sizeTable = zeros((3, self._numLayer))
        self._numTr = 6000 # training examples
        self._numTe = 6000 # testing examples
        self.accuracy = 0 
        
        for i in range(self._numLayer):
            if i == 0:            
                self._sizeTable[:,i] = array([ 28*28/self._numELM[0], 28*28*10/self._numELM[0], 10])
            else:
                self._sizeTable[:,i] = array([ self._numELM[i-1] * self._sizeTable[2, i-1] / self._numELM[i], \
                                               self._numELM[i-1] * self._sizeTable[2, i-1] / self._numELM[i] * 10, 10 ])
        self._ELMlist = []
            
        # initialize all ELM workers in a list
        tic_init = time.time()            
        
        cot = 1
        for ii in range(self._numLayer):
            for i in range(self._numELM[ii]): 
                print "> Initialize No.%d ELM worker..."%cot
                cot += 1
                self. _ELMlist.append(ELM(self._sizeTable[0, ii], self._sizeTable[1, ii], self._sizeTable[2, ii], \
                                self._complexity, self._weightType))
                                
        print "> Initialization complete!", sum(self._numELM), "ELM workers in total."
        
        toc_init = time.time()
        self.timeRun += toc_init - tic_init
        
    @property    
    def getELMspecs(self):
        return self._ELMlist
    def getTimeElapsed(self):
        return self.timeRun
    def getAccuracy(self):
        return self.accuracy
    
    # define the process of training one layer of ELMs                            
    def trainOneLayer(self, layerIdx, _labelTr, _itemTr):
        
        _ELMoutputList = zeros((self._numELM[layerIdx], 10))
        
        if layerIdx == 0:            
            _temp = reshape(_itemTr, (28,28))
            _dataseg_tr = segImage(_temp, self._numELM[layerIdx])
        else:
            _dataseg_tr = reshape(_itemTr, \
            (self._numELM[layerIdx], int (self._numELM[layerIdx-1] * 10 / self._numELM[layerIdx])))

        starteridx = accumulateSum(self._numELM, layerIdx)
        
        for i in range(self._numELM[layerIdx]):

           _ELMoutputList[i, :] = self._ELMlist[starteridx+i].train(_dataseg_tr[i, :], _labelTr).flatten()
            
        return _ELMoutputList.flatten()
    
    # define the process of iteratively training ELMs    
    def trainLoop(self):
        
        tic_tr = time.time()
        for i in range(self._numTr): 
            
            if (i+1)%int(0.05*self._numTr) == 0:
                print "> Training in progress...","{:.0%}".format(i/float(self._numTr)),"complete", "(%d of %d)"%(i+1, self._numTr)
                
            label_tr, item_tr = self._dataset.getTrainingItem(i)
            
            for j in range(self._numLayer):                  
                item_tr = self.trainOneLayer(j, label_tr, item_tr)                      
        toc_tr = time.time()
        
        self.timeRun += toc_tr-tic_tr
        
    # define the process of training one layer of ELMs        
    def testOneLayer(self, layerIdx, _itemTe):
        
        _ELMoutputList = zeros((self._numELM[layerIdx], 10))        
        
        if layerIdx == 0:            
            _temp = reshape(_itemTe, (28,28))
            _dataseg_te = segImage(_temp, self._numELM[layerIdx])
        else:
            _dataseg_te = reshape(_itemTe, \
            (self._numELM[layerIdx], int (self._numELM[layerIdx-1] * 10 / self._numELM[layerIdx])))
        
        starteridx = accumulateSum(self._numELM, layerIdx)
        
        for i in range(self._numELM[layerIdx]):

            _ELMoutputList[i, :] = self._ELMlist[starteridx+i].recall(_dataseg_te[i, :]).flatten()
            
        return _ELMoutputList.flatten()
        
    # define the process of iteratively testing ELMs            
    def testLoop(self):
        
        tic_te = time.time()        
        
        truelabelcount = 0        
        for i in range(self._numTe):
            
            if (i+1)%int(0.05*self._numTe) == 0:
                print "> Testing in progress...","{:.0%}".format(i/float(self._numTe)),"complete", "(%d of %d)"%(i+1, self._numTe)
                
            label_te, item_te = self._dataset.getTestingItem(i)
            
            for j in range(self._numLayer):  
                
                item_te = self.testOneLayer(j, item_te)
                
            if distCal(item_te) == distCal(label_te):
                truelabelcount += 1
                
        toc_te = time.time()
        
        self.timeRun += toc_te - tic_te   
        
        self.accuracy = truelabelcount/float(self._numTe)
        
        print "The classification accuracy:", "{:.0%}".format(self.accuracy)
        print "Time Elapsed:", int(self.timeRun), "seconds"
        
        return self.accuracy  
        
            
        