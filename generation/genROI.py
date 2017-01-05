import scipy.io 
import os, cv2
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

# generate positive and negative ROI examples
class genROI(object):

    def __init__(self, roi):

        #dirpath = os.path.dirname(os.path.realpath(__file__))
        dataBaseDir = "E:\\dataset\\harbourCanvasImage"
        targetBaseDir = "E:\\dataset\\harbourOthers"
        self.baseDataPath = dataBaseDir#os.path.join(dataBaseDir, 'rawDataset') #"C:\\ningboDataset"
        self.targetPath = os.path.join(targetBaseDir, 'ROInew')  #"E:\\ROInew"
        self.secondaryImg = 'image' #"ThreeRegions\\image"
        self.secondaryAno = 'annotation' #"ThreeRegions\\annotation"

        mapDict = {0:"Letter", 1:"Number", 2:"LetNum"}
        '''
        boundSizeDict = {"Letter":[[150,40],[175,45],[200,50]],\
                         "Number":[[200,40],[260,45],[320,50]],\
                         "LetNum":[[150,40],[175,45],[200,50]] }
        '''
        boundSizeDict = {"Letter":[[64,28],[160,36],[200,48]],\
                         "Number":[[200,30],[280,40],[320,50]],\
                         "LetNum":[[112,28],[160,36],[200,48]] }
        self.ROItype = roi
        self.targetBoundSize =     boundSizeDict[mapDict[self.ROItype]]

        repositoryName = [mapDict[self.ROItype] + '-' + str(self.targetBoundSize[0][0])+'-'+str(self.targetBoundSize[0][1]), \
                          mapDict[self.ROItype] + '-' + str(self.targetBoundSize[1][0])+'-'+str(self.targetBoundSize[1][1]), \
                          mapDict[self.ROItype] + '-' + str(self.targetBoundSize[2][0])+'-'+str(self.targetBoundSize[2][1])]

        # save path to positive and negative examples
        self.PositiveSaveDir = [os.path.join(self.targetPath, 'positive', repositoryName[0]),\
                                os.path.join(self.targetPath, 'positive', repositoryName[1]),\
                                os.path.join(self.targetPath, 'positive', repositoryName[2])]

        self.NegativeSaveDir = [os.path.join(self.targetPath, 'negative', repositoryName[0]),\
                                os.path.join(self.targetPath, 'negative', repositoryName[1]),\
                                os.path.join(self.targetPath, 'negative', repositoryName[2])]

        try:
            #for i in range(3):
                #os.mkdir(self.PositiveSaveDir[i])
            for j in range(3):
                os.mkdir(self.NegativeSaveDir[j])
        except:
            pass

    def getROI(self):

        print "Generating positive examples .."

        n_tot = 0
        imageFileList = os.listdir(self.baseDataPath)

        for name in imageFileList: 

            print "Current processing folder:", name
            ImgDirPath = os.path.join(self.baseDataPath, name, self.secondaryImg) 
            AnoDirPath = os.path.join(self.baseDataPath, name, self.secondaryAno)

            for item in os.listdir(AnoDirPath):
                n_tot += 1

                imageName_item = item[:-4] + '.jpg'
                imageName_item_save = item[:-4] + '.bmp'
                
                fullImageName = os.path.join(ImgDirPath, imageName_item)
                fullAnoName = os.path.join(AnoDirPath, item)
                #print "processing: " + fullImageName
                
                matfile = scipy.io.loadmat(fullAnoName)
                position = matfile['boxes'][self.ROItype]
                #ssprint position

                vArmLength = int(position[2]) - int(position[0])
                hArmLength = int(position[3]) - int(position[1])

                if  vArmLength > hArmLength:
                    # try only horizotal ROI
                    continue

                imageEntire = cv2.imread(fullImageName)

                if hArmLength < self.targetBoundSize[0][0]:

                    padvTotal = self.targetBoundSize[0][0] - hArmLength
                    padLeft = padvTotal/2
                    padRight = padvTotal - padLeft

                    padhTotal = self.targetBoundSize[0][1] - vArmLength
                    padTop = padhTotal/2
                    padBot = padhTotal - padTop

                    imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
                    if imageCrop.shape != (self.targetBoundSize[0][0], self.targetBoundSize[0][1], 3):
                        continue

                    saveName = os.path.join(self.PositiveSaveDir[0], imageName_item_save)
                    cv2.imwrite(saveName, imageCrop)

                elif self.targetBoundSize[0][0] <= hArmLength <= self.targetBoundSize[1][0]:

                    padvTotal = self.targetBoundSize[1][0] - hArmLength
                    padLeft = padvTotal/2
                    padRight = padvTotal - padLeft

                    padhTotal = self.targetBoundSize[1][1] - vArmLength
                    padTop = padhTotal/2
                    padBot = padhTotal - padTop

                    imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
                    if imageCrop.shape != (self.targetBoundSize[1][0], self.targetBoundSize[1][1], 3):
                        continue

                    saveName = os.path.join(self.PositiveSaveDir[1], imageName_item_save)
                    cv2.imwrite(saveName, imageCrop)

                elif self.targetBoundSize[1][0] < hArmLength:

                    padvTotal = self.targetBoundSize[2][0] - hArmLength
                    padLeft = padvTotal/2
                    padRight = padvTotal - padLeft

                    padhTotal = self.targetBoundSize[2][1] - vArmLength
                    padTop = padhTotal/2
                    padBot = padhTotal - padTop

                    imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
                    if imageCrop.shape != (self.targetBoundSize[2][0], self.targetBoundSize[2][1], 3):
                        continue

                    saveName = os.path.join(self.PositiveSaveDir[2], imageName_item_save)
                    cv2.imwrite(saveName, imageCrop)


    def getNeg(self):

        print "Generating negative examples .."

        negativePerCanvas = 2
        imageFileList = os.listdir(self.baseDataPath)

        for name in imageFileList: 

            print "Current processing folder:", name
            ImgDirPath = os.path.join(self.baseDataPath, name, self.secondaryImg) 
            AnoDirPath = os.path.join(self.baseDataPath, name, self.secondaryAno)

            for item in os.listdir(AnoDirPath):

                imageName_item = item[:-4] + '.jpg'
                fullImageName = os.path.join(ImgDirPath, imageName_item)
                #print "Processing: "+ fullImageName
                fullAnoName = os.path.join(AnoDirPath, item)

                matfile = scipy.io.loadmat(fullAnoName)
                imageEntire = cv2.imread(fullImageName)
                allPosition = matfile['boxes']

                for scale in range(3):

                    position = allPosition[self.ROItype]

                    vArmLength = int(position[2]) - int(position[0])
                    hArmLength = int(position[3]) - int(position[1])

                    if  vArmLength > hArmLength:
                        # try only horizotal ROI
                        continue

                    n_success = 1
                    n_tot_cout = 0
                    while(n_success <= negativePerCanvas):
                        n_tot_cout += 1
                        if n_tot_cout > 30:
                            break

                        topLeftCor = [np.random.randint(0, imageEntire.shape[0]-self.targetBoundSize[scale][0]), np.random.randint(0, imageEntire.shape[1]-self.targetBoundSize[scale][1]) ]
                        botRightCor = [ topLeftCor[0]+self.targetBoundSize[scale][0], topLeftCor[1]+self.targetBoundSize[scale][1] ]

                        # Dist2bound = np.zeros((3,4))
                        # for i in range(3):
                        #     for j in range(4):
                        #         Dist2bound[i,j] = allPosition[i][j]
                        # # Dist2bound = [ abs() ] 

                        if topLeftCor[0] <= position[0] <= botRightCor[0] or topLeftCor[0] <= position[1] <= botRightCor[0] \
                            or topLeftCor[0] <= position[2] <= botRightCor[0] or topLeftCor[0] <= position[3] <= botRightCor[0]:
                            pass
                        elif topLeftCor[1] <= position[0] <= botRightCor[1] or topLeftCor[1] <= position[1] <= botRightCor[1] \
                            or topLeftCor[1] <= position[2] <= botRightCor[1] or topLeftCor[1] <= position[3] <= botRightCor[1]:
                            pass
                        else:
                            imageCropNegative = imageEntire[topLeftCor[0]:botRightCor[0], topLeftCor[1]:botRightCor[1]]
                            if imageCropNegative.shape != (self.targetBoundSize[scale][0], self.targetBoundSize[scale][1], 3):
                                continue

                            imageName_item_save = item[:-4] + '-' + str(n_success) + '.bmp'
                            saveName = os.path.join(self.NegativeSaveDir[scale], imageName_item_save)
                            #print saveName
                            cv2.imwrite(saveName, imageCropNegative)
                            n_success += 1

if __name__ == '__main__':
    '''
    for i in range(3):

        ROI = genROI(i)
        ROI.getROI()
        #ROI.getNeg()
    '''
    ROI = genROI(1)
    #ROI.getROI()
    ROI.getNeg()

