
import scipy.io
import cv2, os
import numpy as np
import xmltodict
import scipy.signal
import pylab as plt

class ROI2Image(object):

    def __init__(self):

        self.basePath = r"E:\dataset\ST\data"
        self.imgPath = os.path.join(self.basePath, "JPEGImages")
        self.xmlPath = os.path.join(self.basePath, "Annotations")
        self.savePath = r"C:\Users\westwell\Desktop\xmlImage"
        self.mapDict = {'one':1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "zero":0}

        try:
            for i in range(10):
                os.mkdir( os.path.join(self.savePath, str(i)) )
        except:
            pass

    def run(self):

        classNum = np.zeros(10, dtype=np.int)
        for item in os.listdir(self.xmlPath):
            idx = item[:-4]

            img = cv2.imread( os.path.join(self.imgPath, idx+".bmp") )
            xml = self._readxml( os.path.join(self.xmlPath, item) )

            if img == None:
                continue
            elif img.shape[0]< img.shape[1]:
                continue

            img_stack = self._segmentROI(img)

            try:
                for i in range(len(img_stack)):
                    ImgSavePath = os.path.join(self.savePath, str(xml[i]), str(classNum[xml[i]])+'.bmp')
                    classNum[xml[i]] += 1
                    print ImgSavePath
                    cv2.imwrite(ImgSavePath, img_stack[i])
            except:
                continue

    def _readxml(self, input_xml_file_path):

        with open(input_xml_file_path) as fd:
            doc = xmltodict.parse(fd.read())

        label_lst = []
        for i in range(len(doc['annotation']['object'])):
            label_lst.append( self.mapDict[ doc['annotation']['object'][i]['name'] ] )

        return label_lst

    def _segmentROI(self, input_roi):

        height, width, _ = input_roi.shape

        if height > width:
            frag_direction = 1
        else:
            frag_direction = 0

        greyimg = cv2.cvtColor(input_roi, cv2.COLOR_BGR2GRAY)
        rowmax = greyimg.max(axis=frag_direction)-greyimg.max(axis=frag_direction).mean()
        segs = rowmax.flatten() < 0
        segs = scipy.signal.convolve(segs, [1,1,1], mode='same')

        beginning, ending = [], []
        img_lst = []

        for i in range(1, len(segs)-1):
            if segs[i-1] != 0 and segs[i] == 0:
                beginning.append(i)
            elif segs[i] == 0 and segs[i+1] != 0:
                ending.append(i)
            else:
                pass

        if frag_direction == 1:

            for i in range(min(len(ending), len(beginning))):
                img_lst.append( input_roi[beginning[i]:ending[i],:] )   

        elif frag_direction == 0:

            for i in range(min(len(ending), len(beginning)) ):
                img_lst.append( input_roi[:,beginning[i]:ending[i]] )                

        return img_lst

if __name__ == "__main__":

    worker0 = ROI2Image()
    worker0.run()