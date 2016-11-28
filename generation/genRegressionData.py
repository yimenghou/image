import scipy.io 
import os, cv2
import numpy as np
import matplotlib.pylab as plt

# run this script directly to generate positive and negative ROI examples
# checked paded image at ROInew folder

class genROI(object):

	def __init__(self, roi):

		self.baseDataPath = r'E:\dataset\harbourCanvasImage'
		self.secondaryImg = 'image' # secondary directory that saves images
		self.secondaryAno = 'annotation' # secondary directory that saves annotation

		self.targetPath =r'E:\dataset\harbourRegressionData' 

		mapDict = {0:"Letter", 1:"Number", 2:"LetNum"}

		# change the size of image bound, each has 3 scales
		boundSizeDict = {"Letter":[[150,40],[175,45],[200,50]],\
						 "Number":[[200,40],[250,45],[300,50]],\
						 "LetNum":[[150,40],[175,45],[200,50]] }

		self.ROItype = roi
		self.targetBoundSize = 	boundSizeDict[mapDict[self.ROItype]]

	def getROI(self):
		# generate positives
		up_range = [0.24, 0.12]
		step_size = 2

		n_newImg = 0
		imageFileList = os.listdir(self.baseDataPath)

		for name in imageFileList: 

			print "Current processing folder:", name
			ImgDirPath = os.path.join(self.baseDataPath, name, self.secondaryImg) 
			AnoDirPath = os.path.join(self.baseDataPath, name, self.secondaryAno)

			for item in os.listdir(AnoDirPath):

				imageName_item = item[:-4] + '.jpg'
				imageName_item_save = item[:-4] + '.bmp'
				fullImageName = os.path.join(ImgDirPath, imageName_item)
				fullAnoName = os.path.join(AnoDirPath, item)

				matfile = scipy.io.loadmat(fullAnoName)
				position = matfile['boxes'][self.ROItype]

				hArmLength = position[2] - position[0]
				vArmLength = position[3] - position[1]
				position_center = [(position[2]+position[0])/2, (position[3]+position[1])/2]

				if  hArmLength > vArmLength:
					continue

				imageEntire = cv2.imread(fullImageName)
				print fullImageName

				if vArmLength > self.targetBoundSize[2][0]:

					n_success = 0
					while(n_success <4):

						x_delta = step_size*np.random.uniform(-hArmLength*up_range[1], hArmLength*up_range[1])
						y_delta = step_size*np.random.uniform(-vArmLength*up_range[0], vArmLength*up_range[0])

						area_overlay = (hArmLength-x_delta)*(vArmLength-y_delta)
						if float(area_overlay)/(hArmLength*vArmLength) < 0.8:
							continue

						img_shift = imageEntire[ int(position_center[1]-self.targetBoundSize[1][0]/2+y_delta) : \
												 int(position_center[1]+self.targetBoundSize[1][0]/2+y_delta) , \
												 int(position_center[0]-self.targetBoundSize[1][1]/2+x_delta) : \
												 int(position_center[0]+self.targetBoundSize[1][1]/2+x_delta) ]

						v_delta = (vArmLength - self.targetBoundSize[1][0])/self.targetBoundSize[1][0]
						h_delta = (hArmLength - self.targetBoundSize[1][1])/self.targetBoundSize[1][1]

						fix_position = [y_delta/self.targetBoundSize[2][0], x_delta/self.targetBoundSize[2][1], v_delta, h_delta]
						print fix_position
						fix_position = np.array(fix_position, dtype='float')
						mat_data = {"boxes": fix_position}

						saveName_img = os.path.join(self.targetPath, 'img', str(n_newImg)+'.bmp')
						saveName_mat = os.path.join(self.targetPath, 'mat', str(n_newImg)+'.mat')

						try:
							cv2.imwrite(saveName_img, img_shift)
							scipy.io.savemat(saveName_mat, mat_data)
							n_newImg += 1
							n_success += 1
							print "!"
						except:
							pass



def boundFilter(rand_coor, original_coor, thresh = 0.5):

	# this function essentially filter the negative ROI
	# if the random negative image has a overlapping area over thresh with any of the ROIs, it will 
	# return a False. otherwise return a True

	# rand_coor is the coordinate of opponent bound
	# original_coor is the coordinate of self bound

	# compare rand_coor with self one

	boundArea = float(original_coor[2]-original_coor[0])*(original_coor[3]-original_coor[1])

	if original_coor[0] <= rand_coor[2] <= original_coor[2] and \
	   original_coor[1] <= rand_coor[3] <= original_coor[3]:
	    area = (rand_coor[2] - original_coor[0])*(rand_coor[3] - original_coor[1])
	    if area/boundArea >= thresh:
	        return False
	elif original_coor[0] <= rand_coor[0] <= original_coor[2] and \
	    original_coor[1] <= rand_coor[1] <= original_coor[3]:
	    area = (rand_coor[0] - original_coor[0])*(rand_coor[1] - original_coor[1])
	    if area/boundArea >= thresh:
	        return False	
	elif original_coor[0] <= rand_coor[2] <= original_coor[2] and \
	    original_coor[1] <= rand_coor[1] <= original_coor[3]:
	    area = (rand_coor[2] - original_coor[0])*(rand_coor[1] - original_coor[1])
	    if area/boundArea >= thresh:
	        return False
	elif original_coor[0] <= rand_coor[0] <= original_coor[2] and \
	    original_coor[1] <= rand_coor[3] <= original_coor[3]:
	    area = (rand_coor[0] - original_coor[0])*(rand_coor[3] - original_coor[1])
	    if area/boundArea >= thresh:
	        return False
	else:
		return True

if __name__ == '__main__':

	ROI = genROI(1)
	ROI.getROI()