import scipy.io 
import os, cv2
import numpy as np
import matplotlib.pylab as plt

# run this script directly to generate positive and negative ROI examples
# checked paded image at ROInew folder

class genROI(object):

	def __init__(self, roi):

		dirpath = os.path.dirname(os.path.realpath(__file__))
		self.baseDataPath = r'E:\dataset\rawDataset'
		self.secondaryImg = 'image' # secondary directory that saves images
		self.secondaryAno = 'annotation' # secondary directory that saves annotation

		self.targetPath = os.path.join(dirpath, 'RegressionData')  

		self.ds =  1 # specify it if the original is downsampled 

		mapDict = {0:"Letter", 1:"Number", 2:"LetNum"}

		# change the size of image bound, each has 3 scales
		boundSizeDict = {"Letter":[[150,40],[175,45],[200,50]],\
						 "Number":[[200,40],[260,45],[320,50]],\
						 "LetNum":[[150,40],[175,45],[200,50]] }

		self.ROItype = roi
		self.targetBoundSize = 	boundSizeDict[mapDict[self.ROItype]]

	def getROI(self):
		# generate positives
		up_range = [0.1, 0.05]
		step_size = 2

		n_tot = 0
		n_newImg = 0
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

				matfile = scipy.io.loadmat(fullAnoName)
				position = matfile['boxes'][self.ROItype]
				position = position/self.ds

				vArmLength = int(position[2]) - int(position[0])
				hArmLength = int(position[3]) - int(position[1])

				if  vArmLength > hArmLength:
					# try only horizotal ROI
					continue

				imageEntire = cv2.imread(fullImageName)

				if hArmLength < self.targetBoundSize[0][0]:
					continue
					padvTotal = self.targetBoundSize[0][0] - hArmLength
					padLeft = padvTotal/2
					padRight = padvTotal - padLeft

					padhTotal = self.targetBoundSize[0][1] - vArmLength
					padTop = padhTotal/2
					padBot = padhTotal - padTop

					imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
					if imageCrop.shape != (self.targetBoundSize[0][0], self.targetBoundSize[0][1], 3):
						continue

				if self.targetBoundSize[0][0] <= hArmLength <= self.targetBoundSize[1][0]:
					continue
					padvTotal = self.targetBoundSize[1][0] - hArmLength
					padLeft = padvTotal/2
					padRight = padvTotal - padLeft

					padhTotal = self.targetBoundSize[1][1] - vArmLength
					padTop = padhTotal/2
					padBot = padhTotal - padTop

					imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
					if imageCrop.shape != (self.targetBoundSize[1][0], self.targetBoundSize[1][1], 3):
						continue

				if self.targetBoundSize[1][0] < hArmLength:

					padvTotal = self.targetBoundSize[2][0] - hArmLength
					padLeft = padvTotal/2
					padRight = padvTotal - padLeft

					padhTotal = self.targetBoundSize[2][1] - vArmLength
					padTop = padhTotal/2
					padBot = padhTotal - padTop

					imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
					if imageCrop.shape != (self.targetBoundSize[2][0], self.targetBoundSize[2][1], 3):
						continue

				fix_v_top  = step_size*np.random.randint(-vArmLength*up_range[0], vArmLength*up_range[0])
				fix_v_bot  = step_size*np.random.randint(-vArmLength*up_range[0], vArmLength*up_range[0])
				fix_h_left  = step_size*np.random.randint(-hArmLength*up_range[1], hArmLength*up_range[1])
				fix_h_right = step_size*np.random.randint(-hArmLength*up_range[1], hArmLength*up_range[1])

				img_shift = imageEntire[ int(position[1])-padLeft+fix_v_top:int(position[3])+padRight+fix_v_bot, \
										 int(position[0])-padTop+fix_h_left:int(position[2])+padBot+fix_h_right, :]

				fix_position = [fix_v_top, fix_v_bot, fix_h_left, fix_h_right]
				fix_position = np.array(fix_position, dtype='int')
				mat_data = {"boxes": fix_position}

				saveName_img = os.path.join(self.targetPath, 'img', str(n_newImg)+'.jpg')
				saveName_mat = os.path.join(self.targetPath, 'mat', str(n_newImg)+'.mat')
				cv2.imwrite(saveName_img, img_shift)
				scipy.io.savemat(saveName_mat, mat_data)

				n_newImg += 1



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

	for i in range(3):

		ROI = genROI(i)
		ROI.getROI()