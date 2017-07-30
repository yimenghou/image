
"""
pad the images of non-uniform size into uniform ones
"""

import numpy as np
import os, cv2
import scipy

class singleImagePad(object):

	def __init__(self, sourceDataPath, targetDataPath):

		self.sourceDataPath = sourceDataPath
		self.targetDataPath = targetDataPath
		self.classNum = 36

		try:
			for item in os.listdir(self.sourceDataPath):
				subDirName = os.path.join(self.targetDataPath, item)
				os.mkdir(subDirName)
		except:
			pass

	def make(self):

		for class_item in os.listdir(self.sourceDataPath):

			classPath = os.path.join(self.sourceDataPath, class_item)
			N_sample = len(os.listdir(classPath))

			for sample in enumerate(os.listdir(classPath)):

				print "current processing index: %d, totally %d from class: %s"%(sample[0], N_sample, class_item)
				sampleLoadPath = os.path.join(classPath, sample[1])
				sampleSavePath = os.path.join(self.targetDataPath, class_item, sample[1])
				img = cv2.imread(sampleLoadPath)
				img_out = self.process(img)
				cv2.imwrite(sampleSavePath, img_out)

	def process(self, img_in):

		# since each image is not same size, the following will be done:
		# 1. set a target image size, eg: 32*32
		# 2. for every channel in the image, calculate the most possible pad value
		# 3. resize the longest height/width into 32, resize the another by 32*resize_factor
		# 4. pad the short arm with pad value, plus a random noise

		base_arm = 32.0
		img_out = np.ones((base_arm, base_arm, 3))

		for i in range(3):

			hist_val = cv2.calcHist([img_in],[i],None,[10],[0,256])
			hist_val = hist_val.flatten().tolist()

			sort_hist_val = np.sort(hist_val)
			argsort_hist_val = np.argsort(hist_val)

			if 7<=argsort_hist_val[-1]<=9 :
				pad_val = sort_hist_val[-2]
			elif 7<=argsort_hist_val[-2]<=9:
				pad_val = sort_hist_val[-1]
			elif 0<=argsort_hist_val[-1]<=2 :
				pad_val = sort_hist_val[-2]
			elif 0<=argsort_hist_val[-2]<=2 :
				pad_val = sort_hist_val[-1]
			else:
				pad_val = sort_hist_val[-2]

			img_out[:,:,i] = np.ones((base_arm, base_arm))*(hist_val.index(pad_val)*25.6) + np.random.rand(int(base_arm), int(base_arm))*12.8

		height, width, channels = img_in.shape

		if height > width:
			resize_arm = height
			resize_factor = base_arm/resize_arm
			img_base = cv2.resize(img_in, (int(width*resize_factor), int(base_arm)))
			pad_tot = base_arm - int(width*resize_factor)
			pad_half = int(pad_tot/2)
			pad_half0 = int(pad_tot - pad_half)
			img_out[:,pad_half:-pad_half0,:] = img_base 
		elif height < width:
			resize_arm = width
			resize_factor = base_arm/resize_arm
			img_base = cv2.resize(img_in, (int(base_arm), int(height*resize_factor)))
			pad_tot = base_arm - int(height*resize_factor)
			pad_half = int(pad_tot/2)
			pad_half0 = int(pad_tot - pad_half)
			img_out[pad_half:-pad_half0,:,:] = img_base
		else:
			img_out = cv2.resize(img_in, (int(base_arm), int(base_arm)))		 

		return img_out
		
if __name__ == "__main__":

	worker0 = singleImagePad()
	worker0.make()

