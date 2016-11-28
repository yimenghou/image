
import numpy as np
import os, cv2
import scipy

class singleImagePad(object):

	def __init__(self):

		self.sourceDataPath = r"E:\dataset\harbourSingleImage"
		self.targetDataPath = r"E:\dataset\harbourSingleImagePad"
		self.classNum = 10

		try:
			for i in range(self.classNum):
				subDirName = os.path.join(self.targetDataPath, str(i))
				os.mkdir(subDirName)
		except:
			pass

	def make(self):

		for class_item in os.listdir(self.sourceDataPath):

			classPath = os.path.join(self.sourceDataPath, class_item)
			N_sample = len(os.listdir(classPath))

			for sample in enumerate(os.listdir(classPath)):

				print "current processing index: %d, totally %d"%(sample[0], N_sample)
				sampleLoadPath = os.path.join(classPath, sample[1])
				sampleSavePath = os.path.join(self.targetDataPath, class_item, sample[1])
				img = cv2.imread(sampleLoadPath)
				img_out = self.process(img)
				cv2.imwrite(sampleSavePath, img_out)

	def process(self, img_in):

		hist_val = cv2.calcHist([img_in],[0],None,[16],[0,256])
		sort_hist_val = np.sort(hist_val)
		argsort_hist_val = np.argsort(hist_val)

		if 14<=argsort_hist_val[0]<=16 :
			pad_val = sort_hist_val[1]
		elif 14<=argsort_hist_val[1]<=16:
			pad_val = sort_hist_val[0]
		elif 0<=argsort_hist_val[0]<=2 :
			pad_val = sort_hist_val[1]
		elif 0<=argsort_hist_val[1]<=2 :
			pad_val = sort_hist_val[0]
		else:
			pad_val = sort_hist_val[1]

		base_arm = 32.0
		img_out = np.ones((base_arm, base_arm, 3))*pad_val*16

		height, width, channels = img_in.shape

		if height > width:
			resize_arm = height

			resize_factor = base_arm/resize_arm

			img_base = cv2.resize(img_in, (int(base_arm), int(width*resize_factor)))
			pad_tot = base_arm - int(width*resize_factor)
			pad_half = int(pad_tot/2)
			pad_half0 = pad_tot - pad_half
			img_out[pad_half:-pad_half0,:,:] = img_base 
		else:
			resize_arm = width
			resize_factor = base_arm/resize_arm
			img_base = cv2.resize(img_in, (int(height*resize_factor), int(base_arm)))
			pad_tot = base_arm - int(height*resize_factor)
			pad_half = int(pad_tot/2)
			pad_half0 = pad_tot - pad_half
			img_out[:,pad_half:-pad_half0,:] = img_base 

		return img_out
		

if __name__ == "__main__":

	worker0 = singleImagePad()
	worker0.make()

