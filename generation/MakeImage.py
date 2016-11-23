
from loadImg import *

class MakeImage(object):

	def __init__(self):

		imagePath = r"C:\Users\westwell\Desktop\mycalligraphy\copy1"
		sel_class = [i for i in range(10)]

		baseImage = loadImg(imagePath, sel_class)
		self.dataset_base = baseImage.content
		self.labelset_base = baseImage.label

	def synImg(self, img_name_in, factor0, factor1):
	                      
	    row,col = img_name_in.shape
	    pts_original = np.float32([[1,1],[row-2,1],[1,col-2]])
	    pts_process = np.float32([[(row-2)*factor0,(col-2)*factor1],[row-2,1],[1,col-2]])     
	    M = cv2.getAffineTransform(pts_original, pts_process)
	    dst = cv2.warpAffine(img_name_in, M, (col,row))
	    img_processed = cv2.resize(dst, (56, 56), interpolation=cv2.INTER_LINEAR)
	    
	    return img_processed

	def processImg(self):

		factor0 = [-0.2, 0.3, 0.1]
		factor1 = [-0.2, 0.3, 0.1]

		replicate_num = (factor0[1]-factor0[0])*(factor1[1]-factor1[0])/factor1[2]/factor0[2]

		dataset_new = np.zeros((self.dataset_base.shape[0]*replicate_num, self.dataset_base.shape[1]))
		labelset_new = np.zeros((self.labelset_base.shape[0]*replicate_num, self.labelset_base.shape[1]))

        xy_cout = 0

        for i in range(self.dataset_base.shape[0]):   

			for x in np.arange(factor0[0], factor0[1], factor0[2]):
				for y in np.arange(factor1[0], factor1[1], factor0[2]):
					img_processed = self.synImg(self.dataset_base[i,:], x, y)
					dataset_new[xy_cout,:] = img_processed.flatten()
					labelset_new[xy_cout,:] = self.labelset_base[i,:]
					xy_cout += 1

		return dataset_new, labelset_new







