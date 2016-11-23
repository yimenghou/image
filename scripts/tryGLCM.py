
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2grey
from skimage.exposure import histogram
import numpy as np

featureBag = []

path = r"C:\Users\westwell\Desktop\xProject\Data\1\00002.bmp"
img = imread(path)
grey_img = rgb2grey(img)

# color 
rgb_val = [img[:,:,i].mean() for i in range(3)] #[img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()]
hsv_img = rgb2hsv(img)
hsv_val = [hsv_img[:,:,i].mean() for i in range(3)]

hist = histogram(grey_img, 16)
hist_val = hist[0]

# GLCM
descriptor_val = []
prop_list = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'correlation']

descriptor_val = np.zeros(100)
for i in enumerate(prop_list):
	glcm = greycomatrix(np.uint8(grey_img*255), [1,2,3,4,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
	temp = greycoprops(glcm, prop=i[1])

	descriptor_val[i[0]*20:(i[0]+1)*20] = np.hstack((temp))


print np.hstack((np.array(rgb_val), np.array(hsv_val), hist_val, descriptor_val) )




