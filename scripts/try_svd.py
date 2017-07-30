
'''
FFT SVD
'''
# import numpy as np
# import time
# import cv2
# import matplotlib.pylab as plt

# img_path_0 = r"E:\dataset\320+50(ROI)\0\00002-1.bmp"
# img_path_1 = r"E:\dataset\320+50(ROI)\0\00002-2.bmp"
# # C:\Users\westwell\Desktop

# img_0 = cv2.imread(img_path_0,0)
# img_1 = cv2.imread(img_path_1,0)

# fft_img_0 = np.fft.fft2(img_0)
# fft_img_1 = np.fft.fft2(img_1)
# fft_shift = np.fft.fftshift(fft_img_0)
# print fft_shift

# img0_phase = np.angle(fft_img_0)
# img0_mag = np.abs(fft_img_0)

# # print img0_phase[0,:]

# plt.figure()
# plt.subplot(131)
# plt.imshow(img_0)
# plt.subplot(132)
# plt.imshow(img0_phase)
# plt.subplot(133)
# plt.imshow(10*np.log10(img0_mag))
# plt.show()


# U0,S0,V0 = np.linalg.svd(fft_img_0, full_matrices=True)
# U1,S1,V1 = np.linalg.svd(fft_img_1, full_matrices=True)

# S0 = S0 - np.mean(S0)/(S0.max() - S0.min())
# print S0.shape

'''
FFT sinusoid
'''


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import fft

# Fs = 150                         # sampling rate
# Ts = 1.0/Fs                      # sampling interval
# t = np.arange(0,1,Ts)            # time vector
# ff = 5                           # frequency of the signal
# y = np.sin(2 * np.pi * ff * t)

# plt.subplot(2,1,1)
# plt.plot(t,y,'k-')
# plt.xlabel('time')
# plt.ylabel('amplitude')

# plt.subplot(2,1,2)
# n = len(y)                       # length of the signal
# k = np.arange(n)
# T = n/Fs
# frq = k/T # two sides frequency range
# freq = frq[range(n/2)]           # one side frequency range

# Y = np.fft.fft(y)/n              # fft computing and normalization
# Y = Y[range(n/2)]

# plt.plot(freq, abs(Y), 'r-')
# plt.xlabel('freq (Hz)')
# plt.ylabel('|Y(freq)|')

# plt.show()  

'''
Deskew
'''

# import cv2
# import numpy as np
# import matplotlib.pylab as plt

# affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

# def deskew(img):
# 	SZ = 100
# 	m = cv2.moments(img)
# 	if abs(m['mu02']) < 1e-2:
# 	    return img.copy()
# 	skew = m['mu11']/m['mu02']
# 	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
# 	img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
# 	return img   

# imgPath = r"E:\dataset\320+50(ROI)\1\00002.bmp"
# img = cv2.imread(imgPath,0)
# img_out = deskew(img)


# print img_out


# plt.figure()
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(img_out)
# plt.show()

'''
LBP
'''

# import the necessary packages
from skimage import feature
import numpy as np
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2grey

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist

imgPath = r"E:\dataset\320+50(ROI)\1\00002.bmp"
img = imread(imgPath,0)
img_grey = rgb2grey(img)
lbp0 = LocalBinaryPatterns(24,8)
hist = lbp0.describe(img_grey)
print hist