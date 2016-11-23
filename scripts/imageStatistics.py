

import os
import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

basePath = r"C:\ningboDataset"
secondaryPath = r"Secondregion\image"

imageFileList = os.listdir(basePath)

v_list = []
h_list = []

n_tot = 0
for name in enumerate(imageFileList): 

	print "current folder", name[0]
	dirPath = os.path.join(basePath, name[1], secondaryPath) 

	for item in os.listdir(dirPath):
		n_tot += 1

		fullImageName = os.path.join(dirPath, item)

		try:
			img = cv2.imread(fullImageName, 0)
			hist = cv2.calcHist([img], [0], None, [16], [0, 256])

			# plt.figure()
			# plt.plot(hist)
			# plt.show()

			if img.shape[0] > img.shape[1]:
				# vertical ROI
				v_list.append( img.size )
			else:
				h_list.append( img.size )
		except:
			continue

print n_tot

v_image = np.array( v_list )
h_image = np.array( h_list ) 

hist_val = np.histogram(v_image, bins=50)
b, a = signal.butter(2, 0.1)
y_flt = signal.filtfilt(b,a, hist_val[0], method='gust')

print hist_val[1]

plt.figure()
plt.plot(y_flt,'r.-')
plt.plot(hist_val[0], 'b.-')
plt.show()








