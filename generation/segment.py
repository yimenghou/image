# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:48:26 2016

@author: Andre
"""

from pylab import *
from scipy import misc
from matplotlib import gridspec
from scipy.signal import argrelextrema


# read image
imgPath = r"E:\dataset\harbourROIImage\positive\Number-320-50\00002.bmp"
img = misc.imread(imgPath)
# make greyscale
greyimg = img.mean(axis=2).astype(uint8)
# select region of interest - I'm assuming this is known
# roi = greyimg[200:700,1250:1300]
roi = greyimg
# find the maximum value in each row and subtract the mean of these values
rowmax = roi.max(axis=1)-roi.max(axis=1).mean()
# wherever this is negative is where the whole row is (dark) background
segs = rowmax>0

# convolve to generate average of three neighbours
# this only keeps segments larger than a single row
# sig = argrelextrema(rowmax, np.greater, order=4)
segs = convolve(segs, [1,1,1], mode = 'same') 
# set these rows to zero
roi[segs] = 0
print segs

beginning = [0]
ending = []
for i in range(1, len(segs)-1):
	if segs[i-1] != 0 and segs[i] == 0:
		ending.append(i)
	elif segs[i] == 0 and segs[i+1] != 0:
		beginning.append(i)
	else:
		pass

print beginning
print ending


figure(3)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
subplot(gs[0])
imshow(flipud(roi.T)) # rotate the image to show with the segs plot
subplot(gs[1])
plot(segs)
show()

# imgPath2 = r"C:\ningboDataset\1_ating\ThreeRegions\image\00046.jpg"
# img = misc.imread(imgPath2)
# greyimg = img.mean(axis=2).astype(uint8)
# roi = greyimg[200:700,1460:1510]
# rowmax = roi.max(axis=1)-roi.max(axis=1).mean()
# segs = rowmax<0
# segs = convolve(segs, [1,1,1], mode = 'same') 
# roi[segs] = 0

# figure(1)
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
# subplot(gs[0])
# imshow(flipud(roi.T))
# subplot(gs[1])
# plot(segs)
# show()
