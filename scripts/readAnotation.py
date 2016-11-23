
# import scipy.io
import matplotlib.pylab as plt
import scipy.ndimage.filters
import numpy as np
from scipy.spatial.distance import pdist, squareform
# import cv2

# Imgpath = r"C:\ningboDataset\1_ating\ThreeRegions\image\00001.jpg"
# Anopath = r"C:\ningboDataset\1_ating\ThreeRegions\annotation\00001.mat"
# img = cv2.imread(Imgpath, 0)
# ano = scipy.io.loadmat(Anopath)

# plt.figure()
# plt.imshow(img)
# plt.show()
# print ano['boxes'][0,:]

a = np.ones((10,10), dtype='float')
s = 0.5
pairwise_dists = squareform(pdist(a, 'euclidean'))
K = scipy.exp(pairwise_dists ** 2 / s ** 2)

print K