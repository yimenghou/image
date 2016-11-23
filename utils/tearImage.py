from prep import *
import itertools as it
import cv2
import matplotlib.pylab as plt
import time, os

oneImage = cv2.imread("landscape.jpg", 0)

result1 = makePatch( oneImage, (20,20), (4,4) )
result1 = makePatch( oneImage, (30,30), (6,6) )
result1 = makePatch( oneImage, (40,40), (8,8) )






