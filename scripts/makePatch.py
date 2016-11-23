
from utils.prep import *
import cv2, time, multiprocessing

tic = time.time()
imagePath = r"C:\ningboDataset\1920_1080\snapshot20160402171956.jpg"
img = cv2.imread(imagePath, 0)
stack_4d_1 = makePatch(img, (20, 20), (4,4), save_flg = False)
stack_4d_2 = makePatch(img, (30, 30), (6,6), save_flg = False)
stack_4d_3 = makePatch(img, (40, 40), (8,8), save_flg = False)
stack_4d_4 = makePatch(img, (50, 50), (10,10), save_flg = False)

print stack_4d_1[0].shape, stack_4d_2[0].shape, stack_4d_3[0].shape, stack_4d_4[0].shape
toc = time.time()
print toc-tic