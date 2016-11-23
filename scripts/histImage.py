import scipy.io 
import os, cv2
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

basePath = r"C:\ningboDataset"
secondaryImg = r"ThreeRegions\image"
secondaryAno = r"ThreeRegions\annotation"
imageFileList = os.listdir(basePath)

targetBoundSize1 = (150, 40)
PositiveSaveDir1 = r"E:\ROInew\positive\150-40"
NegativeSaveDir1 = r"E:\ROInew\positive\150-40"

targetBoundSize2 = (175, 45)
PositiveSaveDir2 = r"E:\ROInew\positive\175-45"
NegativeSaveDir2 = r"E:\ROInew\positive\175-45"

targetBoundSize3 = (200, 50)
PositiveSaveDir3 = r"E:\ROInew\positive\200-50"
NegativeSaveDir3 = r"E:\ROInew\positive\200-50"


n_tot = 0
for name in enumerate(imageFileList): 

	print "current folder", name[0]
	ImgDirPath = os.path.join(basePath, name[1], secondaryImg) 
	AnoDirPath = os.path.join(basePath, name[1], secondaryAno)

	for item in os.listdir(AnoDirPath):
		n_tot += 1

		imageName_item = item[:-4] + '.jpg'
		imageName_item_save = item[:-4] + '.bmp'
		fullImageName = os.path.join(ImgDirPath, imageName_item)
		fullAnoName = os.path.join(AnoDirPath, item)

		matfile = scipy.io.loadmat(fullAnoName)
		position = matfile['boxes'][0]

		vArmLength = int(position[2]) - int(position[0])
		hArmLength = int(position[3]) - int(position[1])

		if  vArmLength > hArmLength:
			# try only horizotal ROI
			continue

		imageEntire = cv2.imread(fullImageName)

		# try:

		# 	if hArmLength < targetBoundSize1[0]:

		# 		padvTotal = targetBoundSize1[0] - hArmLength
		# 		padLeft = padvTotal/2
		# 		padRight = padvTotal - padLeft

		# 		padhTotal = targetBoundSize1[1] - vArmLength
		# 		padTop = padhTotal/2
		# 		padBot = padhTotal - padTop

		# 		imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
		# 		if imageCrop.shape != (targetBoundSize1[0], targetBoundSize1[1], 3):
		# 			continue

		# 		saveName = os.path.join(PositiveSaveDir1, imageName_item_save)
		# 		cv2.imwrite(saveName, imageCrop)

		# 	elif targetBoundSize1[0] <= hArmLength <= targetBoundSize2[0]:

		# 		padvTotal = targetBoundSize2[0] - hArmLength
		# 		padLeft = padvTotal/2
		# 		padRight = padvTotal - padLeft

		# 		padhTotal = targetBoundSize2[1] - vArmLength
		# 		padTop = padhTotal/2
		# 		padBot = padhTotal - padTop

		# 		imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
		# 		if imageCrop.shape != (targetBoundSize2[0], targetBoundSize2[1], 3):
		# 			continue

		# 		saveName = os.path.join(PositiveSaveDir2, imageName_item_save)
		# 		cv2.imwrite(saveName, imageCrop)

		# 	elif targetBoundSize2[0] <= hArmLength <= targetBoundSize3[0]:

		# 		padvTotal = targetBoundSize3[0] - hArmLength
		# 		padLeft = padvTotal/2
		# 		padRight = padvTotal - padLeft

		# 		padhTotal = targetBoundSize3[1] - vArmLength
		# 		padTop = padhTotal/2
		# 		padBot = padhTotal - padTop

		# 		imageCrop = imageEntire[ int(position[1])-padLeft:int(position[3])+padRight, int(position[0])-padTop:int(position[2])+padBot, :]
		# 		if imageCrop.shape != (targetBoundSize3[0], targetBoundSize3[1], 3):
		# 			continue

		# 		saveName = os.path.join(PositiveSaveDir3, imageName_item_save)
		# 		cv2.imwrite(saveName, imageCrop)

		# except:

		# 	pass

		for i in range(5):

			topLeftCor = [np.random.randint(0, imageEntire.shape[0]-targetBoundSize1[0]), np.random.randint(0, imageEntire.shape[1]-targetBoundSize1[1]) ]
			botRightCor = [ topLeftCor[0]+targetBoundSize1[0], topLeftCor[1]+targetBoundSize1[1] ]
			
			if topLeftCor[0] <= position[0] <= botRightCor[0] or topLeftCor[0] <= position[1] <= botRightCor[0] or topLeftCor[0] <= position[2] <= botRightCor[0]:
				pass
			elif topLeftCor[1] <= position[0] <= botRightCor[1] or topLeftCor[1] <= position[1] <= botRightCor[1] or topLeftCor[1] <= position[2] <= botRightCor[1]:
				pass
			else:
				imageCropNegative = imageEntire[topLeftCor[0]:botRightCor[0], topLeftCor[1]:botRightCor[1]]

				plt.figure()
				plt.imshow(imageCropNegative)
				plt.show()





















