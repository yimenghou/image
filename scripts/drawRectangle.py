import os
import cv2
import scipy.io

path = 'C:\\Users\\westwell\\Desktop\\make\\rawDataset\\1'

anopath = os.path.join(path, "annotation")
imgpath = os.path.join(path, "image")

savepath = "C:\\Users\\westwell\\Desktop\\cv2"

for item in os.listdir(anopath):

	index = item[:-4]
	img_name = index + ".bmp"
	img = cv2.imread(os.path.join(imgpath, img_name))

	matfile = scipy.io.loadmat(os.path.join(anopath, item))

	for i in range(3):
		position = matfile['boxes'][i]
		cv2.rectangle(img, (int(position[0]/2),int(position[1]/2)), (int(position[2]/2),int(position[3]/2)), \
			color=(192,192,192), thickness=2) 

	cv2.imwrite(os.path.join(savepath, img_name), img)



