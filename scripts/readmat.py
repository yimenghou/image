
import scipy.io
import os, cv2, math
import matplotlib.pylab as plt

label_cout = [0]*10

try:
	tarImgBasePath = r'E:\dataset\ningbo_new'
	for i in range(10):
		os.mkdir( os.path.join(tarImgBasePath,str(i)) )
except:
	pass

BasePath = r'C:\ningboDataset'

for worker in os.listdir( BasePath ):
	print "current worker", worker
	matBasePath = os.path.join(BasePath, worker, r"C:\ningboDataset\1_ating\Secondregion\annotation")
	imgBasePath = os.path.join(BasePath, worker, r"C:\ningboDataset\1_ating\Secondregion\AddRectangle")

	for imgitem in os.listdir(imgBasePath):
		matitem = imgitem[:5]+'.mat'
		matfile = scipy.io.loadmat( os.path.join(matBasePath, matitem) )
		imgfile = cv2.imread( os.path.join(imgBasePath, imgitem))
		matcontent = matfile['boxes']
		rows, cols = matcontent.shape

		for i in range(rows):
			topleft = [int(matcontent[i,1])+2, int(matcontent[i,2])+2]
			botright = [ int(math.ceil(matcontent[i,3]))-2, int(math.ceil(matcontent[i,4]))-2 ]
			label = int(matcontent[i,0])

			img_crop = imgfile[ topleft[1]:botright[1], topleft[0]:botright[0] ]

			saveImgName = str(label_cout[label])+'.bmp'
			tarImgClassPath = os.path.join(tarImgBasePath, str(label))
			tarImgPath = os.path.join(tarImgClassPath, saveImgName)
			cv2.imwrite( tarImgPath, img_crop)
			label_cout[label] += 1 



	
