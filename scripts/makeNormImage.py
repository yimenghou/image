import numpy as np
import cv2
import matplotlib.pylab as plt
from scipy import signal

def zeropadding(matrix, padsize,  constant_val = 0):
    return np.pad(matrix, padsize, 'constant', constant_values = constant_val)

def size_normalize(input_img):
    # normalize the image content to fixed size

    dst_arm = 32
    dstimg_size = (40,40)

    img = cv2.imread(input_img, 0)
    hist = cv2.calcHist([img], [0], None, [64], [0,256])

    xy = np.where(img>0)
    img_xy = [xy[0].min(), xy[0].max(), xy[1].min(), xy[1].max()]
    img_crop = img[img_xy[0]:img_xy[1], img_xy[2]:img_xy[3]]

    thresh_1 = (img_xy[1] - img_xy[0])/float((img_xy[3] - img_xy[2]))
    thresh_2 = (img_xy[3] - img_xy[2])/float((img_xy[1] - img_xy[0]))

    if thresh_1 > thresh_2:
        thresh = thresh_1
        img_resize = cv2.resize(img_crop, (int(dst_arm/thresh), dst_arm))
    else:
        thresh = thresh_2
        img_resize = cv2.resize(img_crop, (dst_arm, int(dst_arm/thresh)))

    toppad = int(round( (dstimg_size[0]-img_resize.shape[0])/2 ))
    botpad = dstimg_size[0]-img_resize.shape[0]-toppad
    leftpad = int(round( (dstimg_size[1]-img_resize.shape[1])/2 ))
    rightpad = dstimg_size[1]-img_resize.shape[1]-leftpad
    padsize = ((toppad, botpad), (leftpad, rightpad))

    img_pad = zeropadding(img_resize, padsize)

    return img, img_pad

imgPath = r'E:\dataset\ningbo_new\0\2.bmp'
img = cv2.imread(imgPath,0)
v1 = np.array([2,1,0,-1,-2])
v2 = np.array([-2,-1,0,1,2])

hist = cv2.calcHist([img],[0],None,[32],[0,256])
fltImg1 = signal.convolve(hist.flatten(), v1, mode='valid')
fltImg2 = signal.convolve(hist.flatten(), v2, mode='valid')

plt.figure()
plt.subplot(132)
plt.plot(hist)
plt.subplot(131)
plt.plot(fltImg1)
plt.subplot(133)
plt.plot(fltImg2)


img_org, img_pad = size_normalize(imgPath)
plt.figure()
plt.subplot(121)
plt.imshow(img_org)
plt.subplot(122)
plt.imshow(img_pad)
plt.show()




