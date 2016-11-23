
import cv2
import matplotlib.pylab as plt
import numpy as np

def zeropadding(matrix, padsize,  constant_val = 0):
    return np.pad(matrix, padsize, 'constant', constant_values = constant_val)

def size_normalize(input_img):
    # normalize the image content to fixed size

    dst_arm = 30
    dstimg_size = (32,32)

    img = cv2.imread(input_img, 0)
    thresh = img.shape[1]/float(img.shape[0])

    if thresh <= 1 :
        img_resize = cv2.resize(img, (int(dst_arm*thresh), dst_arm))
    else:
        img_resize = cv2.resize(img, (dst_arm, int(dst_arm*thresh)))

    toppad = int(round( (dstimg_size[0]-img_resize.shape[0])/2 ))
    botpad = dstimg_size[0]-img_resize.shape[0] - toppad
    leftpad = int(round( (dstimg_size[1]-img_resize.shape[1])/2 ))
    rightpad = dstimg_size[1]-img_resize.shape[1] - leftpad
    padsize = ((toppad, botpad), (leftpad, rightpad))
    img_pad = zeropadding(img_resize, padsize)

    return img_pad

path = r"E:\dataset\harbourSingleImage\6\2991.bmp"
img_out = size_normalize(path)

plt.figure()
plt.imshow(img_out)
plt.show()


