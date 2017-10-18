# Andrew Pulver

import numpy as np
import cv2

img = cv2.imread('gt3gray.jpg',cv2.IMREAD_GRAYSCALE).astype(np.float32)

# histogram normalization

hist_normalized_img = ((img-np.min(img))*(255)/(np.max(img) - np.min(img))).astype(np.uint8)

# histogram equalization

flattened_img = img.flatten()

L = 256 # number of distinct intensity values
cdf = np.sum(np.array([flattened_img < i for i in range(L)],dtype=np.float32),axis=1)/np.shape(flattened_img)

yshape,xshape = np.shape(img)

hist_equalized_img = np.zeros(np.shape(img))

for y in range(yshape):
    for x in range(xshape):
        hist_equalized_img[y,x] = np.floor((L-1)*cdf[int(img[y,x])])




cv2.imwrite('gt3_normalized.jpg',hist_normalized_img)
cv2.imwrite('gt3_equalized.jpg',hist_equalized_img)



