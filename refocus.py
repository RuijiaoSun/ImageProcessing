# This program is for removing the unfocused part in an images.
# The program has been integrated into "ImageProcessing.py".
import os
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time

def winVar(img, wlen):
  wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen), borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
  return wsqrmean - wmean*wmean

def var_f(img, win):
    mean = ndimage.uniform_filter(img, win)
    mean_sqr = ndimage.uniform_filter(img**2, win)
    var = mean_sqr - mean**2
    return var

# Read images.
root_dir = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE_Refreshed'
new_dir = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE_Refocused_2'
start = time.time()
# Calculate the variance of each image.
for i in os.listdir(root_dir):
    img_dir = os.path.join(root_dir, i)
    img = cv2.imread(img_dir, 2)                # Read a uint-16 .tiff image,
    new_path = os.path.join(new_dir, i)
    # var = winVar(img, 1)
    var_scipy = ndimage.filters.generic_filter(img, np.var, size=3)
    # var_scipy = var_f(img, 2)
    # Pick those dots that fit well.
    img[var_scipy < np.mean(var_scipy)] = 0

    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite(new_path, img)
    # # Display.
    # plt.figure()
    # plt.imshow(img)
    # plt.colorbar()
    # plt.title("threshold mean")
    # plt.show()

end = time.time()
print("The time for one loop of generic_filter is " + str(end-start) + 's.')