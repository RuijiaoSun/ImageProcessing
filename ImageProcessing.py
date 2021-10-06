# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:00:10 2021


Thid code is based on ants library.

It can be used for denoising, registration2D, registration3D, and postprocessing for registered images.

@author: ruijiao
"""
import argparse
import os
from os.path import join
import sys
import cv2
import ants
import time
from libtiff import TIFF
import numpy as np
from scipy import ndimage

def var_map(img, win):
    mean = ndimage.uniform_filter(np.float32(img), win) # mean
    sqr_mean = ndimage.uniform_filter(np.float32(img)**2, win)
    var = sqr_mean - mean**2
    var /= mean # normalization.
    return var

# ImageProcessing.py '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/OCT' '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE' '/home/ruijiao/A_research/registration/embryo_DAPI/Merged_threshold5'
def usage():
    print("Please input: ")
    print("ImageProcessing.py  'Path For Wide-Field Images'  'Path For High-Resolution Images'  'Path For Post-processed Images' ")
    print("Example: ImageProcessing.py  '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/OCT' "
          "     '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE'  '/home/ruijiao/A_research/registration/embryo_DAPI/Merged_threshold5'")
    print("Requirements: The Format of input images should be '.tiff'.")
    print("Output images will be '.png' format for better visualization.")

def crop(OCT_Raw, OCT_Cropped):
    # Path of Cropped images: OCT_Cropped
    # Images to be cropped named "OCT_PATH_ori".
    # OCT_PATH_Ori is the address of raw .tiff OCT images.
    for i in os.listdir(OCT_Raw):
        if i[-4:-1] == 'tif':
            tif = TIFF.open(os.path.join(OCT_Raw, i), 'r')
            img = tif.read_image()
            img_crop = img[0:480, 138:573]     # [y_range, x_range]
            img_interp = cv2.resize(img_crop, (2048, 2048), interpolation=cv2.INTER_LINEAR)
            OCT_Cropped_img = os.path.join(OCT_Cropped, str(int(i[-8:-5]) - 1).zfill(3) + '.png')
            cv2.imwrite(OCT_Cropped_img, img_interp)

def refresh(LS_Raw, MPE_Refreshed):
    images = os.listdir(LS_Raw)
    for img_name in images:
        # LS images' name length: 20.
        # OCT images' name length: 16.
        if len(img_name) == 20:
            img_path = os.path.join(LS_Raw, img_name)
            img2 = os.path.join(MPE_Refreshed, img_name[-8: -5] + '.png')
            # img = imageio.imread(img_path).astype(np.unit16)

            # # First processing.
            # img = cv2.imread(img_path)
            # img_new = img.copy()
            # img_new[img>0] = 255

            # Second processing.
            img_new = cv2.imread(img_path, -1)
            img_new = (img_new - np.min(img_new)) / (np.max(img_new) - np.min(img_new)) * 255
            # img_new[img_new > 122] = 255
            # img_new[img_new < 123] *= 2
            img_new = img_new.astype(np.uint8)

            # img_new = img / np.max(img) * 255
            # img_new = img + (255 - np.max(img))
            cv2.imwrite(img2, img_new)

def register(OCT_Cropped, MPE_Refreshed, REG_TIF_PATH):
    start = time.time()

    # Register a good image pair as standrad.
    # The paths are for cropped .tiff OCT images & .tiff LS imgae.
    fix_path = OCT_Cropped + str(len(os.listdir(MPE_Refreshed)) // 2)+'.png'
    move_path = MPE_Refreshed + str(len(os.listdir(MPE_Refreshed)) // 2)+'.png'
    # save_path = join(TIFF_PATH, 'std.tif')
    fix_ants = ants.image_read(fix_path)
    move_ants = ants.image_read(move_path)
    outs = ants.registration(fix_ants, move_ants, type_of_transform='SyN')
    # std_img = ants.apply_transforms(fix_ants, move_ants, transformlist = outs['fwdtransforms'], interpolator = 'linear')
    # ants.image_write(std_img, save_path)

    stop1 = time.time()
    print("registration time: " + str(stop1 - start) + " s.")
    for i_MPE in os.listdir(MPE_Refreshed):
        # Set images path.
        img_Name = i_MPE
        i_OCT = img_Name
        save_path = join(REG_TIF_PATH, 'registered' + img_Name[0:-4] + '.tiff')

        move_img = ants.image_read(join(MPE_Refreshed, i_MPE))
        fix_img = ants.image_read(join(OCT_Cropped, i_OCT))

        # Apply transformation field "outs" to each image pair.
        reg_img = ants.apply_transforms(fix_img, move_img, transformlist=outs['fwdtransforms'], interpolator='linear')

        # Set the direction, origin, spacing of registered image the same as fixed images.
        reg_img.set_origin(fix_img.origin)
        reg_img.set_spacing(fix_img.spacing)
        reg_img.set_direction(fix_img.direction)

        # Save registered image as .tiff.
        ants.image_write(reg_img, save_path)

    stop = time.time()
    print("Applying transformation field to all images time: " + str(stop - stop1) + " s.")

def merge(Registered, OCT_Raw, Merged):
    import matplotlib.pyplot as plt
    for i in os.listdir(OCT_Raw):
        reg_path = os.path.join(Registered, 'registered' + str(int(i[-8:-5]) - 1).zfill(3) + '.tiff')
        oct_path = os.path.join(OCT_Raw, i)
        save_path = os.path.join(Merged, 'merged' + i[8:11] + '.png')

        reg = cv2.imread(reg_path, 2)
        oct = cv2.imread(oct_path, 1)

        # var = var_map(reg, 3)

        # Pick those dots that fit well.
        # reg[var < 1] = 0
        kernel_e = np.ones((3, 3), np.uint8)
        reg = cv2.erode(reg, kernel_e, iterations=1)
        oct[oct<170] = 0

        kernel_d = np.ones((2, 2), np.uint8)
        reg = cv2.dilate(reg, kernel_d, iterations=1)
        oct = cv2.dilate(oct, kernel_d, iterations=1)
        # Set threshold for registered GFP image.
        (img_average, img_stddv) = cv2.meanStdDev(reg[:, :])
        # ret, reg_clear = cv2.threshold(reg, thre_l, 0, cv2.THRESH_TOZERO)
        # ret, reg_clear = cv2.threshold(reg_clear, thre_h, 255, cv2.THRESH_TOZERO_INV)
        reg_clear = cv2.resize(reg_clear, (435, 480), interpolation=cv2.INTER_AREA)
        # split channedls
        oct[:, :, 0] = 0
        oct[:, :, 1] = 0
        oct[:, :, 1][0:480, 138:573] = reg_clear[:, :]
        # oct[:, :, 1][62:492, 635:945] = cv2.medianBlur(reg_clear[:, :, 1], 11)
        oct[:, :, 2] = oct[:, :, 2]  # Give red channel less weight.
        cv2.imwrite(save_path, oct)

def process(args):
    OCT_Raw = args.Path_OCT
    LS_Raw = args.Path_LS
    Merged = args.Path_Merged
    ## Crop ".tiff" images from "OCT_RAW".
    os.mkdir('./OCT_Cropped')
    OCT_Cropped = './OCT_Cropped'
    crop(OCT_Raw, OCT_Cropped)
    ## Refresh LS images.
    os.mkdir('./MPE_Refreshed')
    MPE_Refreshed = './MPE_Refreshed'
    refresh(LS_Raw, MPE_Refreshed)
    ## Co-register the cropped wide-field image and refreshed hig-resolution image.
    os.mkdir('./MPE_Refreshed')
    Registered = './Registered'
    register(OCT_Cropped, MPE_Refreshed, Registered)
    ## Merge
    merge(OCT_Raw, Registered, Merged)

# Import parameters from command lines.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Denoising and co-registration")
    parser.add_argument('--Path_OCT', default='/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/OCT', help='Path for wide-field images')
    parser.add_argument('--Path_LS', default='/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE', help='Path for high-resolution images')
    parser.add_argument('--Path_Merged', default='/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/Merged', help='Path for processed merged images')
    args = parser.parse_args()
    process(args)


