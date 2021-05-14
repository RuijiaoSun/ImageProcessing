# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:00:10 2021


Thid code is base on ants library.

It can be used for registration2D, registration3D, and postprocessing for registered images.

@author: ruijiao
"""
import os
from os.path import join
import cv2
# import ants
import time
from libtiff import TIFF

# path for original OCT images.
OCT_PATH_Ori = '/home/ruijiao/A_research/MPE/embryo_center/OCT_TIFF'
OCT_PATH_Cropped = '/home/ruijiao/A_research/MPE/embryo_center/OCT_PNG'

# Paths for images to be registered. After cropped.
MPE_PATH = '/home/ruijiao/A_research/MPE/embryo_center/LS_PNG'
OCT_PATH = '/home/ruijiao/A_research/MPE/embryo_center/OCT_PNG'

# Output path of "Register3D()".
# Input path of "formatChange()".
# REG_TIF_PATH = '../embryo_center/registered_TIFF'    # run code from "/home/ruijiao/A_research/MPE/code"
REG_TIF_PATH = '/home/ruijiao/A_research/MPE/embryo_center/registered_TIFF'
# REG_TIF_PATH = '/home/ruijiao/A_research/MPE/embryo_center/LS_TIFF'

# Output path of "formatChange()".
# Input path of "mergeChannel()". 
REG_PNG_PATH = '/home/ruijiao/A_research/MPE/embryo_center/registered_PNG'
# REG_PNG_PATH = '/home/ruijiao/A_research/MPE/embryo_center/LS_PNG'

# Output path of "mergeChannel()".
MERGE_PNG_PATH = '/home/ruijiao/A_research/MPE/registered/registered_3D_right_1.3_1.2'
MERGE_TIFF_PATH = '/home/ruijiao/A_research/MPE/registered/registered_3D_right_1.3_1.2_TIFF'


def register2D():
    start = time.time()

    # Save path for registered image.
    save_path_tif = '/home/ruijiao/registration/1PE+OCT_right_Jan31/test.tif'
    save_path_png = '/home/ruijiao/registration/1PE+OCT_right_Jan31/test.png'
    # Path for the two to-be-registered images.
    fix_path = '/home/ruijiao/registration/1PE+OCT_right_Jan31/Interp2_images/resized_frame000287.png'
    move_path = '/home/ruijiao/registration/1PE+OCT_right_Jan31/MPE_PNG/287.png'

    # Read and register.
    fix_ants = ants.image_read(fix_path)
    move_ants = ants.image_read(move_path)
    outs = ants.registration(fix_ants, move_ants, type_of_transform='SyN')
    std_img = ants.apply_transforms(fix_ants, move_ants, transformlist=outs['fwdtransforms'], interpolator='linear')
    ants.image_write(std_img, save_path_tif)
    image = cv2.imread(save_path_tif, 2)
    cv2.imwrite(save_path_png, image)
    stop = time.time()
    print("Collapsed time: " + str(stop - start) + " s.")


def register3D():
    start = time.time()

    # Register a good image pair as standrad.
    fix_path = '/home/ruijiao/A_research/MPE/embryo_center/OCT_PNG/26.png'
    move_path = '/home/ruijiao/A_research/MPE/embryo_center/LS_PNG/26.png'
    # save_path = join(TIFF_PATH, 'std.tif')
    fix_ants = ants.image_read(fix_path)
    move_ants = ants.image_read(move_path)
    outs = ants.registration(fix_ants, move_ants, type_of_transform='SyN')
    # std_img = ants.apply_transforms(fix_ants, move_ants, transformlist = outs['fwdtransforms'], interpolator = 'linear')
    # ants.image_write(std_img, save_path)

    stop1 = time.time()
    print("registration time: " + str(stop1 - start) + " s.")
    for i_MPE in os.listdir(MPE_PATH):
        # Set images path.
        img_Name = i_MPE
        i_OCT = img_Name
        save_path = join(REG_TIF_PATH, 'registered' + img_Name[0:-4] + '.tiff')

        move_img = ants.image_read(join(MPE_PATH, i_MPE))
        fix_img = ants.image_read(join(OCT_PATH, i_OCT))

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


def TifToPng():
    for i in os.listdir(REG_TIF_PATH):
        # change ".ti" to ".tif" if the image format is ".tiff".
        if i[-4:-1] == 'tif':
            tif = TIFF.open(os.path.join(REG_TIF_PATH, i), 'r')
            image = tif.read_image()
            # For registered images format change.
            new_path = os.path.join(REG_PNG_PATH, str(i[10:-5]) + '.png')
            # For LS images format change.
            # new_path = os.path.join(REG_PNG_PATH, str(int(i[-8:-5]))+'.png')
            cv2.imwrite(new_path, image)


def PngToTif():
    for i in os.listdir(MERGE_PNG_PATH):
        # change ".ti" to ".tif" if the image format is ".tiff".
        if i[-4:-1] == '.pn':
            image = cv2.imread(join(MERGE_PNG_PATH, i))
            cv2.imwrite(join(MERGE_TIFF_PATH, i[0:-4] + '.tiff'), image)


# thre_l is the lowest pixel value to be kept.
# thre_h is the highest pixel value to be kept.
def mergeChannel(thre_l, thre_h):
    for i in os.listdir(REG_PNG_PATH):
        reg_path = os.path.join(REG_PNG_PATH, i)
        oct_path = os.path.join(OCT_PATH, str(int(i[0: -4])) + '.png')
        save_path = os.path.join(MERGE_PATH, 'merged' + i[0:-4] + '.png')

        reg = cv2.imread(reg_path)
        oct = cv2.imread(oct_path)

        # Set threshold for registered GFP image.
        (img_average, img_stddv) = cv2.meanStdDev(reg[:, :, 1])
        # thre_l = int(img_average * 1.4)
        # thre_h = int(img_average * 2.5)  # Set threshold changing by brightness of images.
        ret, reg_clear = cv2.threshold(reg, thre_l, 0, cv2.THRESH_TOZERO)
        ret, reg_clear = cv2.threshold(reg_clear, thre_h, 255, cv2.THRESH_TOZERO_INV)

        # reg_clear_adap = cv2.adaptiveThreshold(reg_clear[:, :, 0], 0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_TOZERO, 11, 0)
        # split channedls
        reg[:, :, 0] = 0
        reg[:, :, 1] = reg_clear[:, :, 1]
        reg[:, :, 2] = oct[:, :, 0]  # Give red channel less weight.

        cv2.imwrite(save_path, reg)


def crop():
    # Images to be cropped named "PCT_PATH_ori".
    for i in os.listdir(OCT_PATH_Ori):
        if i[-4:-1] == 'tif':
            tif = TIFF.open(os.path.join(OCT_PATH_Ori, i), 'r')
            img = tif.read_image()
            img_crop = img[300:700, 200:600]
            img_interp = cv2.resize(img_crop, (2048, 2048), interpolation=cv2.INTER_LINEAR)
            new_path = os.path.join(OCT_PATH_Cropped, str(int(i[-8:-5]) - 1) + '.png')
            cv2.imwrite(new_path, img_interp)


def merge(thre_l, thre_h):
    for i in os.listdir(OCT_PATH_Ori):
        reg_path = os.path.join(REG_TIF_PATH, 'registered' + str(int(i[8:11]) - 1) + '.tiff')
        oct_path = os.path.join(OCT_PATH_Ori, i)
        save_path = os.path.join(MERGE_PATH, 'merged' + i[8:11] + '.png')

        reg = cv2.imread(reg_path, 2)
        oct = cv2.imread(oct_path, 1)

        # Set threshold for registered GFP image.
        (img_average, img_stddv) = cv2.meanStdDev(reg[:, :])
        ret, reg_clear = cv2.threshold(reg, thre_l, 0, cv2.THRESH_TOZERO)
        ret, reg_clear = cv2.threshold(reg_clear, thre_h, 255, cv2.THRESH_TOZERO_INV)
        reg_clear = cv2.resize(reg_clear, (400, 400), interpolation=cv2.INTER_AREA)
        # split channedls
        oct[:, :, 0] = 0
        oct[:, :, 1] = 0
        oct[:, :, 1][300:700, 200:600] = reg_clear[:, :]
        # oct[:, :, 1][62:492, 635:945] = cv2.medianBlur(reg_clear[:, :, 1], 11)
        oct[:, :, 2] = oct[:, :, 2] * 1.2  # Give red channel less weight.
        # oct[:, :, 2] = cv2.medianBlur(oct[:, :, 2], 3)

        # #Let OCT images be larger and keep the             size of MPE images.
        #
        # oct_new = cv2.resize(oct, (int(1000*2048/500), int(1001*2048/400)), interpolation=cv2.INTER_AREA)
        # oct_new[:, :, 0] = 0
        # oct_new[:, :, 1] = 0
        # oct_new[:, :, 1][int(300*2048/400):int(700*2048/400), int(200*2048/500):int(700*2048/500)] = reg_clear[:, :]
        # oct_new[:, :, 2] = oct_new[:, :, 2]/1.2     #Give red channel less weight.d
        #
        cv2.imwrite(save_path, oct)


# Find an image pair whose MPE image and OCT image can combine well.
# register2D()

# Register a standard image pair and apply the transformation field to all iamges.
# register3D()

# Change the format of registered images.
# TifToPng()
PngToTif()

# Read MPE and OCT images with .png format, denoise and merge them.
# mergeChannel(50, 200)

# Crop original images and save them with ".png" format. Keep local part including to-be-registered information.
# crop()

# Merge original OCT images and registered images together.
# merge(50, 200)