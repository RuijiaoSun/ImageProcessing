import os
from os.path import join
import sys
import cv2
import ants
import time
from libtiff import TIFF
import numpy as np
from scipy import ndimage


## CROP
# path for original .tiff OCT images and cropped .png OCT images.
OCT_PATH_Ori = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/OCT'
OCT_PATH_Cropped = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/OCT_Cropped'

## REFRESH
# path for original .tiff LS images and refreshed .png LS images.
MPE_PATH_ori = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE'
MPE_PATH_Refreshed = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE_Refreshed'

## REGISTRATION
# OCT abd LS image paths for images to be registered. REG_TIF_PATH is registered LS images.
MPE_PATH = MPE_PATH_Refreshed
OCT_PATH = OCT_PATH_Cropped
REG_TIF_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/registered_MPE'

## MERGE
# Output path of "mergeChannel()".
MERGE_PNG_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/Merged_threshold5'

## For display. Not strictly needed.
REG_PNG_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/reigstered_PNG'
MERGE_TIFF_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/Merged_TIFF_[80,255]'

def var_f(img, win):
    mean = ndimage.uniform_filter(img, win)
    mean_sqr = ndimage.uniform_filter(img**2, win)
    var = mean_sqr - mean**2
    return var

def crop():
    # Images to be cropped named ")CT_PATH_ori".
    # OCT_PATH_Ori is the address of raw .tiff OCT images.
    for i in os.listdir(OCT_PATH_Ori):
        if i[-4:-1] == 'tif':
            tif = TIFF.open(os.path.join(OCT_PATH_Ori, i), 'r')
            img = tif.read_image()
            img_crop = img[0:480, 138:573]     # [y_range, x_range]
            img_interp = cv2.resize(img_crop, (2048, 2048), interpolation=cv2.INTER_LINEAR)
            new_path = os.path.join(OCT_PATH_Cropped, str(int(i[-8:-5]) - 1).zfill(3) + '.png')
            cv2.imwrite(new_path, img_interp)

def refresh():
    images = os.listdir(MPE_PATH_ori)
    for img_name in images:
        # LS images' name length: 20.
        # OCT images' name length: 16.
        if len(img_name) == 20:
            img_path = os.path.join(MPE_PATH_ori, img_name)
            img2 = os.path.join(MPE_PATH_Refreshed, img_name[-8: -5] + '.png')
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

OCT_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/OCT_Cropped'
MPE_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE_Refocused_3'
REG_TIF_PATH = "/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/REG_TIFF_PATH"
def register3D():
    start = time.time()

    # Register a good image pair as standrad.
    # The paths are for cropped .tiff OCT images & .tiff LS imgae.
    fix_path = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/OCT_Cropped/142.png'
    move_path = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/MPE_Refocused_3/142.png'
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

thre = 50
thre_h = 200

MERGE_PNG_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/Debackground_LS_3,2_oct_170_d09272021'
def merge(thre_l, thre_h):
    import matplotlib.pyplot as plt
    for i in os.listdir(OCT_PATH_Ori):
        reg_path = os.path.join(REG_TIF_PATH, 'registered' + str(int(i[-8:-5]) - 1).zfill(3) + '.tiff')
        oct_path = os.path.join(OCT_PATH_Ori, i)
        save_path = os.path.join(MERGE_PNG_PATH, 'merged' + i[8:11] + '.png')

        reg = cv2.imread(reg_path, 2)
        oct = cv2.imread(oct_path, 1)

        # var = var_f(reg, 5)

        # Pick those dots that fit well.
        # reg[var < 5*np.mean(var] = 0
        kernel_e = np.ones((3, 3), np.uint8)
        reg = cv2.erode(reg, kernel_e, iterations=1)
        oct[oct<170] = 0

        kernel_d = np.ones((2, 2), np.uint8)
        reg = cv2.dilate(reg, kernel_d, iterations=1)
        oct = cv2.dilate(oct, kernel_d, iterations=1)
        # Set threshold for registered GFP image.
        (img_average, img_stddv) = cv2.meanStdDev(reg[:, :])
        ret, reg_clear = cv2.threshold(reg, thre_l, 0, cv2.THRESH_TOZERO)
        ret, reg_clear = cv2.threshold(reg_clear, thre_h, 255, cv2.THRESH_TOZERO_INV)
        reg_clear = cv2.resize(reg_clear, (435, 480), interpolation=cv2.INTER_AREA)
        # split channedls
        oct[:, :, 0] = 0
        oct[:, :, 1] = 0
        oct[:, :, 1][0:480, 138:573] = reg_clear[:, :]
        # oct[:, :, 1][62:492, 635:945] = cv2.medianBlur(reg_clear[:, :, 1], 11)
        oct[:, :, 2] = oct[:, :, 2]  # Give red channel less weight.
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

def TifToPng():
    for i in os.listdir(REG_TIF_PATH):
        # change ".ti" to ".tif" if the image format is ".tiff".
        if i[-4:-1] == 'tif':
            tif = TIFF.open(os.path.join(REG_TIF_PATH, i), 'r')
            image = tif.read_image()
            # For registered images format change.
            new_path = os.path.join(REG_PNG_PATH, str(i[-8:-5]) + '.png')
            # For LS images format change.
            # new_path = os.path.join(REG_PNG_PATH, str(int(i[-8:-5]))+'.png')
            cv2.imwrite(new_path, image)

# MERGE_PNG_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/Embryo_Merged_e+d_LS+OCT'
# MERGE_TIFF_PATH = '/home/ruijiao/A_research/registration/embryo_DAPI/MK_3D/focus2'
def PngToTif():
    for i in os.listdir(MERGE_PNG_PATH):
        # change ".ti" to ".tif" if the image format is ".tiff".
        if i[-4:-1] == '.pn':
            image = cv2.imread(join(MERGE_PNG_PATH, i))
            cv2.imwrite(join(MERGE_TIFF_PATH, i[0:-4] + '.tiff'), image)

# Crop original images and save them with ".png" format. Keep local part including to-be-registered information.
# crop()
# Adjust the brightness and contrast of images.
# refresh()
# Register a standard image pair and apply the transformation field to all iamges.
# register3D()
# Merge original OCT images and registered images together.
merge(50, 200)
# PngToTif()

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

def mergeChannel(thre_l, thre_h):
    for i in os.listdir(REG_PNG_PATH):
        reg_path = os.path.join(REG_PNG_PATH, i)
        oct_path = os.path.join(OCT_PATH, str(int(i[0: -4])) + '.png')
        save_path = os.path.join(MERGE_PNG_PATH, 'merged' + i[0:-4] + '.png')

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
# Change the format of registered images.
# TifToPng()
# PngToTif()
# Read MPE and OCT images with .png format, denoise and merge them.
# mergeChannel(50, 200)