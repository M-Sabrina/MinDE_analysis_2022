# -*- coding: utf-8 -*-
"""
outlier detection in 1D data
@author: jkerssemakers, 2022
"""
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm

from min_analysis_tools.clean_tools_1D import outlier_flag


def get_stack_dirt_and_light(tif_in, exclude_im, bleach_curve, demo=False):
    """
    Work a stack to get
    1) an average illumination image and
    2) a 'dirt' image of fixed components.
    Prior to this step, bleach curve correction and exclusion zones
    are applied from earlier processing steps.
    """
    firstframe = tif_in[0, :, :]
    object_im = 0 * firstframe
    illum_im = 0 * firstframe
    # get a value for padding:
    usevals = firstframe[exclude_im == 0]
    padval = np.median(usevals)
    st_size = tif_in.shape
    framecounter = 0
    # work stack:
    for ii in range(st_size[0]):
        # fill dark image
        thisframe1 = deepcopy(tif_in[ii, :, :])
        bleachcorrectfactor = bleach_curve[ii]
        thisframe1 = thisframe1 / bleachcorrectfactor
        sumval = np.sum(thisframe1)
        if sumval >= 0:
            framecounter = framecounter + 1
            # pad exclusion zones:
            thisframe2 = deepcopy(thisframe1)
            thisframe2[np.nonzero(exclude_im == 1)] = padval
            # remove outliers:
            thisframe2_arr = np.squeeze(np.asarray(thisframe2))
            inliers, outliers, flags = outlier_flag(
                thisframe2_arr,
                tolerance=3,
                sig_change=0.7,
                how=0,
                demo=False,
                test=False,
            )
            outlier_im = 1 - np.reshape(flags, thisframe1.shape, order="C")
            thisframe3 = deepcopy(thisframe2)
            thisframe3[outlier_im == 1] = padval

            # find illumination component:
            current_dirt_im, current_illum_im = split_image(thisframe3)

            object_im = object_im + current_dirt_im
            illum_im = illum_im + current_illum_im

    object_im = object_im / framecounter
    illum_im = illum_im / framecounter
    # kernel for smoothing :
    kernel_size = int(st_size[1] / 4)
    general_kernel = np.full(
        (kernel_size, kernel_size), 1 / kernel_size**2, dtype=np.float32
    )
    illum_im = cv2.filter2D(illum_im, -1, general_kernel)
    illum_im = illum_im / np.max(illum_im)

    if demo:
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].imshow(thisframe1)
        axs[0, 0].set_title("image")
        axs[0, 1].imshow(thisframe2)
        axs[0, 1].set_title("im_excluded")
        axs[0, 2].imshow(thisframe3)
        axs[0, 2].set_title("im_outliers removed")
        axs[1, 0].imshow(object_im)
        axs[1, 0].set_title("stack:objects")
        axs[1, 1].imshow(illum_im)
        axs[1, 1].set_title("stack:light")

        fig.tight_layout()
        plt.show()

    return object_im, illum_im


def split_image(image_in, demo=False):
    """
    Split an image in a smooth and an objects part.
    Since we use median, we perform a four-step iteration
    for improved determination of the background field.
    """
    im_size = image_in.shape
    object_im = deepcopy(image_in)
    illum_im = np.zeros_like(image_in)

    for _ in range(4):
        refcurve_hor = np.median(object_im, axis=0)
        refcurve_ver = np.median(object_im, axis=1)
        ref_hor_im = npm.repmat(refcurve_hor, im_size[0], 1)
        ref_ver_im = np.transpose(npm.repmat(refcurve_ver, im_size[1], 1))
        illum_im_partial = (ref_hor_im + ref_ver_im) / 2
        object_im = object_im - illum_im_partial
        illum_im = illum_im + illum_im_partial

    if demo:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image_in)
        axs[0].set_title("image")
        axs[1].imshow(object_im)
        axs[1].set_title("objects")
        axs[2].imshow(illum_im)
        axs[2].set_title("light")
        plt.show()

    return object_im, illum_im


def clean_stack(tif_in, exclude_im, dirt_im, illum_im, bleach_curve, demo=False):
    """
    1) apply bleach correction
    2) use exlusion zones
    3) remove static features
    4) illumination correction
    5) remove last outliers
    6) final smooth
    """
    tif_cln = 0 * tif_in
    firstframe = tif_in[0, :, :]
    # get a value for padding:
    usevals = firstframe[np.nonzero(exclude_im == 0)]
    padval = np.median(usevals)
    st_size = tif_in.shape
    for ii in range(st_size[0]):
        thisframe1 = deepcopy(tif_in[ii, :, :])
        # bleach correct:
        bleachcorrectfactor = bleach_curve[ii]
        thisframe1 = thisframe1 / bleachcorrectfactor
        sumval = np.sum(thisframe1)
        if sumval == 0:  # fill dark image
            tif_cln[ii, :, :] = thisframe1 + padval
        else:
            # pad exclusion zones:
            thisframe2 = deepcopy(thisframe1)
            thisframe2[np.nonzero(exclude_im == 1)] = padval
            # remove dirt:
            thisframe2 = thisframe2 - dirt_im
            # correct for light:
            thisframe2 = thisframe2 / illum_im
            # remove remaining outliers:
            thisframe2_arr = np.squeeze(np.asarray(thisframe2))
            inliers, outliers, flags = outlier_flag(
                thisframe2_arr,
                tolerance=2.5,
                sig_change=0.7,
                how=0,
                demo=False,
                test=False,
            )
            outlier_im = 1 - np.reshape(flags, thisframe1.shape, order="C")
            thisframe3 = deepcopy(thisframe2)
            thisframe3[np.nonzero(outlier_im == 1)] = padval
            thisframe4 = thisframe3 - np.min(thisframe3)
            # smooth:
            sq = 5
            squ = np.full((sq, sq), 1 / (sq**2), np.float32)
            thisframe4 = cv2.filter2D(thisframe4, -1, squ)
            # add:
            tif_cln[ii, :, :] = thisframe4

    if demo:
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].imshow(thisframe1)
        axs[0, 0].set_title("image")
        axs[0, 1].imshow(thisframe2)
        axs[0, 1].set_title("im_excluded")
        axs[0, 2].imshow(thisframe3)
        axs[0, 2].set_title("im_outliers removed")
        axs[1, 0].imshow(dirt_im)
        axs[1, 0].set_title("stack:objects")
        axs[1, 1].imshow(illum_im)
        axs[1, 1].set_title("stack:light")
        axs[1, 2].imshow(thisframe4)
        axs[1, 2].set_title("corrected")
        plt.show()

    return np.array(tif_cln)
