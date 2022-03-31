# -*- coding: utf-8 -*-
"""
MinDE pattern analysis: loading or generating data
Created on Nov 23 2021
@author: jkerssemakers
"""
from math import ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def load_stack(stack_path, size, kernel_size=35, demo=0):
    """
    This loads function loads an image stack from a given directory.
    Each frame is downsized to a given 'size' (default 512), applying
    to the shorter axis. Set 'size' to None to skip.
    Each frames is smoothened using a kernel defined by 'kernel_size'.
    Set 'kernel_size' to None to skip.
    """
    MinDE_st_full = io.imread(stack_path)
    ff, rr, cc = np.shape(MinDE_st_full)
    if size is not None and rr > size and cc > size:
        downsize = True
    else:
        downsize = False
    if kernel_size is not None:
        smooth_image = True
    else:
        smooth_image = False

    if downsize is True or smooth_image is True:
        # new image dimensions for downsizing
        if downsize:
            dimensions = (ceil(size * cc / min(cc, rr)), ceil(size * rr / min(cc, rr)))
        # define kernel for smoothening
        if smooth_image:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (
                kernel_size ** 2
            )
        # work all frames
        for ii in range(ff):
            frame = MinDE_st_full[ii, :, :]
            # smooth image
            if smooth_image:
                frame_filt = cv2.filter2D(frame, -1, kernel)
            else:
                frame_filt = frame
            # downsize image
            if downsize:
                frame_ip = cv2.resize(
                    frame_filt, dimensions, interpolation=cv2.INTER_NEAREST
                )
            else:
                frame_ip = frame_filt
            # save images (append to stack)
            if ii == 0:
                MinDE_st = frame_ip
            else:
                MinDE_st = np.dstack((MinDE_st, frame_ip))
        # correctly rearrange axes
        MinDE_st = np.moveaxis(MinDE_st, 2, 0)

    else:  # no action required
        MinDE_st = MinDE_st_full

    # demo section start ------------------------
    if demo > 0:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(MinDE_st[0, :, :])
        ax.set_title("pattern")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        return MinDE_st, fig, ax
    # demo section stop  ------------------------
    else:
        return MinDE_st


def generate_pattern(
    lambda_t=15,  # wavelength
    lambda_x=5,  # oscillation period
    size=512,  # image size (square shape)
    N_frames=20,  # number of frames
    kernel_size=35,  # kernel size for smoothening
    amplitude=255,  # amplitude
    SN_ratio=0,  # signal/noise ratio
    demo=0,  # demo plot or not
):
    """
    This generates a simple spiral image stack for quick testing.
    Settings for the spiral period, degree of turning etc.
    """
    x, y = np.linspace(-10, 10, size), np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    aa0 = np.arctan2(Y, X)
    rr = np.hypot(X, Y)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    for ii in range(N_frames):
        plane = ii
        aa = 6 * np.pi * ((plane - 1) / (2 * lambda_t))  # sets turning
        frame = amplitude * (
            1 - ((np.sin(((rr - lambda_x * (aa0 + aa)) / lambda_x)))) ** 2
        )
        frame = cv2.filter2D(frame, -1, kernel)
        noise = SN_ratio * amplitude * np.random.randn(size, size)
        if ii == 0:
            MinDE_st = frame + noise
        else:
            MinDE_st = np.dstack((MinDE_st, frame + noise))
    MinDE_st = np.moveaxis(MinDE_st, 2, 0)
    # demo section start ------------------------
    if demo > 0:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(MinDE_st[0, :, :])
        ax.set_title("pattern")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        return MinDE_st, fig, ax
    # demo section stop  ------------------------
    else:
        return MinDE_st
