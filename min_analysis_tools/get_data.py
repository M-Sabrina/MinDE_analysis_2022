# -*- coding: utf-8 -*-
"""
MinDE pattern analysis: loading or generating data
Created on Nov 23 2021
@author: jkerssemakers
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
