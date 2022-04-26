# -*- coding: utf-8 -*-
"""
Series of tools for profile analysis
-center-off mass inluding eccentricity and angle

Created on Mon Nov 22 20:38:19 2021
@author: jkerssemakers
"""
import matplotlib.pyplot as plt
import numpy as np


def sub_unit_pos(prf=0, ixa=0, demo=0):
    """
    Get subpixel position by 'pts'-point parabolic fit around indices of local maxima of profile.
    """
    # demo section start ------------------------
    if demo:
        LP = 20
        temp_ax = np.arange(0, LP, 1)
        prf = 5 * np.exp(-((temp_ax - 19) ** 2) / 10**2)
        ixa = prf.argmax(axis=0)
    # demo section stop  ------------------------
    # find peak position
    LP = len(prf)
    LMX = len(ixa)
    xax = np.arange(0, LP, 1)

    pts = 5
    hsp = round((pts - 1) / 2)  # halfspan
    sub_ax = np.arange(-hsp, hsp + 1)  # local fit axis

    for ii in np.arange(0, LMX):
        ix = ixa[ii]
        # edge correction
        shft = hsp * (ix <= hsp - 1) - hsp * (ix >= LP - hsp)
        sub_ax = sub_ax + shft
        sub_prf = prf[ix + shft - hsp : ix + shft + hsp + 1]

        # fit to find the subpixel correction x0
        prms = np.polyfit(sub_ax, sub_prf, 2)
        if prms[0] != 0:
            x_0 = -prms[1] / (2 * prms[0])
        else:
            x_0 = 0
        x_p = xax[ix] + x_0
        prf_p = np.polyval(prms, x_0)
        # demo section start ------------------------
        if demo:
            prf_fit = np.polyval(prms, sub_ax)
            plt.plot(xax, prf, "bo-", markersize=4)
            plt.plot(sub_ax + xax[ix], prf_fit, "r-", markersize=4)
            plt.plot(x_p, prf_p, "ro", markersize=6)
            plt.show()
        # demo section stop  ------------------------
    return x_p, prf_p


def get_maxima(prf=0, N_max=1):
    """
    Get (up to a) pre-set number of maxima, in descending order of peak value.
    """
    if N_max == 1:
        ix = prf.argmax(axis=0)
        max_ix = [ix]
    if N_max > 1:
        higher_than_left = prf[1:-1] > prf[0 : len(prf) - 2]
        higher_than_right = prf[1:-1] > prf[2 : len(prf)]
        max_ix = np.ndarray.flatten(
            np.argwhere(higher_than_left & higher_than_right) + 1
        )
        if len(max_ix) >= 1:
            max_val = prf[max_ix]
            # sort indices by peak size:
            iix_srt = np.argsort(max_val)
            max_ix_srt = max_ix[iix_srt]
            # pick first N, if that many:
            Nc = np.min([N_max, len(max_ix)])
            max_ix = max_ix_srt[::-1][:Nc]
        else:
            max_ix = [np.nan]
    return max_ix
