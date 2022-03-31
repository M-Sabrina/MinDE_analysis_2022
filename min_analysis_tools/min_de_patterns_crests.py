# -*- coding: utf-8 -*-
"""
Function specific for MinDE pattern analysis.
Created on Wed Oct 27 11:42:23 2021
@author: jkerssemakers
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from skimage import morphology as mp

from min_analysis_tools import peak_profile, xy_points


def compare_crestmaps(map1, map2, sampling_density, look_ahead, demo=0):
    """
    Find peak locations per crest cross sections, collected in 2D profile maps).
    Compare these for two points in time.
    """
    rr, cc = np.shape(map1)
    peak_shift = np.zeros((rr))

    # test interrupt -demo=3 or higher:shelved, 2:activate):
    if demo == 2:
        plt.close("all")
        fig, ((ax1, ax2)) = plt.subplots(1, 2)
        ax1.imshow(np.transpose(map1), aspect="auto")
        ax1.set_title("frame 1")  # MinD
        ax2.imshow(np.transpose(map2), aspect="auto")
        ax2.set_title("frame 2")  # MinE
        fig.show()
        breakpoint()  # click-to-code help

    for ii in np.arange(rr):
        prf1 = map1[ii, :]
        prf2 = map2[ii, :]

        # refine peak relations---------------------------------
        # first profile: get most centered maximum of many:
        iix1 = peak_profile.get_maxima(prf1, 5)
        if len(iix1) > 1:
            mid_ii = len(prf1) / 2
            df = abs(iix1 - mid_ii)
            ix1 = iix1[df.argmin(axis=0)]
        elif len(iix1) == 1:
            ix1 = iix1[0]
        elif len(iix1) == 0:
            ix1 = np.nan
        # second profile find related peak (forward or backward):
        if math.isnan(ix1) == False:
            iix2 = peak_profile.get_maxima(prf2, 5)
            # sort by index, irrespective of size:
            iix2 = sorted(iix2)
            if look_ahead == 1:  # a: first forward from 1
                ix2_i = np.ndarray.flatten(np.argwhere(iix2 >= ix1))
                if len(ix2_i) > 0:
                    ix2_i0 = ix2_i[0]
                    ix2 = iix2[ix2_i0]
                else:
                    ix2 = np.nan
            elif look_ahead == -1:  # b: first backward from 1
                ix2_i = np.ndarray.flatten(np.argwhere(iix2 <= ix1))
                if len(ix2_i) > 0:
                    ix2_i0 = ix2_i[-1]
                    ix2 = iix2[ix2_i0]
                else:
                    ix2 = np.nan
        else:
            ix2 = np.nan
        # use two chosen maxima to determine shift:
        badpeaks = math.isnan(ix1 * ix2)
        if badpeaks == False:
            x1, y1 = peak_profile.sub_unit_pos(prf1, [ix1])
            x2, y2 = peak_profile.sub_unit_pos(prf2, [ix2])
            peak_shift[ii] = (x2 - x1) * sampling_density
        else:
            peak_shift[ii] = np.nan

        # test interrupt -demo=3 or higher:shelved, 2:activate):
        if demo == 3 and badpeaks == False:
            plt.close("all")
            plt.plot(prf1, "ro-", markersize=3)
            plt.plot(prf2, "bo-", markersize=3)
            plt.plot(x1, y1, "ko", markersize=6)
            plt.plot(x2, y2, "ko", markersize=6)
            plt.title("cross-section shift")
            plt.xlabel("distance (pixels)")
            plt.ylabel("fluorescence, a.u.")
            plt.show()
            breakpoint()  # click-to-code help

    return peak_shift


def get_rise_or_fall(U, V, Im, demo=0):
    """
    Get increase or decrease of intensity in flow direction: This finds us
    the front and the wake regions of each wave.
    """
    rr, cc = np.shape(Im)
    ax_x, ax_y = np.linspace(1, cc, cc), np.linspace(1, rr, rr)
    XX, YY = np.meshgrid(ax_x, ax_y)
    Velo_mag = np.hypot(U, V)
    nU = U / Velo_mag
    nV = V / Velo_mag
    lookahead = 3
    # indices of nearby pixels, small span
    XX_next = np.round(XX + lookahead * nU)
    YY_next = np.round(YY + lookahead * nV)
    # interpolate
    Im_next = map_coordinates(
        Im, [YY_next.ravel(), XX_next.ravel()], order=3, mode="constant"
    ).reshape(Im.shape)
    # wavesign = np.sign(Im_nxt-Im)
    wavesign = Im_next < Im

    # test interrupt -demo=3:shelved, 2:activate):
    if demo == 2:
        plt.close("all")
        plt.figure()
        plt.imshow(wavesign)
        plt.title("front and wakes areas")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.show()
        breakpoint()  # click-to-code help
    return wavesign


def get_crests(wavesign_im, im, halfspan=5, demo=0):
    """
    Crests are the lines of pixels between the rise and the fall of a wave.
    Smooth them, clean them and get their local wave direction.
    """
    # obtain binary image containing only white crest pixels
    crests_im = (mp.binary_dilation(wavesign_im) ^ wavesign_im) & (im > np.mean(im))
    # transform to xy coordinates, smooth these
    crests_xy = np.argwhere(crests_im > 0)
    cr_xx = crests_xy[:, 1]
    cr_yy = crests_xy[:, 0]
    cr_xx_s, cr_yy_s, alpha_long = xy_points.smooth_lines(
        1.0 * cr_xx, 1.0 * cr_yy, r0=20
    )
    # remove all out-of-bounds coordinates, note the margin for later use
    crests_x, crests_y, ix = xy_points.get_within_image_limits(
        cr_xx_s, cr_yy_s, im, halfspan
    )
    # initialize a wavevector
    forward_wavevector_x = 0 * crests_x
    forward_wavevector_y = 0 * crests_y
    alpha_long = alpha_long[ix]
    # make a wave vector 'fwv' pointing to the front of the wave
    alpha_perp = alpha_long + 90
    # the angle is ambiguous and may point to either front or wake, we try:
    tentative_direction_x = halfspan * np.cos(alpha_perp / 180 * math.pi)
    tentative_direction_y = halfspan * np.sin(alpha_perp / 180 * math.pi)

    # try out in which direction the perp vector points
    try_fwv_x = np.rint(crests_x + tentative_direction_x)
    try_fwv_x = try_fwv_x.astype(int)
    try_fwv_y = np.rint(crests_y + tentative_direction_y)
    try_fwv_y = try_fwv_y.astype(int)
    # the local value tells us if we are in front or wake area
    vsign = wavesign_im[try_fwv_y, try_fwv_x]
    # now, pick the correct vector element-wise
    fwv_x = 0.0 * try_fwv_x
    fwv_y = 0.0 * try_fwv_y
    LV = len(vsign)
    for ii in np.arange(LV):
        if vsign[ii]:
            dir_cor = 1
        else:
            dir_cor = -1
        fwv_x[ii] = crests_x[ii] + dir_cor * tentative_direction_x[ii]
        fwv_y[ii] = crests_y[ii] + dir_cor * tentative_direction_y[ii]
        # vector normalized to unit pixel length
        td_mag = np.hypot(tentative_direction_x[ii], tentative_direction_y[ii])
        forward_wavevector_x[ii] = dir_cor * tentative_direction_x[ii] / td_mag
        forward_wavevector_y[ii] = dir_cor * tentative_direction_y[ii] / td_mag

    # test interrupt -demo=3:shelved, 2:activate):
    if demo == 2:
        plt.close("all")
        # plotting a corner: images
        zm = 300
        Le = len(crests_x)
        span = 10
        cx1 = crests_x
        cy1 = crests_y
        cx2 = cx1 + span * forward_wavevector_x
        cy2 = cy1 + span * forward_wavevector_y
        zm_ix = np.argwhere((cx1 < zm) & (cy1 < zm))
        fig, ax = plt.subplots()
        plt.imshow(im[0 : zm - 1, 0 : zm - 1])
        plt.plot(cx1[zm_ix], cy1[zm_ix], "ro", markersize=3)
        plt.plot(cx2[zm_ix], cy2[zm_ix], "bo", markersize=1)
        ax.set_box_aspect(1)
        plt.title("crest propagation")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.show()
        breakpoint()  # click-to-code help

    return crests_x, crests_y, forward_wavevector_x, forward_wavevector_y


def adjust_stack_orientation(stack):
    # rotate MinDE array to match image directionality
    ff, rr, cc = np.shape(stack)
    stack_rot = np.empty([ff, cc, rr])
    for f in range(ff):
        image = stack[f, :, :]
        image = np.rot90(image)
        image = np.flipud(image)
        image = np.fliplr(image)
        stack_rot[f, :, :] = image

    return stack_rot


def sample_crests(im1, cr_xx, cr_yy, wv_xx, wv_yy, halfspan, sampling_density, demo=0):
    """
    Build and use a sampling grid over the wavecrests and obtain a sampling map.
    cr_xx,yy: crest points; vv_xx,yy: forward vectors. Normalized to pixel units.
    """

    # build a grid from this end use this grid for interpolation;
    stepvector = np.arange(-halfspan, halfspan, sampling_density)
    LV = len(stepvector)
    # define x- and y- coordinates of all crest points
    xxgrid = np.transpose(np.tile(cr_xx, (LV, 1))) + np.outer(wv_xx, stepvector)
    yygrid = np.transpose(np.tile(cr_yy, (LV, 1))) + np.outer(wv_yy, stepvector)
    profilemap = map_coordinates(
        im1, [yygrid.ravel(), xxgrid.ravel()], order=3, mode="constant"
    ).reshape(np.shape(xxgrid))

    # test interrupt -demo=3:shelved, 2:activate):
    if demo == 2:
        plt.close("all")
        # show a zoom-in of the result
        zm_lo = 100
        zm_hi = 200
        rr, cc = np.shape(im1)
        zm_ix1 = np.argwhere(
            (cr_xx > zm_lo) & (cr_yy > zm_lo) & (cr_xx < zm_hi) & (cr_yy < zm_hi)
        )
        zm_ix2 = np.argwhere(
            (xxgrid.ravel() > zm_lo)
            & (yygrid.ravel() > zm_lo)
            & (xxgrid.ravel() < zm_hi)
            & (yygrid.ravel() < zm_hi)
        )
        fig, ((ax1, ax2)) = plt.subplots(1, 2)
        fig.tight_layout()
        ax1.imshow(im1[zm_lo:zm_hi, zm_lo:zm_hi])
        ax1.plot(
            xxgrid.ravel()[zm_ix2] - zm_lo,
            yygrid.ravel()[zm_ix2] - zm_lo,
            "ko",
            markersize=1,
        )
        ax1.plot(cr_xx[zm_ix1] - zm_lo, cr_yy[zm_ix1] - zm_lo, "ro", markersize=5)
        ax1.set_title("mapping overlay")
        ax1.set_xlabel("x (pixels)")
        ax1.set_ylabel("y (pixels)")
        ax2.imshow(profilemap.T[:, zm_ix1], aspect="auto")
        ax2.set_title("sampling 0")
        ax2.set_xlabel("crest index")
        ax2.set_ylabel("position (pixels)")
        fig.show()
        breakpoint()  # click-to-code help

    return profilemap, xxgrid, yygrid
