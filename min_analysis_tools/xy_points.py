# -*- coding: utf-8 -*-
"""
Series of tools for collections of points
-center-off mass inluding eccentricity and angle

Created on Mon Nov 22 20:38:19 2021
@author: jkerssemakers
"""
import math

import matplotlib.pyplot as plt
import numpy as np


def get_com_extra(xx, yy, zz, use_weights, demo=0):
    """
    Central angular moments of a collection of points (xx,yy,zz)
    'zz' can serve as weight of not
    from: http://en.wikipedia.org/wiki/Image_moment
    JacobKers MatLab 2012 --> Python 2021
    """
    # demo section starts here ------------------------
    if demo:
        use_weights = 1
        n = 200
        xx = np.random.rand(n, 1)
        yy = -3 * xx + 0.1 * (np.random.randn(n, 1) - 0.5)
        zz = np.random.rand(n, 1)
    # demo section stop  ------------------------
    if use_weights == 0:  # no weights
        zz = 0 * zz + 1
    # raw moments M_ij of points: Mij=sum(x^i*y^j*Ixy)
    M00 = np.sum(zz)
    M10 = np.sum(xx * zz)
    M11 = np.sum(yy * xx * zz)
    M01 = np.sum(yy * zz)
    M20 = np.sum((xx ** 2) * zz)
    M02 = np.sum((yy ** 2) * zz)
    # Centroid row and column
    xm = M10 / M00
    ym = M01 / M00
    #   Second order central moments
    mu_prime20 = M20 / M00 - xm ** 2
    mu_prime02 = M02 / M00 - ym ** 2
    mu_prime11 = M11 / M00 - xm * ym
    #   theta=0.5*atan(2*mu_prime11/(mu_prime20-mu_prime02))*180/pi;
    at_y = mu_prime20 - mu_prime02
    at_x = 2 * mu_prime11
    theta = 0.5 * 180 / math.pi * np.arctan2(at_x, at_y)

    # eigenvalues covariance matrix:
    lambda1 = (
        0.5 * (mu_prime20 + mu_prime02)
        + 0.5 * (4 * mu_prime11 ** 2 + (mu_prime20 - mu_prime02) ** 2) ** 0.5
    )
    lambda2 = (
        0.5 * (mu_prime20 + mu_prime02)
        - 0.5 * (4 * mu_prime11 ** 2 + (mu_prime20 - mu_prime02) ** 2) ** 0.5
    )
    # eccentricity
    if lambda1 != 0:
        ecc = (1 - lambda2 / lambda1) ** 0.5
    else:
        ecc = -1

    # demo section start ------------------------
    if demo:
        plt.plot(xx, yy, "ro")
        plt.show()
    # demo section stop  ------------------------
    return xm, ym, theta, ecc


def get_nearby(x0=1, y0=1, xx=1, yy=1, r0=1, demo=0):
    """
    Find selection of points near a given point.
    Normalized distance is also exported, for example to be used as weight.
    """
    # demo section start ------------------------
    if demo:
        r0 = 1
        x0 = 0.5
        y0 = 0
        use_weights = 1
        pts = 200
        xx = np.random.randn(pts, 1)
        yy = np.random.randn(pts, 1)
        points = np.concatenate((xx, yy), axis=1)
    # demo section stop  ------------------------
    rn = np.hypot(xx - x0, yy - y0) / r0
    ix = np.argwhere(rn < 1)
    xx_near = xx[ix[:, 0]]
    yy_near = yy[ix[:, 0]]
    rn_near = rn[ix[:, 0]]
    # demo section start ------------------------
    if demo:
        plt.plot(xx, yy, "o")
        plt.plot(xx_near, yy_near, "ro")
        plt.plot(x0, y0, "ko", markersize=10)
        plt.show()
    # demo section stop  ------------------------
    return xx_near, yy_near, rn_near


def smooth_lines(xx, yy, r0, demo=0):
    """
    Smooth a noise collection of xy points with local com.
    """
    # demo section start ------------------------
    if demo:
        pts = 200
        imsz = 200
        xx = np.linspace(10, imsz - 10, pts)
        yy = imsz / 2 * (1 + 0.25 * np.random.rand(pts)) + imsz / 3 * np.sin(
            xx / imsz * 2 * math.pi
        )
        r0 = imsz / 20
    # demo section stop  ------------------------
    Ls = len(xx)
    xx_s = 0 * xx
    yy_s = 0 * yy
    alpha = 0 * xx
    for ii in np.arange(Ls):
        x0 = xx[ii]
        y0 = yy[ii]
        xx_near, yy_near, rn_near = get_nearby(x0, y0, xx, yy, r0)
        # use 1-normalized radius as weight
        zz = 1 - rn_near  # weights
        xm, ym, alfa, ecc = get_com_extra(xx_near, yy_near, zz, use_weights=1)
        xx_s[ii] = xm
        yy_s[ii] = ym
        alpha[ii] = alfa
    # demo section start ------------------------
    if demo:
        plt.plot(xx, yy, "o", markersize=5)
        plt.plot(xx_s, yy_s, "ro", markersize=3)
        plt.show()
    # demo section stop  ------------------------
    return xx_s, yy_s, alpha


def get_within_image_limits(xx, yy, im, hs):
    """
    Small tool to crop coordinates.
    """
    rr, cc = np.shape(im)
    ix = np.nonzero(
        (xx > hs + 1) & (xx < cc - hs - 1) & (yy > hs + 1) & (yy < rr - hs - 1)
    )
    xc = xx[ix]
    yc = yy[ix]
    return xc, yc, ix
