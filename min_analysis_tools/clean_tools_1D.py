# -*- coding: utf-8 -*-
"""
outlier detection in 1D data
@author: jkerssemakers, 2022
"""
import matplotlib.pyplot as plt
import numpy as np


def outlier_flag(data=0, tolerance=2.5, sig_change=0.7, how=1, demo=False, test=False):
    """
    An iterative tool to separate a distribution from its outliers.
    An initial estimate of average 'mu' and standard deviation 'sigma' is used to identify outliers.
    These are removed and [mu,sigma] is re-determnined after which the sequence is repeated
    until sigma does not change much anymore
    Input:
        data: single array of values
        tolerance: outliers are points more than [tolerance] standard deviations away from the average.
        sigchange: iteration stops if sigma is changed less than a fraction  'sigchange'.
        how
        show
        demo
    Output:
        1) array of outliers
        2) array of inliers
        3) flags: binary trace indicating which points where outliers in the original data.
    Jacob Kers '2022
    """
    # test section start ------------------------
    if test:
        # build trace with 2 distributions
        N_pts = 2000
        s1 = 10
        u1 = 0
        N_otl = 200
        s2 = 10
        u2 = 100
        data = s1 * np.random.randn(N_pts, 1) + u1
        for _ in range(N_otl):
            randii = int((N_pts - 1) * np.random.rand(1, 1))
            data[randii] = s2 * np.random.randn(1, 1) + u2
    # test section stop  ------------------------

    sig_ratio = 0
    sigma_nw = 1e20
    flags = np.full_like(data, 1)
    while sig_ratio < sig_change:
        sigma_old = sigma_nw
        inliers = data[flags == 1]
        outliers = data[flags == 0]
        av = np.median(inliers)
        sigma_nw = np.std(inliers)
        sig_ratio = sigma_nw / sigma_old

        if how == 1:
            flags = (data - av) < tolerance * sigma_nw
        elif how == 0:
            flags = abs(data - av) < tolerance * sigma_nw
        elif how == -1:
            flags = (data - av) > -tolerance * sigma_nw

        if demo:
            lo = np.min(inliers)
            hi = np.max(inliers)
            bins = np.linspace(lo, hi, 40)
            fig2, ax2 = plt.subplots(1, 1)
            ax2.hist(inliers, bins, histtype="bar")
            plt.show()

    return inliers, outliers, flags
