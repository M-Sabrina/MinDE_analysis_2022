import numpy as np
from numpy import matlib


def clean_kymograph(reslice):
    """
    Prepares a kymograph for autocorrelation analysis.
    Input: slice as 2D numpy array of dimension (xx,tt) or (yy, tt).
    Output: cleaned slice as 2D numpy array.
    """
    xx, tt = np.shape(reslice)

    # equalize x axis
    mn_prf_x = np.mean(reslice, axis=1)
    stripes_along_x = np.transpose(matlib.repmat(mn_prf_x, tt, 1))
    reslice = reslice - stripes_along_x

    # flatten along t axis
    md_prf_t = np.median(reslice, axis=0)
    md_t = np.median(md_prf_t)
    st_t = np.std(md_prf_t)
    axz = np.arange(tt)
    lo = md_t - 2 * st_t
    hi = md_t + 2 * st_t
    fitsel = np.argwhere((lo < md_prf_t) & (md_prf_t < hi))
    sub_prf = md_prf_t[fitsel]
    sub_axz = axz[fitsel]
    prms = np.polyfit(sub_axz[:, 0], sub_prf[:, 0], 2)
    prf_fit = np.polyval(prms, axz)
    curvedplane_along_t = matlib.repmat(prf_fit, xx, 1)
    reslice = reslice - curvedplane_along_t

    return reslice
