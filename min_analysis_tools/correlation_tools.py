"""
Series of tools for autocorrelation analysis.
"""

from cmath import nan

import matplotlib.pyplot as plt
import numpy as np
from numpy import conj, real
from numpy.fft import fft2, fftshift, ifft2
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import find_peaks

from min_analysis_tools.clean_kymograph import clean_kymograph

cm = 1 / 2.54  # centimeters in inches (for matplotlib figure size)


def autocorrelation(image):
    """
    Performs autocorrelation on an image.
    Input: 2D numpy array.
    Output: 2D numpy array (autocorrelation matrix).
    """

    # normalization
    image = image - np.min(image)
    image = image / np.sum(image)
    image = image - np.mean(image)

    # calculate correlation matrix
    crmx = real(ifft2(fft2(image) * conj(fft2(image))))
    return crmx


def radial_profile(crmx):
    """
    Performs radial averaging of a matrix, around the center.
    Gratefully taken from https://stackoverflow.com/a/21242776.
    Input: 2D numpy array (correlation matrix)
    Output: 1D numpy array (radial average trace)
    """
    ny, nx = np.shape(crmx)

    # for unequal nx and ny, crop to square shape
    if not (ny == nx):
        center = np.array([nx // 2, ny // 2])
        w = min(nx, ny) // 2
        crmx_cropped = crmx[
            center[1] - w : center[1] + w, center[0] - w : center[0] + w
        ]
        crmx = crmx_cropped
    ny, nx = np.shape(crmx)
    center = nx // 2
    y, x = np.indices((crmx.shape))
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), crmx.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def analyze_peaks(trace, kernel_size=0, demo=False, title_demo=None):
    """
    Finds first minimum and maximum of a trace. Optional pre-smoothing. No interpolation.
    Input: slice as 1D numpy array
    Output: position and value of first minimum, position and value of first maximum
    """

    trace_original = trace

    if kernel_size > 0:
        # smooth trace
        # https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
        kernel = np.ones(kernel_size) / kernel_size
        trace = np.convolve(trace, kernel, mode="same")

    # evaluate local maxima and minima of profile
    local_maxima = np.array(find_peaks(trace)[0])
    local_minima = np.array(find_peaks(-trace)[0])

    first_min_pos = local_minima[0]  # first valley represents "anticorrelation"
    # identify first peak (higher position than that of minimum)
    npeak = 0
    first_max_pos = local_maxima[npeak]
    while first_max_pos < first_min_pos:
        npeak = npeak + 1
        first_max_pos = local_maxima[npeak]

    # get amplitudes of radial correlation function at min and max
    first_max_val = trace[first_max_pos]
    first_min_val = trace[first_min_pos]

    if demo:  # show trace and found minimum & maximum
        fig, ax = plt.subplots()
        x_axis = np.arange(len(trace))
        ax.plot(x_axis, trace_original)
        ax.axvline(x=first_min_pos, color="green", linestyle="dotted")
        ax.axvline(x=first_max_pos, color="magenta", linestyle="dotted")
        if title_demo:
            ax.set_title(f"{title_demo}, smoothing kernel = {kernel_size} pixel")
        fig.show()

    return first_min_pos, first_min_val, first_max_pos, first_max_val


def quadratic_spline_roots(spl):
    """
    To be used in function fit_analyze_peaks.
    Gratefully taken from https://stackoverflow.com/questions/50371298/find-maximum-minimum-of-a-1d-interpolated-function [2022/05/28]
    """
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a + b) / 2), spl(b)
        t = np.roots([u + w - 2 * v, w - u, 2 * v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t * (b - a) / 2 + (b + a) / 2)
    return np.array(roots)


def fit_analyze_peaks(trace, demo=False, title_demo=None):
    """
    Finds first minimum and maximum of a trace via cubic fitting.
    Adapted from https://stackoverflow.com/questions/50371298/find-maximum-minimum-of-a-1d-interpolated-function [2022/05/28]
    Input: slice as 1D numpy array
    Output: position and value of first minimum, position and value of first maximum
    """

    # interpolate with polynomials
    x_axis = np.arange(len(trace))
    f = InterpolatedUnivariateSpline(x_axis, trace, k=3)

    # determine first min and max
    cr_pts = quadratic_spline_roots(
        f.derivative()
    )  # positions of local minima and maxima
    cr_vals = f(cr_pts)

    # disentangle local maxima and minima
    mean_trace = np.mean(trace)
    local_maxima = cr_pts[cr_vals > mean_trace]
    local_maxima_vals = cr_vals[cr_vals > mean_trace]
    local_minima = cr_pts[cr_vals < mean_trace]
    local_minima_vals = cr_vals[cr_vals < mean_trace]

    # assume that first local minimum is correctly identified
    first_min_pos = local_minima[0]
    first_min_val = local_minima_vals[0]

    # take first local maximum after first local minimum (probably 1st or 2nd)
    for n, max_candidate in enumerate(local_maxima):
        if max_candidate > first_min_pos:
            first_max_pos = max_candidate
            first_max_val = local_maxima_vals[n]
            break

    if demo:  # show fit
        fig, ax = plt.subplots()
        ax.plot(x_axis, trace)
        x_temp = np.linspace(0, len(trace) - 1, 1000)
        y_temp = f(x_temp)
        ax.plot(x_temp, y_temp, "--")
        ax.axvline(x=first_min_pos, color="green", linestyle="dotted")
        ax.axvline(x=first_max_pos, color="magenta", linestyle="dotted")
        if title_demo:
            ax.set_title(title_demo)
        fig.show()

    return first_min_pos, first_min_val, first_max_pos, first_max_val


def get_spatial_correlation_matrixes(
    MinDE_st_full,
    frames_to_analyse,
    demo=1,
    verbose=True,
):
    """
    Performs spatial autocorrelation for a set number of frames (frames_to_analyze) on an image stack (stack).
    Output: numpy array of size frames_to_analyse x (size of correlation matrix).
    If save is True, save exemplary image and correlation matrix as image to savepath.
    If demo is True, show this image.
    """

    # load stack as np array
    fz, fy, fx = np.shape(MinDE_st_full)

    # check number of images
    if frames_to_analyse > fz:
        frames_to_analyse = fz
    if verbose:
        print(f"Analysing {frames_to_analyse} frames")

    # perform autocorrelation analysis for every frames
    for framenr in range(frames_to_analyse):

        image = MinDE_st_full[framenr, :, :]

        # calculate correlation matrix for image
        crmx = autocorrelation(image)

        # shift correlation matrix to have peak in the center
        crmx = fftshift(crmx)

        # normalize to maximum
        crmx = crmx / np.max(crmx)

        # first frames: create storage for correltaion matrixes
        # also, plot example image and correlation matrix if requested
        if framenr == 0:
            crmx_storage = np.empty((frames_to_analyse, fy, fx))
            crmx_storage[framenr, :, :] = crmx

            if demo > 0:
                fig, (ax_orig, ax_corr) = plt.subplots(1, 2, figsize=(18 * cm, 10 * cm))
                ax_orig.imshow(image)
                ax_orig.set_title("Image")
                ax_orig.set_xlabel("x (pixels)")
                ax_orig.set_ylabel("y (pixels)")
                ax_corr.imshow(crmx, extent=[-fx // 2, fx // 2, -fy // 2, fy // 2])
                ax_corr.set_title("Autocorrelation")
                ax_corr.set_xlabel("$\Delta$x (pixels)")
                ax_corr.set_ylabel("$\Delta$y (pixels)")
                ax_corr.yaxis.set_label_position("right")
                ax_corr.yaxis.tick_right()
                plt.subplots_adjust(wspace=0.1)
        else:
            crmx_storage[framenr, :, :] = crmx

    # return correlation matrixes (and figure handles, if requested)
    if demo > 0:
        return crmx_storage, fig, ax_corr, ax_orig
    else:
        return crmx_storage


def analyze_radial_profiles(
    crmx_storage,
    nmperpix=None,
    demo=1,
):
    """
    Create radially averaged profile from spatial autocorrelation matrix.
    Analyze this profile trace with respect to its first minimum (valley) and maximum (peak),
    with the latter corresponding to the pattern's characteristic wavelength.
    """

    # get size of correlation matrix
    fz, fy, fx = np.shape(crmx_storage)

    # prepare storage for characteristic parameters
    first_min_pos = np.empty(fz)
    first_min_val = np.empty(fz)
    first_max_pos = np.empty(fz)
    first_max_val = np.empty(fz)

    # recognize unit
    if nmperpix is None:
        unit = "pixels"
    else:
        unit = "µm"

    # prepare figure to plot radial profiles
    if demo > 0:
        fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 10 * cm))
        ax.set_xlabel(f"distance ({unit})")
        ax.set_ylabel("autocorrelation (a.u.)")

    # calculate radial profiles and determine characteristic values
    for framenr in range(fz):

        # calculate radial profile from correlation matrix
        crmx = crmx_storage[framenr, :, :]
        radialprofile = radial_profile(crmx)

        # prepare x axes (in pixels or um) for plotting
        x_axis = np.array(range(np.shape(radialprofile)[0]))  # 1 2 3 ...
        if nmperpix is not None:
            x_axis = x_axis * nmperpix / 1000  # convert to um, if factor is provided

        # plot profiles
        if demo > 0:
            ax.plot(x_axis, radialprofile, label=f"frame {framenr+1}")

        try:
            if demo >= 2:  # diagnosis (requires click to proceed):
                demo_fitting = True
                title_demo = f"Radial profile of frame nr {framenr+1}"
            else:
                demo_fitting = False
                title_demo = None
            # analyze local maxima and minima (peaks and valleys)
            (
                first_min_pos[framenr],
                first_min_val[framenr],
                first_max_pos[framenr],
                first_max_val[framenr],
            ) = analyze_peaks(
                radialprofile,
                kernel_size=int(min(fx, fy) / 100),
                demo=demo_fitting,
                title_demo=title_demo,
            )
        except Exception as e:
            print(f"Peak analysis not successfull for frame {framenr+1}")
            print(f"Cause: {e}")
            first_min_pos[framenr] = None
            first_min_val[framenr] = None
            first_max_pos[framenr] = None
            first_max_val[framenr] = None

    # convert positions from pixels to um
    if nmperpix is not None:
        first_max_pos = first_max_pos * nmperpix / 1000  # wavelength in µm
        first_min_pos = first_min_pos * nmperpix / 1000  # wavelength / 2 µm (approx)

    # calculate means for plot
    mean_min_pos = np.mean(first_min_pos)
    mean_max_pos = np.mean(first_max_pos)

    # indicate found min and max positions in plot
    if demo > 0:
        ax.legend(loc="upper right")
        ax.axvline(x=mean_min_pos, color="green", linestyle="dotted", linewidth=1.0)
        ax.axvline(x=mean_max_pos, color="magenta", linestyle="dotted", linewidth=1.0)

    # return means (and figure handles, if requested)
    if demo > 0:
        return (
            first_min_pos,
            first_min_val,
            first_max_pos,
            first_max_val,
            fig,
            ax,
        )
    else:
        return (
            first_min_pos,
            first_min_val,
            first_max_pos,
            first_max_val,
        )


def get_temporal_correlation_matrixes(
    MinDE_st_resliced,
    axis,
    kymoband,
    reps_per_kymostack,
    demo=1,
    verbose=True,
):
    """
    Performs autocorrelation for a set number of resliced frames (reps_per_kymostack) on a
    resliced image stack (stack).
    Output: numpy array of size reps_per_kymostack x (size of correlation matrix).
    If save is True, save exemplary resliced frames and correlation matrix as image to savepath.
    If demo is True, show this image.
    """

    ax1, ax2, ax3 = np.shape(MinDE_st_resliced)
    # for "x": ax1 = y, ax2 = x, ax3 = z of original shape z y x
    # for "y" now: ax1 = x, ax2 = y, ax3 = z of original shape z y x

    # pick a "band" of kymographs to analyze
    slices2analyze = ax1 * np.linspace(
        0.5 - kymoband / 2, 0.5 + kymoband / 2, reps_per_kymostack
    )
    slices2analyze = slices2analyze.astype(int)

    # message to console
    if axis == "x":
        constant_axis = "y"
    else:
        constant_axis = "x"
    if verbose:
        print(f"Analyzing t-{axis} slices for {constant_axis} = {slices2analyze}")

    # perform autocorrelation on selected resliced frames and collect the results
    for ni in range(reps_per_kymostack):

        # load and prepare
        slice_i = slices2analyze[ni]
        resliceim = MinDE_st_resliced[slice_i, :, :]
        resliceim = clean_kymograph(resliceim)
        crmx = autocorrelation(resliceim)

        # first kymo: create storage of suitable size, save example kymo
        if ni == 0:
            crmx_storage = np.empty((reps_per_kymostack, ax2, ax3))
            crmx_storage[ni, :, :] = crmx

            if demo > 0:
                # prepare correlation matrix for visualization
                crmx = fftshift(crmx)
                fig, (ax_orig, ax_corr) = plt.subplots(1, 2, figsize=(18 * cm, 10 * cm))
                ax_orig.imshow(resliceim, aspect="auto")
                ax_orig.set_title(f"Resliced image t-{axis}, {constant_axis}={slice_i}")
                ax_orig.set_xlabel("t (frames)")
                ax_orig.set_ylabel(f"{axis} (pixels)")
                ax_corr.imshow(
                    crmx,
                    aspect="auto",
                    extent=[-ax3 // 2, ax3 // 2, -ax2 // 2, ax2 // 2],
                )
                ax_corr.set_title("Autocorrelation")
                ax_corr.set_xlabel("$\Delta$t (frames)")
                ax_corr.set_ylabel(f"$\Delta${axis} (pixels)")
                ax_corr.yaxis.set_label_position("right")
                ax_corr.yaxis.tick_right()
                plt.subplots_adjust(wspace=0.1)
        else:
            crmx_storage[ni, :, :] = crmx

    if demo > 0:
        return crmx_storage, slices2analyze, fig, ax_corr, ax_orig
    else:
        return crmx_storage, slices2analyze


def analyze_temporal_profiles(
    axis,
    crmx_storage,
    slices2analyze,
    frpermin=None,
    demo=1,
):
    """
    Take first row in autocorrelation matrix (corresponding to t=0).
    Analyze this profile trace with respect to its first minimum (valley) and maximum (peak),
    with the latter corresponding to the pattern's oscillation period.
    """

    # get size of correlation matrix storage
    ax1, ax2, ax3 = np.shape(crmx_storage)  # note: ax1 = reps_per_kymostack

    # prepare storage for characteristic parameters
    first_min_pos = np.empty(ax1)
    first_min_val = np.empty(ax1)
    first_max_pos = np.empty(ax1)
    first_max_val = np.empty(ax1)

    # recognize unit
    if frpermin is None:
        unit = "frames"
    else:
        unit = "s"

    # prepare figure to plot radial profiles
    if demo > 0:
        fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 10 * cm))
        ax.set_xlabel(f"$\Delta$t ({unit})")
        ax.set_ylabel("autocorrelation (a.u.)")

    # plot profiles and calculate characteristic values
    for framenr in range(ax1):

        # define trace to analyze (first row, from t=0 to t=tmax/2)
        crmx = crmx_storage[framenr, :, :]
        tmp = crmx[0, :]  # first row
        prf_tcor = tmp[0 : (ax3 // 2)]  # first tmax/2 time points
        prf_tcor = prf_tcor / max(prf_tcor)

        # prepare x axes (in frames or s) for plotting
        x_axis = np.array(range(np.shape(prf_tcor)[0]))  # 1 2 3 ...
        if frpermin is not None:
            x_axis = x_axis * 60 / frpermin  # convert to seconds, if factor is provided

        if axis == "x":
            constant_axis = "y"
        else:
            constant_axis = "x"

        # plot profiles
        if demo > 0:
            ax.plot(
                x_axis, prf_tcor, label=f"{constant_axis} = {slices2analyze[framenr]}"
            )

        try:
            if demo >= 2:  # diagnosis (requires click to proceed):
                demo_fitting = True
                title_demo = f"Fitting trace of resliced frame nr (= slice at constant {constant_axis}) {slices2analyze[framenr]}"
            else:
                demo_fitting = False
                title_demo = None
            # analyze local maxima and minima (peaks and valleys)
            (
                first_min_pos[framenr],
                first_min_val[framenr],
                first_max_pos[framenr],
                first_max_val[framenr],
            ) = fit_analyze_peaks(prf_tcor, demo=demo_fitting, title_demo=title_demo)
        except Exception as e:
            print(
                f"Peak analysis not successful for resliced frame {framenr+1} ({constant_axis} = {slices2analyze[framenr]})"
            )
            print(f"Cause: {e}")
            first_min_pos[framenr] = None
            first_min_val[framenr] = None
            first_max_pos[framenr] = None
            first_max_val[framenr] = None

    # convert positions from pixels to um
    if frpermin is not None:
        first_max_pos = first_max_pos * 60 / frpermin  # oscillation period in s
        first_min_pos = first_min_pos * 60 / frpermin  # oscillation period / 2 (approx)

    # calculate means
    mean_min_pos = np.mean(first_min_pos)
    mean_max_pos = np.mean(first_max_pos)

    # indicate found min and max positions in plot
    if demo > 0:
        ax.legend(loc="upper right")
        ax.axvline(x=mean_min_pos, color="green", linestyle="dotted", linewidth=1.0)
        ax.axvline(x=mean_max_pos, color="magenta", linestyle="dotted", linewidth=1.0)

    # return arrays
    if demo > 0:
        return (
            first_min_pos,
            first_min_val,
            first_max_pos,
            first_max_val,
            fig,
            ax,
        )
    else:
        return (
            first_min_pos,
            first_min_val,
            first_max_pos,
            first_max_val,
        )
