"""
This function performs local distance ananlysis on two image stack of Min patterns,
one for MinD and one for MinE.
It returns a list of distance magnitudes and vector components in x- and y-direction,
and shows the results in the form of a 2D and 1D histograms.
Results are shown in a way that depicts how far MinE is running behind MinD as positive.

Full output (return values):
    all_velocities - velocity magnitude (pixels/frame)
    all_forward_wavevector_x - unit vector velocity x-component
    all_forward_wavevector_y - unit vector velocity y-component
    all_wheels - 2D histogram ("velocity wheel") data
    all_crests_x - crest position x
    all_crests_y - crest position y
    all_framenr - number of frame
    all_max_x1 - position of maximum of current frame (in units of sampling density)
    all_max_y1 - intensity at x1 
    all_max_x2 - position of maximum of next frame (in units of sampling density)
    all_max_y2 - intensity at x2
If demo set to True: additional output - figure handles (fig, ax_wheel, ax_sum)

Reference: Cees Dekker Lab; project: MinDE; researcher: Sabrina Meindlhumer.
Code designed & written by Jacob Kerssemakers and Sabrina Meindlhumer, 2022.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyoptflow import HornSchunck

from min_analysis_tools import (
    min_de_patterns_crests,
    min_de_patterns_velocity,
    peak_profile,
)


def local_DE_compare_analysis(
    MinD_st,
    MinE_st,
    frames_to_analyse=5,  # first ... frames
    halfspan=20,  # approximately half the wavelength
    sampling_width=1,  # density for subpixel resolution sampling
    edge=10,  # width of 2D histogram and max of 1D histogram
    bins_wheel=50,  # number of bins (horizontal/vertical) for velocity wheel (2D histogram)
    binwidth_sum=2.5,  # binwidth for velocity mangitude histogram
    kernel_size_general=20,  # kernel for first smoothing step
    kernel_size_flow=35,  # kernel for additional smoothing step
    look_ahead=-1,  # 1 -> in propagation direction, -1 -> against it
    demo=1,  # return figure handles
):
    """
    MinDE pattern velocity analysis
    Created 11/2021
    @authors: jkerssemakers, M-Sabrina
    Note:for Horn-Schunck, follow install instructions on https://github.com/scivision/pyoptflow
    """

    # rotate both Min stacks to match image directionality
    MinD_st = min_de_patterns_crests.adjust_stack_orientation(MinD_st)
    MinE_st = min_de_patterns_crests.adjust_stack_orientation(MinE_st)

    # build kernel for first smoothing step (for processing)
    general_kernel = np.ones((kernel_size_general, kernel_size_general), np.float32) / (
        kernel_size_general**2
    )

    # build kernel for obtaining flow pattern
    flow_kernel = np.ones((kernel_size_flow, kernel_size_flow), np.float32) / (
        kernel_size_flow**2
    )

    # work frames
    ff, rr, cc = np.shape(MinD_st)
    # check number of images
    if frames_to_analyse > ff:
        frames_to_analyse = ff
    print(f"Analysing {frames_to_analyse} frames")

    for fi in range(frames_to_analyse - 1):
        print(f"Working frame {fi} to {fi+1}")

        imD0_raw = MinD_st[fi, :, :]
        imD1_raw = MinD_st[fi + 1, :, :]
        imE0_raw = MinE_st[fi, :, :]

        imD0_smz = cv2.filter2D(imD0_raw, -1, general_kernel)
        imD1_smz = cv2.filter2D(imD1_raw, -1, general_kernel)
        imE0_smz = cv2.filter2D(imE0_raw, -1, general_kernel)

        imD0_smz_flow = cv2.filter2D(imD0_smz, -1, flow_kernel)
        imD1_smz_flow = cv2.filter2D(imD1_smz, -1, flow_kernel)

        # perform flow field analysis on image pair. note we use minD for optical flow
        U, V = HornSchunck(imD0_smz_flow, imD1_smz_flow, alpha=100, Niter=100)
        # obtain a binary image with 1 just where the intensity rises (the 'front' of a wave)
        wavesign_im = min_de_patterns_crests.get_rise_or_fall(U, V, imD0_raw, demo=demo)
        # crests are the lines of pixels between the rise and the fall of a wave
        (
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
        ) = min_de_patterns_crests.get_crests(wavesign_im, imD0_smz, 10, demo=demo)

        # use wavevect to get start and stop sampling coordinates in the direction of the flow
        profile_map1, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            imD0_smz,
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
            halfspan,
            sampling_width,
            demo=demo,
        )
        profile_map2, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            imE0_smz,
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
            halfspan,
            sampling_width,
            demo=demo,
        )
        # use these maps to get the local velocity per crest point
        (
            delta_x_DE,
            max_x1,
            max_y1,
            max_x2,
            max_y2,
        ) = min_de_patterns_crests.compare_crestmaps(
            profile_map1,
            profile_map2,
            sampling_width,
            look_ahead,
            return_peaks=True,
            demo=demo,
        )

        # build and analyze the  'velocity wheel'
        thiswheel, vxedges, vyedges = min_de_patterns_velocity.work_wheel(
            delta_x_DE,
            forward_wavevector_x,
            forward_wavevector_y,
            edge,
            bins_wheel,
            demo=demo,
        )

        # save frame number
        framenr = np.array([fi] * len(delta_x_DE))

        # save variables
        if fi == 0:
            all_wheels = thiswheel
            all_delta_x_DE = delta_x_DE
            all_forward_wavevector_x = forward_wavevector_x
            all_forward_wavevector_y = forward_wavevector_y
            all_crests_x = crests_x
            all_crests_y = crests_y
            all_framenr = framenr
            all_max_x1 = max_x1
            all_max_y1 = max_y1
            all_max_x2 = max_x2
            all_max_y2 = max_y2
        else:
            all_wheels = all_wheels + thiswheel
            all_delta_x_DE = np.append(all_delta_x_DE, delta_x_DE)
            all_forward_wavevector_x = np.append(
                all_forward_wavevector_x, forward_wavevector_x
            )
            all_forward_wavevector_y = np.append(
                all_forward_wavevector_y, forward_wavevector_y
            )
            all_crests_x = np.append(all_crests_x, crests_x)
            all_crests_y = np.append(all_crests_y, crests_y)
            all_framenr = np.append(all_framenr, framenr)
            all_max_x1 = np.append(all_max_x1, max_x1)
            all_max_y1 = np.append(all_max_y1, max_y1)
            all_max_x2 = np.append(all_max_x2, max_x2)
            all_max_y2 = np.append(all_max_y2, max_y2)

    # demo section start ------------------------
    # plot full stackresult
    if demo == 1:
        fig, (ax_wheel, ax_sum) = plt.subplots(1, 2)
        ax_wheel.imshow(
            all_wheels,
            interpolation="nearest",
            origin="lower",
            extent=[vxedges[0], vxedges[-1], vyedges[0], vyedges[-1]],
        )
        ax_wheel.set_box_aspect(1)
        ax_wheel.axvline(x=0, color="white")
        ax_wheel.axhline(y=0, color="white")
        ax_wheel.set_xlabel("x-distance (pixels)")
        ax_wheel.set_ylabel("y-distance (pixels)")

        ax_sum.hist(
            -delta_x_DE,  # set to negative to indicate E is running behind D
            bins=np.arange(-edge, edge + binwidth_sum, binwidth_sum),
            color="royalblue",
            edgecolor="black",
        )
        ax_sum.set_ylabel("counts")
        ax_sum.set_xlabel("distance DE (pixels)")
        fig.tight_layout()

        distance_med = np.nanmedian(delta_x_DE)
        # some extra numbers processed from the histogram
        bins = np.arange(-edge, edge + binwidth_sum, binwidth_sum)
        (hist_prf, bin_edges) = np.histogram(delta_x_DE, bins)
        pki = peak_profile.get_maxima(hist_prf, N_max=1)
        fwhm_pix = peak_profile.get_FWHM(hist_prf, pki[0])
        fwhm_pifr = binwidth_sum * fwhm_pix
        print(f"peak distance: {bin_edges[pki[0]]:.02f} pixels")
        print(f"FWHM distance: {fwhm_pifr:.02f} pixels")
        print(f"Median distance: {distance_med:.02f} pixels")
        # demo section stop  ------------------------
        return (
            delta_x_DE,
            forward_wavevector_x,
            forward_wavevector_y,
            all_wheels,
            all_crests_x,
            all_crests_y,
            all_framenr,
            all_max_x1,
            all_max_y1,
            all_max_x2,
            all_max_y2,
            fig,
            ax_wheel,
            ax_sum,
        )
    else:
        return (
            delta_x_DE,
            all_forward_wavevector_x,
            all_forward_wavevector_y,
            all_wheels,
            all_crests_x,
            all_crests_y,
            all_framenr,
            all_max_x1,
            all_max_y1,
            all_max_x2,
            all_max_y2,
        )
