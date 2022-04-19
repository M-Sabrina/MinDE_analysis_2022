import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyoptflow import HornSchunck

from min_analysis_tools import min_de_patterns_crests, min_de_patterns_velocity


def local_velocity_analysis(
    MinDE_st,
    frames_to_analyse=10,  # first ... frames
    halfspan=20,  # approximately half the wavelength
    sampling_density=1,  # density for subpixel resolution sampling
    edge=100,  # width of velocity wheel (2D histogram) and max of magnitude histogram
    bins_wheel=50,  # number of bins (horizontal/vertical) for velocity wheel (2D histogram)
    binwidth_sum=10,  # binwidth for velocity mangitude histogram
    kernel_size_general=20,  # kernel for first smoothing step
    kernel_size_flow=35,  # kernel for additional smoothing step
    look_ahead=1,  # 1 -> in propagation direction, -1 -> against it
    demo=1,  # return figure handles
):
    """
    MinDE pattern velocity analysis
    Created 11/2021
    @authors: jkerssemakers, M-Sabrina
    Note:for Horn-Schunck, follow install instructions on https://github.com/scivision/pyoptflow
    """

    # rotate MinDE array to match image directionality
    MinDE_st = min_de_patterns_crests.adjust_stack_orientation(MinDE_st)

    # build kernel for first smoothing step (for processing)
    general_kernel = np.ones((kernel_size_general, kernel_size_general), np.float32) / (
        kernel_size_general**2
    )

    # build kernel for obtaining flow pattern
    flow_kernel = np.ones((kernel_size_flow, kernel_size_flow), np.float32) / (
        kernel_size_flow**2
    )

    # work all frames
    ff, rr, cc = np.shape(MinDE_st)
    # check number of images
    if frames_to_analyse > ff:
        frames_to_analyse = ff

    print(f"Analysing {frames_to_analyse} frames")

    for fi in range(frames_to_analyse - 1):
        print(f"Working frame {fi} to {fi+1}")

        im0_raw = MinDE_st[fi, :, :]
        im1_raw = MinDE_st[fi + 1, :, :]

        im0_smz = cv2.filter2D(im0_raw, -1, general_kernel)
        im1_smz = cv2.filter2D(im1_raw, -1, general_kernel)

        im0_smz_flow = cv2.filter2D(im0_smz, -1, flow_kernel)
        im1_smz_flow = cv2.filter2D(im1_smz, -1, flow_kernel)

        # perform flow field analysis on image pair
        U, V = HornSchunck(im0_smz_flow, im1_smz_flow, alpha=100, Niter=100)
        # obtain a binary image with 1 just where the intensity rises (the 'front' of a wave)
        wavesign_im = min_de_patterns_crests.get_rise_or_fall(
            U, V, im0_smz_flow, demo=demo
        )
        # crests are the lines of pixels between the rise and the fall of a wave
        (
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
        ) = min_de_patterns_crests.get_crests(
            wavesign_im, im0_smz, halfspan / 2, demo=demo
        )
        # use wavevect to get start and stop sampling coordinates in the direction of the flow
        profile_map1, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            im0_smz,
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
            halfspan,
            sampling_density,
            demo=demo,
        )
        profile_map2, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            im1_smz,
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
            halfspan,
            sampling_density,
            demo=demo,
        )
        # use these maps to get the local velocity per crest point
        velocities = min_de_patterns_crests.compare_crestmaps(
            profile_map1, profile_map2, sampling_density, look_ahead, demo=demo
        )

        # build and analyze the  'velocity wheel'
        thiswheel, vxedges, vyedges = min_de_patterns_velocity.work_wheel(
            velocities,
            forward_wavevector_x,
            forward_wavevector_y,
            edge,
            bins_wheel,
            demo=demo,
        )
        if fi == 0:
            all_wheels = thiswheel
        else:
            all_wheels = all_wheels + thiswheel

    # plot full stackresult:
    if demo > 0:
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
        ax_wheel.set_xlabel("x-velocity (pixels/frame)")
        ax_wheel.set_ylabel("y-velocity (pixels/frame)")

        ax_sum.hist(
            velocities,
            bins=np.arange(0, edge + binwidth_sum, binwidth_sum),
            color="royalblue",
            edgecolor="black",
        )
        ax_sum.set_ylabel("counts")
        ax_sum.set_xlabel("velocity magnitude (pixels/frame)")
        fig.tight_layout()
        print(
            f"Median velocity magnitude: {np.nanmedian(velocities):.02f} pixels/frame"
        )

        return (
            velocities,
            forward_wavevector_x,
            forward_wavevector_y,
            all_wheels,
            fig,
            ax_wheel,
            ax_sum,
        )
    else:
        return velocities, forward_wavevector_x, forward_wavevector_y, all_wheels
