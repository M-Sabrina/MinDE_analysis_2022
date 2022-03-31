def local_DE_compare_analysis(
    MinD_st,
    MinE_st,
    frames_to_analyse=5,  # first ... frames
    halfspan=20,  # approximately half the wavelength
    sampling_density=1,  # density for subpixel resolution sampling
    edge=10,  # width of 2D histogram and max of 1D histogram
    bins_wheel=50,  # number of bins (horizontal/vertical) for velocity wheel (2D histogram)
    binwidth_sum=2.5,  # binwidth for velocity mangitude histogram
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
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from pyoptflow import HornSchunck

    from . import min_de_patterns_crests, min_de_patterns_velocity

    # rotate both Min stacks to match image directionality
    MinD_st = min_de_patterns_crests.adjust_stack_orientation(MinD_st)
    MinE_st = min_de_patterns_crests.adjust_stack_orientation(MinE_st)

    # build kernel for obtaining flow pattern
    kernel = np.ones((kernel_size_flow, kernel_size_flow), np.float32) / (
        kernel_size_flow ** 2
    )

    # work frames
    ff, rr, cc = np.shape(MinD_st)
    # check number of images
    if frames_to_analyse > ff:
        frames_to_analyse = ff
    print(f"Analysing {frames_to_analyse} frames")

    for fi in range(frames_to_analyse - 1):
        print(f"Working frame {fi} to {fi+1}")

        imD0 = MinD_st[fi, :, :]
        imD1 = MinD_st[fi + 1, :, :]
        imE0 = MinE_st[fi, :, :]

        imD0_smz = cv2.filter2D(imD0, -1, kernel)
        imD1_smz = cv2.filter2D(imD1, -1, kernel)

        # perform flow field analysis on image pair. note we use minD for optical flow
        U, V = HornSchunck(imD0_smz, imD1_smz, alpha=100, Niter=100)
        # obtain a binary image with 1 just where the intensity rises (the 'front' of a wave)
        wavesign_im = min_de_patterns_crests.get_rise_or_fall(U, V, imD0, demo=demo)
        # crests are the lines of pixels between the rise and the fall of a wave
        (
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
        ) = min_de_patterns_crests.get_crests(wavesign_im, imD0, 10, demo=demo)

        # use wavevect to get start and stop sampling coordinates in the direction of the flow
        profile_map1, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            imD0,
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
            halfspan,
            sampling_density,
            demo=demo,
        )
        profile_map2, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            imE0,
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
            halfspan,
            sampling_density,
            demo=demo,
        )
        # use these maps to get the local velocity per crest point
        delta_x_DE = min_de_patterns_crests.compare_crestmaps(
            profile_map1,
            profile_map2,
            sampling_density,
            look_ahead,
            demo=demo,
        )

        # build and analyze the  'velocity wheel'
        thiswheel, vxedges, vyedges = min_de_patterns_velocity.work_wheel(
            delta_x_DE,
            forward_wavevector_x,
            forward_wavevector_y,
            edge,
            bins_wheel,
        )
        if fi == 0:
            all_wheels = thiswheel
        else:
            all_wheels = all_wheels + thiswheel

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
            delta_x_DE,
            bins=np.arange(-edge, edge + binwidth_sum, binwidth_sum),
            color="royalblue",
            edgecolor="black",
        )
        ax_sum.set_ylabel("counts")
        ax_sum.set_xlabel("distance DE (pixels)")
        fig.tight_layout()

        print(f"Median DE-crest distance: {np.nanmedian(delta_x_DE):.02f} pixels")

        # demo section stop  ------------------------
        return (
            delta_x_DE,
            forward_wavevector_x,
            forward_wavevector_y,
            all_wheels,
            fig,
            ax_wheel,
            ax_sum,
        )
    else:
        return delta_x_DE, forward_wavevector_x, forward_wavevector_y, all_wheels
