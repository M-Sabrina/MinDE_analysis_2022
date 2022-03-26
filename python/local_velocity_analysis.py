def local_velocity_analysis(
    MinDE_st,
    frames_to_analyse=10,  # first ... frames
    halfspan=20,  # approximately half the wavelength
    sampling_density=1,  # density for subpixel resolution sampling
    edge=100,  # width of velocity wheel (2D histogram) and max of magnitude histogram
    bins_wheel=50,  # number of bins (horizontal/vertical) for velocity wheel (2D histogram)
    binwidth_sum=10,  # binwidth for velocity mangitude histogram
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
    import cv2
    import matplotlib.pyplot as plt
    import min_de_patterns_crests
    import min_de_patterns_velocity
    import numpy as np
    from pyoptflow import HornSchunck

    # rotate MinDE array to match image directionality
    MinDE_st = min_de_patterns_crests.adjust_stack_orientation(MinDE_st)

    # build kernel for obtaining flow pattern
    kernel = np.ones((kernel_size_flow, kernel_size_flow), np.float32) / (
        kernel_size_flow ** 2
    )

    # work all frames
    ff, rr, cc = np.shape(MinDE_st)
    # check number of images
    if frames_to_analyse > ff:
        frames_to_analyse = ff

    print(f"Analysing {frames_to_analyse} frames")

    for fi in range(frames_to_analyse - 1):
        print(f"Working frame {fi} to {fi+1}")

        im0 = MinDE_st[fi, :, :]
        im1 = MinDE_st[fi + 1, :, :]

        im0_smz = cv2.filter2D(im0, -1, kernel)
        im1_smz = cv2.filter2D(im1, -1, kernel)

        # perform flow field analysis on image pair
        U, V = HornSchunck(im0_smz, im1_smz, alpha=100, Niter=100)
        # obtain a binary image with 1 just where the intensity rises (the 'front' of a wave)
        wavesign_im = min_de_patterns_crests.get_rise_or_fall(U, V, im0, demo=demo)
        # crests are the lines of pixels between the rise and the fall of a wave
        (
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
        ) = min_de_patterns_crests.get_crests(wavesign_im, im0, halfspan / 2, demo=demo)
        # use wavevect to get start and stop sampling coordinates in the direction of the flow
        profile_map1, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            im0,
            crests_x,
            crests_y,
            forward_wavevector_x,
            forward_wavevector_y,
            halfspan,
            sampling_density,
            demo=demo,
        )
        profile_map2, xxgrid, yygrid = min_de_patterns_crests.sample_crests(
            im1,
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
        ax_wheel.set_xlabel("x-velocity (pixel/frame)")
        ax_wheel.set_ylabel("y-velocity (pixel/frame)")

        ax_sum.hist(
            velocities,
            bins=np.arange(0, edge + binwidth_sum, binwidth_sum),
            color="royalblue",
            edgecolor="black",
        )
        ax_sum.set_ylabel("counts")
        ax_sum.set_xlabel("velocity magnitude (pixel/frame)")
        fig.tight_layout()
        print(f"Median velocity magnitude: {np.nanmedian(velocities)} pixel/frame")

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
