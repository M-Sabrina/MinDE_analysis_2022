import csv
from enum import Enum
from pathlib import Path

import correlation_tools
import get_data
import matplotlib.pyplot as plt
import numpy as np
from local_DE_compare_analysis import local_DE_compare_analysis
from local_velocity_analysis import local_velocity_analysis


class Action(Enum):
    GLOBAL_SPATIAL = 1  # -> spatial autocorrelation
    GLOBAL_TEMPORAL = 2  # -> temporal autocorrelation
    LOCAL_VELOCITY = 3  # -> local velocity analysis
    LOCAL_DISTANCES = 4  # -> local 2-channel crest comparison -> 2 input files required


# Choose action HERE (see list in class "Action" above):
action = Action.GLOBAL_SPATIAL

# General / display parameters:
size = 512  # data will be downsized to this (set "None" to skip)
kernel_size = 16  # kernel size for initial image smoothening (set "None" to skip)
frames_to_analyse = 5  # set at a very large number to analyse all frames

# Stack information (set both to None to work with pixel and frames only):
nmperpix = 594  # nanometer per pixel (assuming aspect ratio = 1)
frpermin = 4  # frames per minute

# Temporal autocorrelation analysis parameters:
reps_per_kymostack = 5  # pick ... kymographs around middle
kymoband = 0.8  # analyse middle ... part of image

# Local analysis parameters:
halfspan = 30  # halfspan for velocities / distances (ideally ~ wavelength/2)
kernel_size_flow = 35  # building smoothening kernel needed for flow analysis

# Define stack_path -> SET
# single file (actions 1, 2, 3)
infile = Path(
    r"INPUT_PATH_HERE\my_file.tif"
)
# two files (action 4)
in_E = Path(
    r"INPUT_PATH_HERE\my_file_E.tif"
)
in_D = Path(
    r"INPUT_PATH_HERE\my_file_D.tif"
)

###########################################################

# create outpath
outpath = infile.parent / f"results"
if not outpath.is_dir():
    outpath.mkdir()

# load stack
if action == Action.LOCAL_DISTANCES:
    MinE_st = get_data.load_stack(in_E, size=size, kernel_size=kernel_size)
    stackname_E = in_E.stem
    MinD_st = get_data.load_stack(in_D, size=size, kernel_size=kernel_size)
    stackname_D = in_D.stem
else:
    Min_st = get_data.load_stack(infile, size=size, kernel_size=kernel_size)
    stackname = infile.stem

###########################################################

## Global analysis: Spatial autocorrelation

if action == Action.GLOBAL_SPATIAL:

    # recognize unit
    if nmperpix is None:
        unit = "pixel"
    else:
        unit = "µm"

    # autocorrelation --> obtain array of autocorrelation matrixes for analyzed frames
    (
        crmx_storage,
        fig,
        ax_corr,
        ax_orig,
    ) = correlation_tools.get_spatial_correlation_matrixes(
        Min_st,
        frames_to_analyse,
    )
    example_fig_path = outpath / f"{stackname}_spatial_autocorr.png"
    fig.savefig(example_fig_path, dpi=500)
    plt.close(fig)

    # calculate radially averaged profile traces
    # analyze them with respect to first min and max, mean-average
    (
        first_min_pos,
        first_min_val,
        first_max_pos,
        first_max_val,
        peak_valley_diff,
        fig,
        ax,
    ) = correlation_tools.analyze_radial_profiles(crmx_storage, nmperpix)
    profile_fig_path = outpath / f"{stackname}_radialprofiles.png"
    fig.savefig(profile_fig_path, dpi=500)
    plt.close(fig)

    print(f"mean position of first valley: {np.mean(first_min_pos)} {unit}")
    print(f"mean position of first peak (!): {np.mean(first_max_pos)} {unit}")
    print(f"mean peak-valley difference: {np.mean(peak_valley_diff)}")

    # create csv file for saving characteristic parameters later
    csv_file = outpath / f"{stackname}_spatial_autocorrelation_results.csv"
    # create header for csv file
    header = [
        "frame_nr",  # frame number
        f"valley_pos ({unit})",  # position of first minimum
        "valley_val",  # amplitude of first minimum
        f"peak_pos ({unit})",  # position of first maximum --> wavelength
        "peak_val",  # amplitude of first maximum
        "peak_valley_diff",
    ]
    with open(csv_file, "w") as csv_f:  # will overwrite existing
        writer = csv.writer(csv_f, delimiter="\t")
        writer.writerow(header)
        for n in range(first_max_pos.size):
            writer.writerow(
                [
                    n + 1,
                    first_min_pos[n],
                    first_min_val[n],
                    first_max_pos[n],
                    first_max_val[n],
                    peak_valley_diff[n],
                ]
            )

###########################################################

## Global analysis: Temporal autocorrelation

elif action == Action.GLOBAL_TEMPORAL:

    # recognize unit
    if frpermin is None:
        unit = "frame"
    else:
        unit = "s"

    MinDE_shift_tx = np.moveaxis(Min_st, 0, -1)  # creates t-x resliced frames
    MinDE_shift_yt = np.moveaxis(Min_st, -1, 0)  # creates y-t resliced frames
    ff, rr, cc = np.shape(MinDE_shift_yt)
    MinDE_shift_ty = np.empty((ff, cc, rr))
    for frame in range(ff):  # tranpose images to create t-y resliced frames
        MinDE_shift_ty[frame, :, :] = np.transpose(MinDE_shift_yt[frame, :, :])

    # create csv file for saving characteristic parameters later
    csv_file_path = outpath / f"{stackname}_temporal_autocorrelation_results.csv"
    # create header for csv file
    header = [
        f"x (pixel)",
        f"y (pixel)",
        f"valley_pos ({unit})",  # position of first minimum
        "valley_val",  # amplitude of first minimum
        f"peak_pos ({unit})",  # position of first maximum --> oscillation period
        "peak_val",  # amplitude of first maximum
        "peak_valley_diff",
    ]
    with open(csv_file_path, "w") as csv_f:  # will overwrite existing
        # create the csv writer
        writer = csv.writer(csv_f, delimiter="\t")
        writer.writerow(header)

    # for x-t slices or for y-t slices
    for axis in ["x", "y"]:

        # pick correct resliced file
        if axis == "x":
            MinDE_st_resliced = MinDE_shift_tx
        else:
            MinDE_st_resliced = MinDE_shift_ty

        # temporal autocorrelation on selected slices
        (
            crmx_storage,
            slices2analyze,
            fig,
            ax_corr,
            ax_orig,
        ) = correlation_tools.get_temporal_correlation_matrixes(
            MinDE_st_resliced,
            axis,
            kymoband,
            reps_per_kymostack,
        )
        example_fig_path = outpath / f"{stackname}_{axis}t_temporal_autocorr.png"
        fig.savefig(example_fig_path, dpi=500)
        plt.close(fig)

        # take trace in first row of each correlation matrix
        # analyze it with respect to first min and max, mean-average
        (
            first_min_pos,
            first_min_val,
            first_max_pos,
            first_max_val,
            peak_valley_diff,
            fig,
            ax,
        ) = correlation_tools.analyze_temporal_profiles(
            axis,
            crmx_storage,
            slices2analyze,
            frpermin,
        )
        profile_fig_path = outpath / f"{stackname}_t{axis}_profiles.png"
        fig.savefig(profile_fig_path, dpi=500)

        # store characteristc parameters in csv and output average to console

        if axis == "x":

            with open(csv_file_path, "a") as csv_f:  # append to existing
                # create the csv writer
                writer = csv.writer(csv_f, delimiter="\t")
                # write data to the csv file
                for n in range(first_max_pos.size):
                    writer.writerow(
                        [
                            "all",  # all x values along line for constant y
                            slices2analyze[n] + 1,  # y (resliced stack framenr)
                            first_min_pos[n],
                            first_min_val[n],
                            first_max_pos[n],
                            first_max_val[n],
                            peak_valley_diff[n],
                        ]
                    )
            # first time: store values for averaging later
            tmp_first_min_pos = first_min_pos
            tmp_first_min_val = first_min_val
            tmp_first_max_pos = first_max_pos
            tmp_first_max_val = first_max_val
            tmp_peak_valley_diff = peak_valley_diff

        if axis == "y":

            with open(csv_file_path, "a") as csv_f:  # append to existing
                # create the csv writer
                writer = csv.writer(csv_f, delimiter="\t")
                # write data to the csv file
                for n in range(first_max_pos.size):
                    writer.writerow(
                        [
                            slices2analyze[n] + 1,  # x (resliced stack framenr)
                            "all",  # all y values along line for constant x
                            first_min_pos[n],
                            first_min_val[n],
                            first_max_pos[n],
                            first_max_val[n],
                            peak_valley_diff[n],
                        ]
                    )

            # second time: join arrays from x- and y-slices and output average to console
            first_min_pos = np.append(tmp_first_min_pos, first_min_pos)
            first_min_val = np.append(tmp_first_min_val, first_min_val)
            first_max_pos = np.append(tmp_first_max_pos, first_max_pos)
            first_max_val = np.append(tmp_first_max_val, first_max_val)
            peak_valley_diff = np.append(tmp_peak_valley_diff, peak_valley_diff)

    print(f"mean position of first valley: {np.mean(first_min_pos)} {unit}")
    print(f"mean position of first peak (!): {np.mean(first_max_pos)} {unit}")
    print(f"mean peak-valley difference: {np.mean(peak_valley_diff)}")

###########################################################

## Local analysis: Velocity wheel

elif action == Action.LOCAL_VELOCITY:
    (
        velocities,
        forward_wavevector_x,
        forward_wavevector_y,
        all_wheels,
        fig,
        ax_wheel,
        ax_sum,
    ) = local_velocity_analysis(
        Min_st,
        frames_to_analyse=frames_to_analyse,
        halfspan=halfspan,  # halfspan for get_velocities (ideally ~ wavelength/2)
        sampling_density=0.25,  # in pixel units
        edge=50,  # outer edge (+/-) for velocity wheel and velocity histogram
        bins_wheel=50,  # number of horizontal/vertical bins for histogram wheels
        binwidth_sum=5,  # binwidth for velocity magnitude histogram,
        kernel_size_flow=kernel_size_flow,  # building smoothening kernel needed for flow analysis
        look_ahead=1,  # for ridge advancement search: 1=forward, -1 is backward
        demo=True,
    )

    savename = outpath / f"{stackname}_velocity_wheel.png"
    fig.savefig(savename, dpi=500)
    plt.close(fig)

    # create header for csv file and define multiplication factor, if given
    if nmperpix is not None and frpermin is not None:
        unit = "nm/s"
        factor = (nmperpix) / (60 / frpermin)
    else:
        unit = "pixel"
        factor = 1
    # create header for csv file
    header = [
        f"velocities ({unit})",
        "forward_wavevector_x",
        "forward_wavevector_y",
    ]

    # create csv file and save data
    csv_file = outpath / f"{stackname}_velocities_results.csv"
    with open(csv_file, "w") as csv_f:  # will overwrite existing
        # create the csv writer
        writer = csv.writer(csv_f, delimiter="\t")
        writer.writerow(header)
        for n in range(velocities.size):
            writer.writerow(
                [
                    velocities[n] * factor,
                    forward_wavevector_x[n],
                    forward_wavevector_y[n],
                ]
            )

    if nmperpix is not None and frpermin is not None:
        print(f"Median velocity magnitude: {np.nanmedian(velocities * factor)} nm/s")

###########################################################

# Local analysis: 2-channel crest detection

elif action == Action.LOCAL_DISTANCES:
    (
        distances_DE,
        forward_wavevector_x,
        forward_wavevector_y,
        all_wheels,
        fig,
        ax_wheel,
        ax_sum,
    ) = local_DE_compare_analysis(
        MinD_st,
        MinE_st,
        frames_to_analyse=frames_to_analyse,
        halfspan=halfspan,  # halfspan for distances (ideally ~ wavelength/2)
        sampling_density=1,  # in pixel units
        edge=10,  # outer edge (+/-) for DE-shift wheel and histogram
        bins_wheel=50,  # number of horizontal/vertical bins for histogram wheels
        binwidth_sum=1,  # binwidth for distance histogram
        kernel_size_flow=kernel_size_flow,  # building smoothening kernel needed for flow analysis
        look_ahead=-1,  # for ridge advancement search: -1 is backward, E follows D
        demo=True,
    )
    savename = outpath / f"{stackname_D}_DE_distance_wheel.png"
    fig.savefig(savename, dpi=500)
    plt.close(fig)

    # create header for csv file and define multiplication factor, if given
    if nmperpix is not None:
        unit = "µm"
        factor = nmperpix / 1000
    else:
        unit = "pixel"
        factor = 1
    header = [
        f"distances_DE ({unit})",
        "forward_wavevector_x",
        "forward_wavevector_y",
    ]

    # create csv file and save data
    csv_file = outpath / f"{stackname_D}_distances_DE_results.csv"
    with open(csv_file, "w") as csv_f:  # will overwrite existing
        # create the csv writer
        writer = csv.writer(csv_f, delimiter="\t")
        writer.writerow(header)
        for n in range(distances_DE.size):
            writer.writerow(
                [
                    distances_DE[n] * factor,
                    forward_wavevector_x[n],
                    forward_wavevector_y[n],
                ]
            )

    if nmperpix is not None:
        print(f"Median DE-crest distance: {np.nanmedian(distances_DE * factor)} µm")
