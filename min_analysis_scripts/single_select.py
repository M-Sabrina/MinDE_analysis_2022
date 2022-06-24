"""
This file serves as a tool for quick analysis of a given Min pattern stack.
It is thus similar to the notebook "Quickstart_Min_analysis.ipynb", with differs
from it insofar as it will save the output (png and csv files) in a directory
"results" within the set inpath.

Input: Pre-cleaned Min pattern stack (tif file, 3D)
Output: Png and csv files with characteristic paramters, depending on analysis

To start, pick one out of 4 provided actions from the class "Actions" below
by setting the path to the file in the variable "inpath" (or separately in 
"in_E" and "in_D" for local distance analysis).
Set the input parameters for either global or local analysis, then run the script.
See comments below for more details on the individual parameters.

Reference: Cees Dekker Lab; project: MinDE; researcher: Sabrina Meindlhumer.
Code designed & written by Jacob Kerssemakers and Sabrina Meindlhumer, 2022.
"""

import csv
from enum import IntEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from min_analysis_tools import correlation_tools
from min_analysis_tools.get_auto_halfspan import get_auto_halfspan
from min_analysis_tools.local_DE_compare_analysis import local_DE_compare_analysis
from min_analysis_tools.local_velocity_analysis import local_velocity_analysis

# Plot settings:
plt.rc("font", size=10)  # controls default text size
plt.rc("axes", titlesize=10)  # fontsize of the title
plt.rc("axes", labelsize=10)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
plt.rc("legend", fontsize=6)  # fontsize of the legend
plt.rcParams.update({"font.family": "arial"})
cm = 1 / 2.54  # centimeters in inches (for matplotlib figure size)
fig_size = (12 * cm, 5 * cm)


class Action(IntEnum):
    GLOBAL_SPATIAL = 1  # -> spatial autocorrelation
    GLOBAL_TEMPORAL = 2  # -> temporal autocorrelation
    LOCAL_VELOCITY = 3  # -> local velocity analysis
    LOCAL_DISTANCES = 4  # -> local 2-channel crest comparison -> 2 input files required


# Choose action HERE (see list in class "Action" above):
action = Action.LOCAL_DISTANCES

# General / display parameters:
frames_to_analyse = 5  # integer; set to a very large number to analyse all frames

# Stack information (set both to None to work with pixel and frames only):
nmperpix = None  # nanometer per pixel (assuming aspect ratio = 1)
frpermin = None  # frames per minute

# Temporal autocorrelation analysis parameters:
reps_per_kymostack = 5  # pick ... kymographs around middle
kymoband = 0.8  # analyse middle ... part of image

# Local analysis parameters:
halfspan = None  # halfspan for velocities / distances (ideally ~ wavelength/2)
# set halfspan to "None" to use automatic halfspan (determined from spatial autocorrelation)
sampling_width = 0.25  # in pixel units
edge = 30  # outer edge(+/-) for histograms; start with ~50 for vel. / ~10 for DE-shift
bins_wheel = 50  # number of horizontal/vertical bins for histogram wheels
binwidth_sum = 2.5  # binwidth for 1D hist
kernel_size_general = 15  # kernel for first smoothing step
kernel_size_flow = 35  # building smoothening kernel needed for flow analysis

# Define stack_path -> SET
# single file (actions 1, 2, 3)
infile = Path(r"INPUT_PATH_HERE\my_file.tif")
# two files (action 4)
in_E = Path(r"INPUT_PATH_HERE\my_file_E.tif")
in_D = Path(r"INPUT_PATH_HERE\my_file_D.tif")

###########################################################

# load stack
if action == Action.LOCAL_DISTANCES:
    MinE_st = io.imread(in_E)
    stackname_E = in_E.stem
    MinD_st = io.imread(in_D)
    stackname_D = in_D.stem
    outpath = in_E.parent / f"results"
else:
    Min_st = io.imread(infile)
    stackname = infile.stem
    outpath = infile.parent / f"results"

# create outpath
if not outpath.is_dir():
    outpath.mkdir()

###########################################################

## Global analysis: Spatial autocorrelation

if action == Action.GLOBAL_SPATIAL:

    # recognize unit
    if nmperpix is None:
        unit = "pixels"
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
    fig.set_size_inches(fig_size)
    fig.tight_layout()
    fig.savefig(example_fig_path, dpi=500)
    plt.close(fig)

    # calculate radially averaged profile traces
    # analyze them with respect to first min and max, mean-average
    (
        first_min_pos,
        first_min_val,
        first_max_pos,
        first_max_val,
        fig,
        ax,
    ) = correlation_tools.analyze_radial_profiles(crmx_storage, nmperpix)
    profile_fig_path = outpath / f"{stackname}_radialprofiles.png"
    fig.set_size_inches(fig_size)
    fig.tight_layout()
    fig.savefig(profile_fig_path, dpi=500)
    plt.close(fig)

    print(f"mean position of first valley: {np.nanmean(first_min_pos):.02f} {unit}")
    print(f"mean position of first peak (!): {np.nanmean(first_max_pos):.02f} {unit}")

    # create csv file for saving characteristic parameters later
    csv_file = outpath / f"{stackname}_spatial_autocorrelation_results.csv"
    # create header for csv file
    header = [
        "frame_nr",  # frame number
        f"valley_pos ({unit})",  # position of first minimum
        "valley_val",  # amplitude of first minimum
        f"peak_pos ({unit})",  # position of first maximum --> wavelength
        "peak_val",  # amplitude of first maximum
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
                ]
            )

###########################################################

## Global analysis: Temporal autocorrelation

elif action == Action.GLOBAL_TEMPORAL:

    # recognize unit
    if frpermin is None:
        unit = "frames"
    else:
        unit = "s"

    MinDE_shift_tx = np.moveaxis(Min_st, 0, -1)  # creates t-x resliced frames
    MinDE_shift_yt = np.moveaxis(Min_st, -1, 0)  # creates y-t resliced frames
    ff, rr, cc = np.shape(MinDE_shift_yt)
    MinDE_shift_ty = np.empty((ff, cc, rr))
    for frame in range(ff):  # transpose images to create t-y resliced frames
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
    ]
    with open(csv_file_path, "w") as csv_f:  # will overwrite existing
        # create the csv writer
        writer = csv.writer(csv_f, delimiter="\t")
        writer.writerow(header)

    # for t-x slices or for t-y slices
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
        example_fig_path = outpath / f"{stackname}_t{axis}_temporal_autocorr.png"
        fig.set_size_inches(fig_size)
        fig.tight_layout()
        fig.savefig(example_fig_path, dpi=500)
        plt.close(fig)

        # take trace in first row of each correlation matrix
        # analyze it with respect to first min and max, mean-average
        (
            first_min_pos,
            first_min_val,
            first_max_pos,
            first_max_val,
            fig,
            ax,
        ) = correlation_tools.analyze_temporal_profiles(
            axis,
            crmx_storage,
            slices2analyze,
            frpermin,
        )
        profile_fig_path = outpath / f"{stackname}_t{axis}_profiles.png"
        fig.set_size_inches(fig_size)
        fig.tight_layout()
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
                        ]
                    )
            # first time: store values for averaging later
            tmp_first_min_pos = first_min_pos
            tmp_first_min_val = first_min_val
            tmp_first_max_pos = first_max_pos
            tmp_first_max_val = first_max_val

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
                        ]
                    )

            # second time: join arrays from x- and y-slices and output average to console
            first_min_pos = np.append(tmp_first_min_pos, first_min_pos)
            first_min_val = np.append(tmp_first_min_val, first_min_val)
            first_max_pos = np.append(tmp_first_max_pos, first_max_pos)
            first_max_val = np.append(tmp_first_max_val, first_max_val)

    print(f"mean position of first valley: {np.nanmean(first_min_pos):.02f} {unit}")
    print(f"mean position of first peak (!): {np.nanmean(first_max_pos):.02f} {unit}")

###########################################################

## Local analysis: Velocity wheel

elif action == Action.LOCAL_VELOCITY:

    if halfspan is None:
        halfspan = get_auto_halfspan(Min_st, frames_to_analyse)
        print(f"Auto-halfspan determined: {halfspan} pixels")

    (
        velocities,
        forward_wavevector_x,
        forward_wavevector_y,
        wheels,
        crests_x,
        crests_y,
        framenr,
        max_x1,
        max_y1,
        max_x2,
        max_y2,
        fig,
        ax_wheel,
        ax_sum,
    ) = local_velocity_analysis(
        Min_st,
        frames_to_analyse,
        halfspan,  # halfspan for get_velocities (ideally ~ wavelength/2)
        sampling_width,  # in pixel units
        edge,  # outer edge (+/-) for velocity wheel and velocity histogram
        bins_wheel,  # number of horizontal/vertical bins for histogram wheels
        binwidth_sum,  # binwidth for velocity magnitude histogram,
        kernel_size_general,  # kernel for first smoothing step
        kernel_size_flow,  # kernel for additional smoothing step
        look_ahead=1,  # for ridge advancement search: 1=forward, -1 is backward
        demo=True,
    )

    savename = outpath / f"{stackname}_velocity_wheel.png"
    fig.set_size_inches(fig_size)
    fig.tight_layout()
    fig.savefig(savename, dpi=500)
    plt.close(fig)

    # create header for csv file and define multiplication factor, if given
    if nmperpix is not None and frpermin is not None:
        unit = "nm/s"
        factor = (nmperpix) / (60 / frpermin)
    else:
        unit = "pixels/frame"
        factor = 1
    # create header for csv file
    header = [
        f"velocities ({unit})",
        "forward_wavevector_x",
        "forward_wavevector_y",
        "crests_x (pixel position)",
        "crests_y (pixel position)",
        "frame_nr",
        f"trace_max_1 ({sampling_width}*pixels)",
        "intensity_max_1",
        f"trace_max_2 ({sampling_width}*pixels)",
        "intensity_max_2",
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
                    crests_x[n],
                    crests_y[n],
                    framenr[n],
                    max_x1[n],
                    max_y1[n],
                    max_x2[n],
                    max_y2[n],
                ]
            )

    if nmperpix is not None and frpermin is not None:
        print(
            f"Median velocity magnitude: {np.nanmedian(velocities * factor):.02f} nm/s"
        )

###########################################################

# Local analysis: 2-channel crest detection
# TO-DO

elif action == Action.LOCAL_DISTANCES:

    if halfspan is None:
        halfspan = get_auto_halfspan(MinE_st, frames_to_analyse)
        print(f"Auto-halfspan determined: {halfspan} pixel")

    (
        distances_DE,
        forward_wavevector_x,
        forward_wavevector_y,
        all_wheels,
        crests_x,
        crests_y,
        framenr,
        max_x1,
        max_y1,
        max_x2,
        max_y2,
        fig,
        ax_wheel,
        ax_sum,
    ) = local_DE_compare_analysis(
        MinD_st,
        MinE_st,
        frames_to_analyse,
        halfspan,  # halfspan for distances (ideally ~ wavelength/2)
        sampling_width,  # in pixel units
        edge,  # outer edge (+/-) for DE-shift wheel and histogram
        bins_wheel,  # number of horizontal/vertical bins for histogram wheels
        binwidth_sum,  # binwidth for distance histogram
        kernel_size_general,  # kernel for additional smoothing step
        kernel_size_flow,  # building smoothening kernel needed for flow analysis
        look_ahead=-1,  # for ridge advancement search: -1 is backward, E follows D
        demo=True,
    )
    savename = outpath / f"{stackname_D}_DE_distance_wheel.png"
    fig.set_size_inches(fig_size)
    fig.tight_layout()
    fig.savefig(savename, dpi=500)
    plt.close(fig)

    # create header for csv file and define multiplication factor, if given
    if nmperpix is not None:
        unit = "µm"
        factor = nmperpix / 1000
    else:
        unit = "pixels"
        factor = 1
    header = [
        f"distances_DE ({unit})",
        "forward_wavevector_x",
        "forward_wavevector_y",
        "crests_x (pixel position)",
        "crests_y (pixel position)",
        "frame_nr",
        f"trace_max_1 ({sampling_width}*pixels)",
        "intensity_max_1",
        f"trace_max_2 ({sampling_width}*pixels)",
        "intensity_max_2",
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
                    crests_x[n],
                    crests_y[n],
                    framenr[n],
                    max_x1[n],
                    max_y1[n],
                    max_x2[n],
                    max_y2[n],
                ]
            )

    if nmperpix is not None:
        print(
            f"Median DE-crest distance: {np.nanmedian(distances_DE * factor):.02f} µm"
        )
