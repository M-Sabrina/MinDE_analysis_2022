"""
This file serves as a demonstration of the tools provided within this package,
showing results for simulated as well as real example data.
Hence, it fulfills a similar function as the DEMO Jupyter notebooks, but
in a script format and with a stronger focus on functionality rather than
illustration.

The user can choose from the provided example data sets (class "Selection")
and choose one out of four analysis types to be performed on the data set
(class "Action"). The possible actions are Global Spatial Analysis, Global
Temporal Analysis, Local Velocity Analysis and Local MinDE-Distance Analysis.
Input data is provided in the folder "example_data".
To start the demonstation, scroll down and set parameters as well as data
set and action, then run the script.

Reference: Cees Dekker Lab; project: MinDE; researcher: Sabrina Meindlhumer.
Code designed & written by Jacob Kerssemakers and Sabrina Meindlhumer, 2022.
"""

from enum import IntEnum
from os.path import abspath, dirname
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from min_analysis_tools import correlation_tools, get_data
from min_analysis_tools.get_auto_halfspan import get_auto_halfspan
from min_analysis_tools.local_DE_compare_analysis import local_DE_compare_analysis
from min_analysis_tools.local_velocity_analysis import local_velocity_analysis


# Options for actions to perform (choose below):
class Action(IntEnum):
    GLOBAL_SPATIAL = 1  # -> spatial autocorrelation
    GLOBAL_TEMPORAL = 2  # -> temporal autocorrelation
    LOCAL_VELOCITY = 3  # -> local velocity analysis
    LOCAL_DISTANCES = 4  # -> local dual-channel crest comparison -> selection 5 only


# Options for exemplary dataset to use (choose below):
class Selection(IntEnum):
    SIMULATION = 0  # simple simulated spiral
    SPIRAL = 1  # Min spiral (example data)
    SOUTHEAST = 2  # Min southeast-directed traveling waves (example data)
    WEST = 3  # Min west-directed traveling waves (example data) -> dual example
    STITCH_HORIZONTAL = 4  # Min large horizontally stitched pattern (example data)
    STITCH_SQUARE = 5  # Min large square stitched pattern (example data)


# Options for output graphics (choose below):
class Demo(IntEnum):
    DEFAULT = 1  # Jupyter notebook (default)
    DIAGNOSIS = 2  # diagnosis (includes user interrupts for click-to-code break)
    DEEP = 3  # as DIAGNOSIS, shelved (deep loops etc.)


# Choose action HERE (see options in class "Action" above):
action = Action.GLOBAL_SPATIAL

# Choose example dataset HERE (see options in class "Selection" above):
selection = Selection.SPIRAL

# Choose level of output graphics HERE (see options in class "Demo" above):
demo = Demo.DEFAULT

# General parameters
frames_to_analyse = 5  # set at a very large number to analyse all frames

# Global temporal autocorrelation analysis parameters
reps_per_kymostack = 5  # pick ... kymographs around middle
kymoband = 0.8  # analyse middle ... part of image

# Local analysis parameters
halfspan = None  # halfspan for velocities / distances (ideally ~ wavelength/2)
# set halfspan to "None" to use automatic halfspan (determined from spatial autocorrelation)
kernel_size_general = 15  # kernel for first smoothing step
kernel_size_flow = 35  # building smoothening kernel needed for flow analysis

###########################################################

# load correct stack(s) from example_data directory

stem = Path(dirname(abspath(__file__))).parent

if selection == Selection.SIMULATION:  # simple simulated spiral
    stackname = "simulation"
    Min_st, fig, ax = get_data.generate_pattern(
        lambda_t=50, lambda_x=1, N_frames=10, demo=True
    )
    fig.show()
elif selection == Selection.SPIRAL:  # Min spiral (example data)
    stack_path = stem / "example_data" / "demo_spiral.tif"
    stackname = "spiral"
elif (
    selection == Selection.SOUTHEAST
):  # Min northwest-directed traveling waves (example data)
    stack_path = stem / "example_data" / "demo_southeast.tif"
    stackname = "southeast"
elif (
    selection == Selection.WEST
):  # Min northwest-directed traveling waves (example data)
    stack_path = stem / "example_data" / "paper_west_E.tif"
    stackname = "west"
elif (
    selection == Selection.STITCH_HORIZONTAL
):  # Min large horizontally stitched pattern (example data)
    stack_path = stem / "example_data" / "real_horizontal_stitch.tif"
    stackname = "stitch_horizontal"
elif (
    selection == Selection.STITCH_SQUARE
):  # Min large square stitched pattern (example data)
    stack_path = stem / "example_data" / "real_square_stitch.tif"
    stackname = "stitch_square"

if action == action.LOCAL_DISTANCES:
    stack_path_D = stem / "example_data" / "paper_west_D.tif"
    stack_path_E = stem / "example_data" / "paper_west_E.tif"
    print("Selection set to dual channel example")
    MinE_st = io.imread(stack_path_E)
    fig_E, ax_E = plt.subplots(1, 1)
    ax_E.imshow(MinE_st[0, :, :])
    ax_E.set_title("pattern")
    ax_E.set_xlabel("x (pixels)")
    ax_E.set_ylabel("y (pixels)")
    ax_E.set_title("MinE stack")
    fig_E.show()
    MinD_st = io.imread(stack_path_D)
    fig_D, ax_D = plt.subplots(1, 1)
    ax_D.imshow(MinD_st[0, :, :])
    ax_D.set_title("pattern")
    ax_D.set_xlabel("x (pixels)")
    ax_D.set_ylabel("y (pixels)")
    ax_D.set_title("MinD stack")
    fig_D.show()
else:
    Min_st = io.imread(stack_path)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(Min_st[0, :, :])
    ax.set_title("pattern")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.show()

###########################################################

## Global analysis: Spatial autocorrelation

if action == Action.GLOBAL_SPATIAL:
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
    fig.show()

    # calculate radially averaged profile traces
    # analyze them with respect to first min and max, mean-average
    (
        first_min_pos,
        first_min_val,
        first_max_pos,
        first_max_val,
        fig,
        ax,
    ) = correlation_tools.analyze_radial_profiles(crmx_storage)
    fig.show()

    print(f"mean position of first valley: {np.mean(first_min_pos):.02f} pixels")
    print(f"mean position of first peak (!): {np.mean(first_max_pos):.02f} pixels")

###########################################################

## Global analysis: Temporal autocorrelation

elif action == Action.GLOBAL_TEMPORAL:

    MinDE_shift_tx = np.moveaxis(Min_st, 0, -1)  # creates t-x resliced frames
    MinDE_shift_yt = np.moveaxis(Min_st, -1, 0)  # creates y-t resliced frames
    ff, rr, cc = np.shape(MinDE_shift_yt)
    MinDE_shift_ty = np.empty((ff, cc, rr))
    for frame in range(ff):  # tranpose images to create t-y resliced frames
        MinDE_shift_ty[frame, :, :] = np.transpose(MinDE_shift_yt[frame, :, :])

    # for t-x slices or for t-y slices
    for axis in ["x", "y"]:

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
        fig.show()

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
        )
        fig.show()

        # keep values for averaging x- and y-slices
        if axis == "x":  # first time: just store values for now
            tmp_first_min_pos = first_min_pos
            tmp_first_min_val = first_min_val
            tmp_first_max_pos = first_max_pos
            tmp_first_max_val = first_max_val
        if axis == "y":  # second time: join arrays from x- and y-slices
            first_min_pos = np.append(tmp_first_min_pos, first_min_pos)
            first_min_val = np.append(tmp_first_min_val, first_min_val)
            first_max_pos = np.append(tmp_first_max_pos, first_max_pos)
            first_max_val = np.append(tmp_first_max_val, first_max_val)

    print(f"mean position of first valley: {np.mean(first_min_pos):.02f} frames")
    print(f"mean position of first peak (!): {np.mean(first_max_pos):.02f} frames")

###########################################################

## Local analysis: Velocity wheel

elif action == Action.LOCAL_VELOCITY:

    if halfspan is None:
        halfspan = get_auto_halfspan(Min_st, frames_to_analyse, verbose=True)
        print(f"Auto-halfspan determined: {halfspan} pixels")

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
        kernel_size_general=kernel_size_general,  # kernel for first smoothing step
        kernel_size_flow=kernel_size_flow,  # kernel for additional smoothing step
        look_ahead=1,  # for ridge advancement search: 1=forward, -1 is backward
        demo=demo,
    )
    fig.show()

###########################################################

# Local analysis: 2-channel crest detection

elif action == Action.LOCAL_DISTANCES:

    if halfspan is None:
        halfspan = get_auto_halfspan(MinE_st, frames_to_analyse)
        print(f"Auto-halfspan determined: {halfspan} pixels")

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
        sampling_density=0.25,  # in pixel units
        edge=10,  # outer edge (+/-) for DE-shift wheel and velocity histogram
        bins_wheel=50,  # number of horizontal/vertical bins for histogram wheels
        binwidth_sum=1,  # binwidth for distance histogram
        kernel_size_general=kernel_size_general,  # kernel for first smoothing step
        kernel_size_flow=kernel_size_flow,  # building smoothening kernel needed for flow analysis
        look_ahead=-1,  # for ridge advancement search: -1 is backward, assume E follows D
        demo=demo,
    )
    fig.show()


###########################################################

print("Press any key to end demo")
input()
plt.close("all")
