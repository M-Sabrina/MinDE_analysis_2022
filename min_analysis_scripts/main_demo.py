from enum import Enum, IntEnum
from os.path import abspath, dirname
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from min_analysis_tools import correlation_tools, get_data
from min_analysis_tools.get_auto_halfspan import get_auto_halfspan
from min_analysis_tools.local_DE_compare_analysis import local_DE_compare_analysis
from min_analysis_tools.local_velocity_analysis import local_velocity_analysis


# Options for actions to perform (choose below):
class Action(Enum):
    GLOBAL_SPATIAL = 1  # -> spatial autocorrelation
    GLOBAL_TEMPORAL = 2  # -> temporal autocorrelation
    LOCAL_VELOCITY = 3  # -> local velocity analysis
    LOCAL_DISTANCES = 4  # -> local dual-channel crest comparison -> selection 5 only


# Options for exemplary dataset to use (choose below):
class Selection(Enum):
    SIMULATION = 0  # simple simulated spiral
    SPIRAL = 1  # Min spiral (example data)
    NORTHWEST = 2  # Min northwest-directed traveling waves (example data)
    STITCH_HORIZONTAL = 3  # Min large horizontally stitched pattern (example data)
    STITCH_SQUARE = 4  # Min large square stitched pattern (example data)
    DUAL_SPIRAL = 5  # Min spiral dataset for dual-channel analysis (example data)


# Options for output graphics (choose below):
class Demo(IntEnum):
    DEFAULT = 1  # Jupyter notebook (default)
    DIAGNOSIS = 2  # diagnosis (includes user interrupts for click-to-code break)
    DEEP = 3  # as DIAGNOSIS, shelved (deep loops etc.)


# Choose action HERE (see options in class "Action" above):
action = Action.LOCAL_DISTANCES

# Choose example dataset HERE (see options in class "Selection" above):
selection = Selection.SPIRAL

# Choose level of output graphics HERE (see options in class "Demo" above):
demo = Demo.DEFAULT

# General / display parameters
size = 512  # data will be downsized to this (set "None" to skip)
kernel_size = 15  # kernel size for initial image smoothening (set "None" to skip)
frames_to_analyse = 5  # set at a very large number to analyse all frames

# Global temporal autocorrelation analysis parameters
reps_per_kymostack = 5  # pick ... kymographs around middle
kymoband = 0.8  # analyse middle ... part of image

# Local analysis parameters
halfspan = None  # halfspan for velocities / distances (ideally ~ wavelength/2)
# set halfspan to "None" to use automatic halfspan (determined from spatial autocorrelation)
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
    stack_path = stem / "example_data" / "real_spiral.tif"
    stackname = "spiral"
elif (
    selection == Selection.NORTHWEST
):  # Min northwest-directed traveling waves (example data)
    stack_path = stem / "example_data" / "real_northwest.tif"
    stackname = "northwest"
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
elif (
    selection == Selection.DUAL_SPIRAL
):  # Min example dataset for dual-channel analysis
    stack_path = stem / "example_data" / "real_dual_E.tif"
    stackname = "dual_spiral"

if action == action.LOCAL_DISTANCES:
    selection = Selection.DUAL_SPIRAL
    stack_path_D = stem / "example_data" / "real_dual_D.tif"
    stack_path_E = stem / "example_data" / "real_dual_E.tif"
    print("Selection set to dual channel example")
    MinE_st, fig_E, ax_E = get_data.load_stack(
        stack_path_E, size=size, kernel_size=kernel_size, demo=True
    )
    ax_E.set_title("MinE stack")
    fig_E.show()
    MinD_st, fig_D, ax_D = get_data.load_stack(
        stack_path_D, size=size, kernel_size=kernel_size, demo=True
    )
    ax_D.set_title("MinD stack")
    fig_D.show()
else:
    Min_st, fig, ax = get_data.load_stack(
        stack_path, size=size, kernel_size=kernel_size, demo=True
    )
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
        peak_valley_diff,
        fig,
        ax,
    ) = correlation_tools.analyze_radial_profiles(crmx_storage)
    fig.show()

    print(f"mean position of first valley: {np.mean(first_min_pos):.02f}")
    print(f"mean position of first peak (!): {np.mean(first_max_pos):.02f}")
    print(f"mean peak-valley difference: {np.mean(peak_valley_diff):.02f}")

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
            peak_valley_diff,
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
            tmp_peak_valley_diff = peak_valley_diff
        if axis == "y":  # second time: join arrays from x- and y-slices
            first_min_pos = np.append(tmp_first_min_pos, first_min_pos)
            first_min_val = np.append(tmp_first_min_val, first_min_val)
            first_max_pos = np.append(tmp_first_max_pos, first_max_pos)
            first_max_val = np.append(tmp_first_max_val, first_max_val)
            peak_valley_diff = np.append(tmp_peak_valley_diff, peak_valley_diff)

    print(f"mean position of first valley: {np.mean(first_min_pos):.02f}")
    print(f"mean position of first peak (!): {np.mean(first_max_pos):.02f}")
    print(f"mean peak-valley difference: {np.mean(peak_valley_diff):.02f}")

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
        sampling_density=1,  # in pixel units
        edge=5,  # outer edge (+/-) for DE-shift wheel and velocity histogram
        bins_wheel=50,  # number of horizontal/vertical bins for histogram wheels
        binwidth_sum=1,  # binwidth for distance histogram
        kernel_size_flow=35,  # building smoothening kernel needed for flow analysis
        look_ahead=-1,  # for ridge advancement search: -1 is backward, assume E follows D
        demo=demo,
    )
    fig.show()


###########################################################

print("Press any key to end demo")
input()
plt.close("all")
