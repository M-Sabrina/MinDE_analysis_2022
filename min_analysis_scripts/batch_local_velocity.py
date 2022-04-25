"""
MinDE pattern local velocity analysis

Local velocity analysis is performed by
1. identifying individual wave crest positions, using optical flow analysis (using Horn-
   Schunck algorithm)
2. calculating the peak shift from one frame to the next, relying on multi-point parabolic
   fitting of intensity traces in direction of wave propagation for both frames

Input: directory with pre-cleaned movies as tif files (3D).
Output: png files showing results in 2D and 1D histogram, csv file for velocities
(magnitudes and unit vector components).

Reference: Cees Dekker Lab; project: MinDE; researcher: Sabrina Meindlhumer.
Code designed & written by Jacob Kerssemakers and Sabrina Meindlhumer, 2022.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from min_analysis_tools.get_auto_halfspan import get_auto_halfspan
from min_analysis_tools.local_velocity_analysis import local_velocity_analysis

# define inpath  -> SET
inpath = Path(r"INPUT_PATH_HERE")

# general settings -> SET
nmperpix = None  # nanometer per pixel (assume aspect ratio 1), set to None for pixel
frpermin = None  # frames per minute. Set to None to stick to frames
frames_to_analyse = 10  # analyse first ... frames per stack

# Local analysis parameters -> SET
halfspan = None  # halfspan for velocities / distances (ideally ~ wavelength/2)
# set halfspan to "None" to use automatic halfspan (determined from spatial autocorrelation)
sampling_density = 0.25  # in pixel units
edge = 30  # outer edge(+/-) for histograms; rec.: start ~50
bins_wheel = 50  # number of horizontal/vertical bins for histogram wheels
binwidth_sum = 2.5  # binwidth for 1D hist.; rec.: start ~5
kernel_size_general = 20  # kernel for first smoothing step
kernel_size_flow = 50  # building smoothening kernel needed for flow analysis

###########################################################

# create outpath
outpath = inpath.parent / f"{inpath.stem}_local_velocity"
if not outpath.is_dir():
    outpath.mkdir()

# create csv file for saving each file's mean-averaged characteristic parameters later
csv_file = outpath / "local_velocity_results.csv"
# recognize unit
if nmperpix is not None and frpermin is not None:
    unit = "nm/s"
    factor = nmperpix / (60 / frpermin)
else:
    unit = "pixels/frame"
    factor = 1
# create header for csv file
header = [
    "filename",
    f"velocities ({unit})",
    "forward_wavevector_x",
    "forward_wavevector_y",
]
with open(csv_file, "w") as csv_f:  # will overwrite existing
    # create the csv writer
    writer = csv.writer(csv_f, delimiter="\t")
    writer.writerow(header)

###########################################################

## Local analysis: Velocity wheel
for stack in inpath.glob("**/*.tif"):  # find all tif files in inpath

    # load stack (tif file)
    Min_st = io.imread(stack)
    stackname = str(stack.stem)
    print(f"Loading {stackname}")

    if halfspan is None:
        halfspan_tmp = get_auto_halfspan(Min_st, frames_to_analyse)
        print(f"Auto-halfspan determined: {halfspan_tmp} pixels")

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
        frames_to_analyse,  # analyse first ... frames
        halfspan_tmp,  # halfspan for get_velocities (ideally ~ wavelength/2)
        sampling_density,  # in pixel units
        edge,  # outer edge (+/-) for velocity wheel and velocity histogram
        bins_wheel,  # number of horizontal/vertical bins for histogram wheels
        binwidth_sum,  # binwidth for velocity magnitude histogram,
        kernel_size_general,  # kernel for additional smoothing step
        kernel_size_flow,  # building smoothening kernel needed for flow analysis
        look_ahead=1,  # for ridge advancement search: 1=forward, -1 is backward
        demo=True,
    )
    savename = outpath / f"{stackname}_velocity_wheel.png"
    fig.savefig(savename, dpi=500)
    plt.close(fig)

    # save data to csv file
    with open(csv_file, "a") as csv_f:  # will append to existing
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
