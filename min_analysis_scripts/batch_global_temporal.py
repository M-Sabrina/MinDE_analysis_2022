"""
For temporal correlation, we generate several x-t or y-t kymographs per movie evenly
distributed over a set middle fraction fraction of an image. For each such kymograph, 
autocorrelation analysis is performed.
The Δx=0 or Δy=0 line of these y-t or x-t autocorrelation maps then in effect represents a
temporal correlation curve averaged over all the original image points on this line. Next,
these correlation curves are averaged between different kymographs. Thus, the final correlation
curve in effect represents the average temporal correlation signal sampled from selected surface
locations. Analogous to the spatial correlation analysis, the first maximum after Δt=0 indicates
a main oscillation period.

Input: directory with pre-cleaned movies as tif files (3D).
Output: png files for autocorrelation curves, csv file for characteristic parameters.

Characteristic parameters are:
    "valley_pos" - position of first minimum in seconds
    "valley_val" - amplitude of first minimum
    "peak_pos" - position of first maximum in seconds --> oscillation period
    "peak_val" - amplitude of first maximum

Reference: Cees Dekker Lab; project: MinDE; researcher: Sabrina Meindlhumer.
Code designed & written by Jacob Kerssemakers and Sabrina Meindlhumer, 2022.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from skimage import io

from min_analysis_tools import correlation_tools
from min_analysis_tools.batch_reslice_stack import reslice_stack

# parameters -> SET
frpermin = None  # frames per minute. Set to None to stick to frames
reps_per_kymostack = 5  # pick ... kymographs around middle
kymoband = 0.8  # analyse middle ... part of image

# define inpath -> SET
inpath = Path(r"INPUT_PATH_HERE")

# recognize unit
if frpermin is None:
    unit = "frame"
else:
    unit = "s"

# reslice stacks
reslicepath = reslice_stack(inpath)
print(f"Resliced files saved to {str(reslicepath)}")

# create outpath
outpath = inpath.parent / f"{inpath.stem}_correlation_temporal"
if not outpath.is_dir():
    outpath.mkdir()

# create csv file for saving characteristic parameters later
csv_file_path = outpath / f"temporal_autocorrelation_results.csv"
# create header for csv file
header = [
    "filename",
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

# perform temporal analysis for resliced stacks
for stack in inpath.glob("**/*.tif"):  # find all tif files in `inpath`

    # for t-x slices or for t-y slices
    for axis in ["x", "y"]:

        # load resliced stack as np array
        stackname = str(stack.stem)
        reslicename = f"{stackname}_resliced_{axis}.tif"
        reslicefile = reslicepath / reslicename
        MinDE_st_resliced = io.imread(reslicefile)

        print(f"Loading {reslicename}")
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
            demo=True,
        )
        example_fig_path = outpath / f"{stackname}_t{axis}_temporal_autocorr.png"
        fig.savefig(example_fig_path, dpi=500)
        plt.close(fig)

        # take trace in first row of each correlation matrix, analyze first min and max
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
            demo=True,
        )
        profile_fig_path = outpath / f"{stackname}_t{axis}_profiles.png"
        fig.savefig(profile_fig_path, dpi=500)
        plt.close(fig)

        # save for tx slices
        if axis == "x":
            with open(csv_file_path, "a") as csv_f:  # append to existing
                # create the csv writer
                writer = csv.writer(csv_f, delimiter="\t")
                # write data to the csv file
                for n in range(first_max_pos.size):
                    writer.writerow(
                        [
                            stackname,
                            "all",  # all x values along line for constant y
                            slices2analyze[n] + 1,  # y (resliced stack framenr)
                            first_min_pos[n],
                            first_min_val[n],
                            first_max_pos[n],
                            first_max_val[n],
                        ]
                    )
        # save for ty slices
        elif axis == "y":
            with open(csv_file_path, "a") as csv_f:  # append to existing
                # create the csv writer
                writer = csv.writer(csv_f, delimiter="\t")
                # write data to the csv file
                for n in range(first_max_pos.size):
                    writer.writerow(
                        [
                            stackname,
                            slices2analyze[n] + 1,  # x (resliced stack framenr)
                            "all",  # all y values along line for constant x
                            first_min_pos[n],
                            first_min_val[n],
                            first_max_pos[n],
                            first_max_val[n],
                        ]
                    )
