"""
Spatial autocorrelation analysis is performed on a set number of images per movie.
For each autocorrelation output image, a radial profile is recorded starting from
the main central correlation peak. The resulting spatial radial correlation curves 
are subjected to maxima analysis. The first maximum after radius R=0 indicates the
most predominant distance between wave edges, irrespective of propagation direction.
This distance is identified as the pattern's global wavelength.

Input: directory with pre-cleaned movies as tif files (3D).
Output: png files for radial autocorrelation curves, csv file for characteristic
parameters.

Characteristic parameters are:
    "valley_pos" - position of first minimum in um
    "valley_val" - amplitude of first minimum
    "peak_pos" - position of first maximum in um --> wavelength
    "peak_val" - amplitude of first maximum

Reference: Cees Dekker Lab; project: MinDE; researcher: Sabrina Meindlhumer.
Code designed & written by Jacob Kerssemakers and Sabrina Meindlhumer, 2022.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from skimage import io

from min_analysis_tools import correlation_tools

# settings -> SET
nmperpix = None  # nanometer per pixel (assume aspect ratio 1), set to None for pixel
frames_to_analyse = 10  # analyse first ... frames per stack

# define inpath  -> SET
inpath = Path(r"INPUT_PATH_HERE")

# create outpath
outpath = inpath.parent / f"{inpath.stem}_correlation_spatial"
if not outpath.is_dir():
    outpath.mkdir()

# create csv file for saving each file's mean-averaged characteristic parameters later
csv_file = outpath / "spatial_autocorrelation_results.csv"
# recognize unit
if nmperpix is None:
    unit = "pixels"
else:
    unit = "µm"
# create header for csv file
header = [
    "filename",
    "frame_number",
    f"valley_pos ({unit})",  # position of first minimum
    "valley_val",  # amplitude of first minimum
    f"peak_pos ({unit})",  # position of first maximum --> wavelength
    "peak_val",  # amplitude of first maximum
]
with open(csv_file, "w") as csv_f:  # will overwrite existing
    # create the csv writer
    writer = csv.writer(csv_f, delimiter="\t")
    writer.writerow(header)


for stack in inpath.glob("**/*.tif"):  # find all tif files in inpath

    # load stack (tif file)
    MinDE_st_full = io.imread(stack)
    stackname = str(stack.stem)
    print(f"Loading {stackname}")

    # autocorrelation --> obtain array of autocorrelation matrixes for analyzed frames
    (
        crmx_storage,
        fig,
        ax_corr,
        ax_orig,
    ) = correlation_tools.get_spatial_correlation_matrixes(
        MinDE_st_full,
        frames_to_analyse,
        demo=True,
    )
    example_fig_path = outpath / f"{stackname}_spatial_autocorr.png"
    fig.savefig(example_fig_path, dpi=500)
    plt.close(fig)

    # calculate radially averaged profile traces, analyze them with respect to first min and max
    (
        first_min_pos,
        first_min_val,
        first_max_pos,
        first_max_val,
        fig,
        ax,
    ) = correlation_tools.analyze_radial_profiles(
        crmx_storage,
        nmperpix,
        demo=True,
    )
    profile_fig_path = outpath / f"{stackname}_radialprofiles.png"
    fig.savefig(profile_fig_path, dpi=500)
    plt.close(fig)

    # save characteristic variables to csv file in outpath
    with open(csv_file, "a") as csv_f:  # will append to file
        # create the csv writer
        writer = csv.writer(csv_f, delimiter="\t")
        # write data to the csv file
        for n in range(first_max_pos.size):
            writer.writerow(
                [
                    stackname,
                    n + 1,
                    first_min_pos[n],
                    first_min_val[n],
                    first_max_pos[n],
                    first_max_val[n],
                ]
            )
