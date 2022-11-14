"""
This file is meant to assist in finding suitable smoothing parameters for local analysis.
It will process all tif files in an input folder, running the local analysis for a low
given number of frames for different combinations of parameters, and saves the output
as an image (2D histogram or "velocity wheel") and csv for each general/flow smoothing
parameter set. Can be run in parallel over all files in the input folder (see setting in
"main" below).
Other local and general parameters can be set below in the function "screen_file".
"""

import csv
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from min_analysis_tools.get_auto_halfspan import get_auto_halfspan
from min_analysis_tools.local_velocity_analysis import local_velocity_analysis


def screen_file(stackpath):

    # general settings -> SET
    nmperpix = 594  # nanometer per pixel (assume aspect ratio 1), set to None for pixel
    frpermin = 4  # frames per minute. Set to None to stick to frames
    frames_to_analyse = 5  # analyse first ... frames per stack

    # Local analysis parameters -> SET
    halfspan = None  # halfspan for velocities / distances (ideally ~ wavelength/2)
    # set halfspan to "None" to use automatic halfspan (determined from spatial autocorrelation)
    sampling_width = 0.25  # in pixel units
    edge = 50  # outer edge(+/-) for histograms; rec.: start ~50
    bins_wheel = 50  # number of horizontal/vertical bins for histogram wheels
    binwidth_sum = 2.5  # binwidth for 1D hist.; rec.: start ~5

    Min_st = io.imread(stackpath)
    stackname = str(stackpath.stem)
    print(f"Loading {stackname}")
    outpath = stackpath.parent / "screening_results"

    for kernel_size_general in [10, 20, 30, 40, 50]:

        for kernel_size_flow in [10, 20, 30, 40, 50]:

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

            savename = (
                outpath
                / f"{stackname}_velocity_wheel_{kernel_size_general}_{kernel_size_flow}.png"
            )
            fig.tight_layout()
            fig.savefig(savename, dpi=200)
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
            csv_file = (
                outpath
                / f"{stackname}_velocities_results_{kernel_size_general}_{kernel_size_flow}.csv"
            )
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


if __name__ == "__main__":

    inpath = Path(r"INPATH_HERE")  # -> SET
    parallel = True  # -> SET

    # create outpath
    outpath = inpath / "screening_results"
    outpath.mkdir(exist_ok=True)

    if parallel:  # parallel = True -> run all files in folder in parallel
        with Pool() as p:
            p.map(screen_file, inpath.glob("**/*.tif"))
    else:  # parallel = False -> run files in folder one by one
        for stackpath in inpath.glob("**/*.tif"):
            screen_file(stackpath)
