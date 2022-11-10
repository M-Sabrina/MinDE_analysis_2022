"""
This file cleans a single image stack by performing the routine "clean_image_stack.py" and
saves it to a designated output folder.
Input has to be directory of a tif file containing data for one channel and one sample position.
Parameters can be set below in the main section (below function "single_clean").
"""

from pathlib import Path

from skimage import io

from min_analysis_tools.clean_image_stack import clean_image_stack

###############################################################################


def single_clean(inpath, autosave=True, outpath=None, savedpi=300):
    # Load stack:
    Min_st = io.imread(inpath)
    # Clean stack:
    Min_st_cln, fig, axs = clean_image_stack(Min_st)
    # If autosave is true, create outpath as subfolder:
    if autosave:
        outpath = inpath.parent / f"cleaned_stacks"
    # Save stack and collage of first frame:
    outpath.mkdir(exist_ok=True)
    io.imsave(
        outpath / f"{inpath.stem}_cln.tif",
        Min_st_cln,
        check_contrast=False,
    )
    fig.savefig(outpath / f"{inpath.stem}_cln.png", dpi=savedpi)


###############################################################################

if __name__ == "__main__":
    # Define paths & saving option -> SET
    inpath = Path(r"INPUT_PATH_HERE/my_file.tif")  # set file path incl. name
    autosave = False  # if True, create subfolder in inpath to store results
    outpath = Path(r"OUTPUT_PATH_HERE")  # disregarded if autosave is True

    # Collage saving parameters -> SET
    savedpi = 300

    # Run cleaning
    single_clean(inpath, autosave, outpath, savedpi)
