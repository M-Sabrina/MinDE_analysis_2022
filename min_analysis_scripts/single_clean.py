"""
This file cleans a single image stack by performing the routine "clean_image_stack.py" and
saves it to a designated output folder.
Input has to be directory of a tif file containing data for one channel and one sample position.
"""

from pathlib import Path

from skimage import io

from min_analysis_tools.clean_image_stack import clean_image_stack

# Define inpath & outpath -> SET
inpath = Path(r"INPUT_PATH_HERE/my_file.tif")
outpath = Path(r"OUTPUT_PATH_HERE/my_file.tif")

# Collage saving parameters
savedpi = 300

# Load stack:
Min_st = io.imread(inpath)

# Clean stack:
Min_st_cln, fig, axs = clean_image_stack(Min_st, autosave=False)

# Save stack and collage of first frame:
outpath.mkdir(exist_ok=True)
io.imsave(
    outpath / f"{inpath.stem}_cln.tif",
    Min_st_cln,
    check_contrast=False,
)
fig.savefig(outpath / f"{inpath.stem}_cln.png", dpi=savedpi)
