"""
This file cleans an image stack by performing the following steps:
- Create an exclusion image exclude_im to flag and exclude outliers (very high/low intensity
areas, which would distort further cleaning).
- Correct for fluorescence bleaching by normalizing each frame to its mean intensity value.
- Creat an 'illumination correction map' illum_im by smoothing and averaging all movie images,
and normalizing the images to their maximum intensity.
- Create a static image object_im by averaging over all the moving (surface pattern)
features of all images in the stack. This background image then only contains static
fluorescent features such as local specks, holes, and scratches.
- Correct each image via the following image operation: tif_cln = (tif_in - object_im)/illum_im.
- Slightly smooth images to diminish effect of sharp edges and remaining artefacts.

Input: single tif image stack (3D), non-processed
Output: cleaned tif image stack (plus image showing procedure for first frame)
"""

from os.path import abspath, dirname
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

import min_analysis_tools.clean_tools_2D as clean_tools_2D
from min_analysis_tools.clean_tools_1D import outlier_flag

####################################################################################


def clean_image_stack(tif_inpath, save=True):
    """
    tif_inpath: path, full file path of tif file
    save: bool, save cleaned stack + image showing cleaning steps for first frame
    """

    # load stack:
    tif_in = io.imread(tif_inpath)

    # get an exclusion image:
    maxim = np.max(tif_in, axis=0)
    maxarr = np.squeeze(np.asarray(maxim))
    inliers, outliers, flags = outlier_flag(
        maxarr, tolerance=3, sig_change=0.7, how=1, demo=False, test=False
    )
    exclude_im = 1 - np.reshape(flags, maxim.shape, order="C")

    # get bleach correction (first order):
    st_size = tif_in.shape
    bleach_curve = np.zeros(st_size[0], dtype=float)
    for ii in range(st_size[0]):
        thisframe = tif_in[ii, :, :]
        usevals = thisframe[np.nonzero(exclude_im == 0)]
        bleach_curve[ii] = np.mean(usevals)
    bleach_curve = bleach_curve / bleach_curve[1]
    axz = np.arange(0, st_size[0], 1)
    prms = np.polyfit(axz, bleach_curve, 1)
    bleach_fit = np.polyval(prms, axz)

    # get_dirt_and_light images:
    object_im, illum_im = clean_tools_2D.get_stack_dirt_and_light(
        tif_in, exclude_im, bleach_fit, demo=False
    )

    # correct stack:
    tif_cln = clean_tools_2D.clean_stack(
        tif_in, exclude_im, object_im, illum_im, bleach_fit, demo=False
    )

    # create overview on cleaning of first frame:
    firstplane = tif_in[0, :, :]
    firstplane_cln = tif_cln[0, :, :]
    fig, axs = plt.subplots(2, 3, figsize=(11.69, 8.27))  # A4 landscape
    axs[0, 0].imshow(firstplane)
    axs[0, 0].set_title("first image")
    axs[0, 1].imshow(exclude_im)
    axs[0, 1].set_title("exclusion image")
    axs[0, 2].plot(bleach_curve, "bo-")
    axs[0, 2].plot(bleach_fit, "r-")
    axs[0, 2].set_title("masked-area bleach curve")
    axs[0, 2].set_xlabel("frame index")
    axs[0, 2].set_ylabel("relative intensity (a.u.)")
    axs[1, 0].imshow(object_im)
    axs[1, 0].set_title("stack: objects")
    axs[1, 1].imshow(illum_im)
    axs[1, 1].set_title("stack: light")
    axs[1, 2].imshow(firstplane_cln)
    axs[1, 2].set_title("first image: cleaned")
    fig.tight_layout()

    # save cleaned stack and collage of first frame:
    if save:
        tif_cln_path = tif_inpath.parent / f"cleaned_stacks"
        tif_cln_path.mkdir(exist_ok=True)
        io.imsave(
            tif_cln_path / f"{tif_inpath.stem}_cln.tif",
            tif_cln,
            check_contrast=False,
        )
        fig.savefig(tif_cln_path / f"{tif_inpath.stem}_cln.pdf")

    return tif_cln, fig, axs


####################################################################################

if __name__ == "__main__":
    tif_inpath = (
        Path(dirname(abspath(__file__))).parent / "example_data" / "demo_raw.tif"
    )
    clean_image_stack(tif_inpath)
