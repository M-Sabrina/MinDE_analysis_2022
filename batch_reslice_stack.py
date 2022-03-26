from pathlib import Path

import numpy as np
from skimage import io


def reslice_stack(inpath: Path):
    """
    This function tilts patterns. For quicker analysis, xy-t stacks are tilted
    tx-y and ty-x, such that each saved frame is essentially a kymograph.
    Input: directory with pre-cleaned movies.
    Output: per movie, an tx and a ty movie.
    Reference: Cees Dekker Lab; project: MinDE; researcher: Sabrina Meindlhumer.
    Code designed & written in MatLab by Jacob Kerssemakers, 2016.
    Rewritten to Python by Sabrina Meindlhumer, 2022.
    """

    # find file and path names
    outpath = inpath.parent / f"{inpath.stem}_resliced"
    if not outpath.is_dir():
        outpath.mkdir()

    for stack in inpath.glob("**/*.tif"):  # find all tif files in `inpath`
        # load stack as np array
        print(f"Reslicing {str(stack)}")
        MinDE_st_full = io.imread(stack)
        stackname = str(stack.stem)

        # create new file names for resliced stacks
        shift_tx_file = outpath / f"{stackname}_resliced_x.tif"
        shift_ty_file = outpath / f"{stackname}_resliced_y.tif"

        # tilt and save stack
        MinDE_shift_tx = np.moveaxis(MinDE_st_full, 0, -1)
        io.imsave(shift_tx_file, MinDE_shift_tx, check_contrast=False)

        # tilt stack
        MinDE_shift_yt = np.moveaxis(MinDE_st_full, -1, 0)
        # transpose images to have t axis in 1st dimension
        ff, rr, cc = np.shape(MinDE_shift_yt)
        MinDE_shift_ty = np.empty((ff, cc, rr))
        for frame in range(ff):
            MinDE_shift_ty[frame, :, :] = np.transpose(MinDE_shift_yt[frame, :, :])
        # save stack
        io.imsave(shift_ty_file, MinDE_shift_ty, check_contrast=False)

    print("Reslicing complete")
    return outpath
