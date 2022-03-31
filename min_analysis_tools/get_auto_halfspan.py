def get_auto_halfspan(Min_st, frames_to_analyse, verbose):
    """
    Estimates a suitable value for halfspan to perform local analysis.
    Input: Min stack (3D numpy array), number of frames to analyse (int).
    """
    import numpy as np

    from min_analysis_tools.correlation_tools import (
        analyze_radial_profiles,
        get_spatial_correlation_matrixes,
    )

    crmx_storage = get_spatial_correlation_matrixes(
        Min_st, frames_to_analyse, demo=False, verbose=verbose
    )

    # calculate radially averaged profile traces
    # analyze them with respect to first min and max, mean-average
    (
        first_min_pos,
        first_min_val,
        first_max_pos,
        first_max_val,
        peak_valley_diff,
    ) = analyze_radial_profiles(crmx_storage, demo=False)

    return int(np.mean(first_min_pos))
