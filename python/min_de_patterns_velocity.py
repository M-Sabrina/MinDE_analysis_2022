# -*- coding: utf-8 -*-
"""
Function specific for MinDE pattern analysis.
Created on Wed Oct 27 11:42:23 2021
@author: jkerssemakers
"""
import matplotlib.pyplot as plt
import numpy as np


def work_wheel(
    velocities,
    forward_wavevector_x,
    forward_wavevector_y,
    edge=100,
    bins=50,
    demo=1,
):
    """
    Build and analyze a 2D histogram of local velocities.
    """
    vx = velocities * forward_wavevector_x
    vy = velocities * forward_wavevector_y

    # blank out center
    nonzero_ix = np.argwhere(abs(velocities) > edge / 20)
    vx = vx[nonzero_ix[:, 0]]
    vy = vy[nonzero_ix[:, 0]]

    xax = np.linspace(-edge, edge, bins)
    yax = np.linspace(-edge, edge, bins)
    wheel, vxedges, vyedges = np.histogram2d(vx, vy, bins=(xax, yax))

    # test interrupt: 3 or higher=shelved, 2=activate
    if demo == 3:
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        plt.imshow(
            wheel,
            interpolation="nearest",
            origin="lower",
            extent=[vxedges[0], vxedges[-1], vyedges[0], vyedges[-1]],
        )
        ax.set_box_aspect(1)
        plt.title("velocity histogram")
        plt.xlabel("x-velocity (pixel/frame)")
        plt.ylabel("y-velocity (pixel/frame)")
        plt.show()
        breakpoint()

    return wheel, vxedges, vyedges
