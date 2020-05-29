"""
    This module contains the Subgrid main class and interpolation functions
"""

import numpy as np
import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, float_me, int_me, fone
from pdfflow.functions import inner_subgrid
from pdfflow.functions import first_subgrid
from pdfflow.functions import last_subgrid


class Subgrid(tf.Module):
    """
    Wrapper class around subgrdis.
    This class reads the LHAPDF grid, parses it and stores all necessary
    information as tensorflow tensors.

    Saving this information as tf.tensors allows to reuse tf.function compiled function
    with no retracing.

    Note:
        the x and q2 arrays are padded with an extra value next to the boundaries
        to avoid out of bound errors driven by numerical precision
        The size we save for the arrays correspond to the size of the arrays before padding

    Parameters
    ----------
        grid: collections.namedtuple
            tuple containing (x, q2, flav, grid)
            which correspond to the x and q2 arrays
            the flavour scheme and the pdf grid values respectively
        i: int
            index of the subgrid
        total: int
            total number of subgrids of the family

    Attributes
    ----------
        log(x)
        log(Q2)
        Q2
        values of the pdf grid
    """

    def __init__(self, grid, i = 0, total = 0):
        super().__init__()
        # Save the boundaries of the grid
        xmin = min(grid.x)
        xmax = max(grid.x)
        self.log_xmin = float_me(np.log(xmin))
        self.log_xmax = float_me(np.log(xmax))

        q2min = min(grid.q2)
        q2max = max(grid.q2)
        self.log_q2min = float_me(np.log(q2min))
        self.log_q2max = float_me(np.log(q2max))

        # Save grid shape information
        self.s_x = int_me(grid.x.size)
        self.s_q2 = int_me(grid.q2.size)

        # Insert a padding at the beginning and the end
        log_xpad = np.pad(np.log(grid.x), 1, mode='edge')
        log_xpad[0] *= 0.99
        log_xpad[-1] *= 1.01

        log_q2pad = np.pad(np.log(grid.q2), 1, mode='edge')
        log_q2pad[0] *= 0.99
        log_q2pad[-1] *= 1.01

        self.padded_x = float_me(log_xpad)
        self.padded_q2 = float_me(log_q2pad)

        # Finally parse the grid
        # the grid is sized (x.size * q.size, flavours)
        reshaped_grid = grid.grid.reshape(grid.x.size, grid.q2.size, -1)
        # and pad it with 0s in x and q
        padded_grid = np.pad(reshaped_grid, ((1,1),(1,1),(0,0)))
        # flatten the x and q dimensions again and store it
        self.padded_grid = float_me(padded_grid.reshape(-1, grid.flav.size))

        # Depending on the number of the grid, select with interpolation function should be run
        if i == 0:
            self.fn_interpolation = first_subgrid
        elif i == total:
            self.fn_interpolation = last_subgrid
        else:
            self.fn_interpolation = inner_subgrid

    def __call__(self, pids, shape, arr_x, arr_q2):
        result = self.fn_interpolation(pids,
                shape,
                arr_x,
                arr_q2,
                self.log_xmin,
                self.log_xmax,
                self.padded_x,
                self.s_x,
                self.log_q2min,
                self.log_q2max,
                self.padded_q2,
                self.s_q2,
                self.padded_grid)
        return result
