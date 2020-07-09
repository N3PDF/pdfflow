"""
    This module contains the Subgrid main class.

    Upon instantiation the subgrid class will select one of the
    interpolation functions defined in :py:module:`functions` and
    compile it using the `GRID_FUNCTION_SIGNATURE` defined in this function.
"""

import numpy as np
import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, float_me, int_me
from pdfflow.alphaS_functions import first_alphaS_subgrid
from pdfflow.alphaS_functions import inner_alphaS_subgrid
from pdfflow.alphaS_functions import last_alphaS_subgrid

# Compilation signature and options of the subgrid functions
ALPHAS_GRID_FUNCTION_SIGNATURE = [
    tf.TensorSpec(shape=[1], dtype=DTYPEINT),  # shape
    tf.TensorSpec(shape=[None], dtype=DTYPE),  # a_q2
    tf.TensorSpec(shape=[], dtype=DTYPE),  # q2min
    tf.TensorSpec(shape=[], dtype=DTYPE),  # q2max
    tf.TensorSpec(shape=[None], dtype=DTYPE),  # padded_q2
    tf.TensorSpec(shape=[], dtype=DTYPEINT),  # s_q2
    tf.TensorSpec(shape=[None], dtype=DTYPE),  # grid
]
AUTOGRAPH_OPT = tf.autograph.experimental.Feature.ALL
OPT = {
    "experimental_autograph_options": AUTOGRAPH_OPT,
    "input_signature": GRID_FUNCTION_SIGNATURE,
}


class AlphaS_Subgrid(tf.Module):
    """
    Wrapper class around alphaS subgrdis.
    This class reads the LHAPDF alphaS grid, parses it and stores all necessary
    information as tensorflow tensors.

    Saving this information as tf.tensors allows to reuse tf.function compiled function
    with no retracing.

    Note:
        the q2 array is padded with an extra value next to the boundaries
        to avoid out of bound errors driven by numerical precision
        The size we save for the array corresponds to the size of the array before padding

    Parameters
    ----------
        grid: collections.namedtuple
            tuple containing (x, alphaS grid)
            which corresponds to the q2 array and
            the alphaS grid values respectively
        i: int
            index of the subgrid
        total: int
            total number of subgrids of the family
        compile_functions: bool
            whether to tf-compile the interpolation function(default True)

    Attributes
    ----------
        log(Q2)
        Q2
        values of the alphaS grid
    """

    def __init__(self, grid, i=0, total=0, compile_functions=True):
        super().__init__()
        # Save the boundaries of the grid
        q2min = min(grid.q2)
        q2max = max(grid.q2)
        self.log_q2min = float_me(np.log(q2min))
        self.log_q2max = float_me(np.log(q2max))

        # Save grid shape information
        self.s_q2 = int_me(grid.q2.size)

        # Insert a padding at the beginning and the end
        log_q2pad = np.pad(np.log(grid.q2), 1, mode="edge")
        log_q2pad[0] *= 0.99
        log_q2pad[-1] *= 1.01

        self.padded_q2 = float_me(log_q2pad)

        # Finally parse the grid
        # the grid is sized (q.size), pad it with 0s
        padded_grid = np.pad(grid.grid, (1, 1))

        # Depending on the index of the grid, select which interpolation function should be run
        if i == 0:
            self.fn_interpolation = first_alphaS_subgrid
        elif i == (total - 1):
            self.fn_interpolation = last_alphaS_subgrid
        else:
            self.fn_interpolation = inner_alphaS_subgrid

        self.name_sg = f"grid_{i}"

        if compile_functions:
            self.fn_interpolation = tf.function(self.fn_interpolation, **OPT)

    def __call__(self, shape, arr_q2):
        result = self.fn_interpolation(
            shape,
            arr_q2,
            self.log_q2min,
            self.log_q2max,
            self.padded_q2,
            self.s_q2,
            self.padded_grid,
        )
        return result
