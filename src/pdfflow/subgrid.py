"""
    This module contains the Subgrid main class.

    Upon instantiation the subgrid class will select one of the
    interpolation functions defined in :py:module:`functions` and
    compile it using the `GRID_FUNCTION_SIGNATURE` defined in this function.
"""

import numpy as np
import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, float_me, int_me
from pdfflow.functions import inner_subgrid
from pdfflow.functions import first_subgrid
from pdfflow.functions import last_subgrid
from pdfflow.alphas_functions import alphas_first_subgrid
from pdfflow.alphas_functions import alphas_inner_subgrid
from pdfflow.alphas_functions import alphas_last_subgrid

# Compilation signature and options of the subgrid functions
GRID_FUNCTION_SIGNATURE = [
    tf.TensorSpec(shape=[2], dtype=DTYPEINT),  # shape
    tf.TensorSpec(shape=[None], dtype=DTYPE),  # a_x
    tf.TensorSpec(shape=[None], dtype=DTYPE),  # a_q2
    tf.TensorSpec(shape=[], dtype=DTYPE),  # xmin
    tf.TensorSpec(shape=[], dtype=DTYPE),  # xmax
    tf.TensorSpec(shape=[None], dtype=DTYPE),  # padded_x
    tf.TensorSpec(shape=[], dtype=DTYPEINT),  # s_x
    tf.TensorSpec(shape=[], dtype=DTYPE),  # q2min
    tf.TensorSpec(shape=[], dtype=DTYPE),  # q2max
    tf.TensorSpec(shape=[None], dtype=DTYPE),  # padded_q2
    tf.TensorSpec(shape=[], dtype=DTYPEINT),  # s_q2
    tf.TensorSpec(shape=[None, None], dtype=DTYPE),  # grid
]

ALPHAS_GRID_FUNCTION_SIGNATURE = [
    tf.TensorSpec(shape=[], dtype=DTYPEINT),  # shape
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
        compile_functions: bool
            whether to tf-compile the interpolation function(default True)
        alpha_s: bool
            whether the function to compile is for a PDF grid or an alpha_s grid

    Attributes
    ----------
        log(x)
        log(Q2)
        Q2
        values of the pdf grid
    """

    def __init__(self, grid, i=0, total=0, compile_functions=True, alpha_s=False):
        name_sg = f"grid_{i}"
        self.alpha_s = alpha_s
        if alpha_s:
            name_sg += "_alpha"
        self.name_sg = name_sg
        super().__init__(name=f"Parent_{name_sg}")
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

        # Depending on whether it is an alphs_s grid or a pdf grid
        # we might need to change some options

        compilation_options = OPT.copy()

        if alpha_s:
            # the grid is sized (q.size), pad it with 0s
            self.padded_grid = float_me(np.pad(grid.grid, (1, 1)))

            if i == 0:
                self.fn_interpolation = alphas_first_subgrid
            elif i == (total - 1):
                self.fn_interpolation = alphas_last_subgrid
            else:
                self.fn_interpolation = alphas_inner_subgrid

            # Change the function signature to that of alpha_s
            compilation_options["input_signature"] = ALPHAS_GRID_FUNCTION_SIGNATURE
        else:
            # If this is a pdf grid, save also the x information
            xmin = min(grid.x)
            xmax = max(grid.x)
            self.log_xmin = float_me(np.log(xmin))
            self.log_xmax = float_me(np.log(xmax))

            self.s_x = int_me(grid.x.size)

            log_xpad = np.pad(np.log(grid.x), 1, mode="edge")
            log_xpad[0] *= 0.99
            log_xpad[-1] *= 1.01

            self.padded_x = float_me(log_xpad)

            # Finally parse the grid
            # the grid is sized (x.size * q.size, flavours)
            reshaped_grid = grid.grid.reshape(grid.x.size, grid.q2.size, -1)

            # and pad it with 0s in x and q
            padded_grid = np.pad(reshaped_grid, ((1, 1), (1, 1), (0, 0)))
            # flatten the x and q dimensions again and store it
            self.padded_grid = float_me(padded_grid.reshape(-1, grid.flav.size))

            # Depending on the index of the grid, select with interpolation function should be run
            if i == 0:
                self.fn_interpolation = first_subgrid
            elif i == (total - 1):
                self.fn_interpolation = last_subgrid
            else:
                self.fn_interpolation = inner_subgrid


        if compile_functions:
            self.fn_interpolation = tf.function(self.fn_interpolation, **compilation_options)

    def __call__(self, shape, arr_q2, pids=None, arr_x=None):
        if self.alpha_s:
            if pids is not None or arr_x is not None:
                raise ValueError("alpha_s interpolation does not accept x-input or flavours")

            result = self.fn_interpolation(
                shape,
                arr_q2,
                self.log_q2min,
                self.log_q2max,
                self.padded_q2,
                self.s_q2,
                self.padded_grid,
            )
        else:
            padded_grid = tf.gather(self.padded_grid, pids, axis=-1, name=self.name_sg)
            result = self.fn_interpolation(
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
                padded_grid,
            )
        return result
