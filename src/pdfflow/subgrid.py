"""
    This module contains the Subgrid main class and interpolation functions
"""

import numpy as np
import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, float_me, int_me, fone
from pdfflow.neighbour_knots import four_neighbour_knots
from pdfflow.interpolations import default_bicubic_interpolation
from pdfflow.interpolations import extrapolate_linear

INTERPOLATE_SIGNATURE = [
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPEINT),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPEINT),
    tf.TensorSpec(shape=[None, None], dtype=DTYPE),
]


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


    Attributes
    ----------
        log(x)
        log(Q2)
        Q2
        values of the pdf grid
    """

    def __init__(self, grid):
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


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def interpolate(
    a_x,
    a_q2,
    log_xmin,
    log_xmax,
    padded_x,
    s_x,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """ 
    Basic Bicubic Interpolation inside the subgrid
    Four Neighbour Knots selects grid knots around each query point to
    make the interpolation: 4 knots on the x axis and 4 knots on the q2
    axis are needed for each point, plus the pdf fvalues there.
    Default bicubic interpolation performs the interpolation itself
    
    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        log_xmin: tf.tensor of shape []
            value for the lowest knot on the x axis
        log_xmax: tf.tensor of shape []
            value for the greatest knot on the x axis
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
        log_q2min: tf.tensor of shape []
            value for the lowest knot on the q2 axis
            (current subgrid)
        log_q2max: tf.tensor of shape []
            value for the greatest knot on the q2 axis
            (current subgrid)
        padded_q2: tf.tensor of shape [None]
            value for all the knots on the q2 axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when querying points near boundaries
        s_q2: tf.tensor of shape []
            size of q2 knots tensor without padding
        actual_padded: tf.tensor of shape [None,None]
            pdf values: first axis is the flattened padded (q2,x) grid,
            second axis is needed pid column (dimension depends on the query)
    """
    a0, a1, a2, a3, a4 = four_neighbour_knots(a_x, a_q2, padded_x, padded_q2, actual_padded)

    return default_bicubic_interpolation(a_x, a_q2, a0, a1, a2, a3, a4, s_x, s_q2)


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowx_extrapolation(
    a_x,
    a_q2,
    log_xmin,
    log_xmax,
    padded_x,
    s_x,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """ 
    Extrapolation in low x regime 

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        log_xmin: tf.tensor of shape []
            value for the lowest knot on the x axis
        log_xmax: tf.tensor of shape []
            value for the greatest knot on the x axis
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
        log_q2min: tf.tensor of shape []
            value for the lowest knot on the q2 axis
            (current subgrid)
        log_q2max: tf.tensor of shape []
            value for the greatest knot on the q2 axis
            (current subgrid)
        padded_q2: tf.tensor of shape [None]
            value for all the knots on the q2 axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when querying points near boundaries
        s_q2: tf.tensor of shape []
            size of q2 knots tensor without padding
        actual_padded: tf.tensor of shape [None,None]
            pdf values: first axis is the flattened padded(q2,x) grid,
            second axis is needed pid column (dimension depends on the query)
    """
    corn_x = padded_x[1:3]
    s = tf.size(a_x, out_type=DTYPEINT)

    x, q2 = tf.meshgrid(corn_x, a_q2, indexing="ij")

    y = interpolate(
        tf.reshape(x, [-1]),
        tf.reshape(q2, [-1]),
        log_xmin,
        log_xmax,
        padded_x,
        s_x,
        log_q2min,
        log_q2max,
        padded_q2,
        s_q2,
        actual_padded,
    )

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], y[:s], y[s:])


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowq2_extrapolation(
    a_x,
    a_q2,
    log_xmin,
    log_xmax,
    padded_x,
    s_x,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """ 
    Extrapolation in low q2 regime 

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        log_xmin: tf.tensor of shape []
            value for the lowest knot on the x axis
        log_xmax: tf.tensor of shape []
            value for the greatest knot on the x axis
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
        log_q2min: tf.tensor of shape []
            value for the lowest knot on the q2 axis
            (current subgrid)
        log_q2max: tf.tensor of shape []
            value for the greatest knot on the q2 axis
            (current subgrid)
        padded_q2: tf.tensor of shape [None]
            value for all the knots on the q2 axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when querying points near boundaries
        s_q2: tf.tensor of shape []
            size of q2 knots tensor without padding
        actual_padded: tf.tensor of shape [None,None]
            pdf values: first axis is the flattened padded (q2,x) grid,
            second axis is needed pid column (dimension depends on the query)
    """

    corn_q2 = tf.stack([padded_q2[1], 1.01 * padded_q2[1]], 0)

    x, q2 = tf.meshgrid(a_x, corn_q2)

    s = tf.size(a_x, out_type=DTYPEINT)

    fq2Min = interpolate(
        tf.reshape(x, [-1]),
        tf.reshape(q2, [-1]),
        log_xmin,
        log_xmax,
        padded_x,
        s_x,
        log_q2min,
        log_q2max,
        padded_q2,
        s_q2,
        actual_padded,
    )

    fq2Min1 = fq2Min[s:]
    fq2Min = fq2Min[:s]

    a_q2 = tf.math.exp(a_q2)
    corn_q2 = tf.math.exp(corn_q2[:1])

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(mask, tf.maximum(float_me(-2.5), (fq2Min1 - fq2Min) / fq2Min / 0.01), fone)
    corn_q2 = tf.expand_dims(corn_q2, 1)
    a_q2 = tf.expand_dims(a_q2, 1)

    return fq2Min * tf.math.pow(a_q2 / corn_q2, anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2)


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def highq2_extrapolation(
    a_x,
    a_q2,
    log_xmin,
    log_xmax,
    padded_x,
    s_x,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """ 
    Extrapolation in high q2 regime 

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        log_xmin: tf.tensor of shape []
            value for the lowest knot on the x axis
        log_xmax: tf.tensor of shape []
            value for the greatest knot on the x axis
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
        log_q2min: tf.tensor of shape []
            value for the lowest knot on the q2 axis
            (current subgrid)
        log_q2max: tf.tensor of shape []
            value for the greatest knot on the q2 axis
            (current subgrid)
        padded_q2: tf.tensor of shape [None]
            value for all the knots on the q2 axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when querying points near boundaries
        s_q2: tf.tensor of shape []
            size of q2 knots tensor without padding
        actual_padded: tf.tensor of shape [None,None]
            pdf values: first axis is the flattened padded (q2,x) grid,
            second axis is needed pid column (dimension depends on the query)
    """
    corn_q2 = padded_q2[-2:-4:-1]

    x, q2 = tf.meshgrid(a_x, corn_q2)
    s = tf.size(a_x, out_type=DTYPEINT)

    y = interpolate(
        tf.reshape(x, [-1]),
        tf.reshape(q2, [-1]),
        log_xmin,
        log_xmax,
        padded_x,
        s_x,
        log_q2min,
        log_q2max,
        padded_q2,
        s_q2,
        actual_padded,
    )

    return extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], y[:s], y[s:])


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowx_highq2_extrapolation(
    a_x,
    a_q2,
    log_xmin,
    log_xmax,
    padded_x,
    s_x,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """ 
    Extrapolation in high q2, low x regime 

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        log_xmin: tf.tensor of shape []
            value for the lowest knot on the x axis
        log_xmax: tf.tensor of shape []
            value for the greatest knot on the x axis
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
        log_q2min: tf.tensor of shape []
            value for the lowest knot on the q2 axis
            (current subgrid)
        log_q2max: tf.tensor of shape []
            value for the greatest knot on the q2 axis
            (current subgrid)
        padded_q2: tf.tensor of shape [None]
            value for all the knots on the q2 axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when querying points near boundaries
        s_q2: tf.tensor of shape []
            size of q2 knots tensor without padding
        actual_padded: tf.tensor of shape [None,None]
            pdf values: first axis is the flattened padded (q2,x) grid,
            second axis is needed pid column (dimension depends on the query)
    """

    corn_x = padded_x[1:3]
    corn_q2 = padded_q2[-2:-4:-1]

    x, q2 = tf.meshgrid(corn_x, corn_q2)

    f = interpolate(
        tf.reshape(x, [-1]),
        tf.reshape(q2, [-1]),
        log_xmin,
        log_xmax,
        padded_x,
        s_x,
        log_q2min,
        log_q2max,
        padded_q2,
        s_q2,
        actual_padded,
    )

    fxMin = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[:1], f[2:3])

    fxMin1 = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[1:2], f[3:])

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], fxMin, fxMin1)


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowx_lowq2_extrapolation(
    a_x,
    a_q2,
    log_xmin,
    log_xmax,
    padded_x,
    s_x,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """ 
    Extrapolation in low q2, low x regime 

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        log_xmin: tf.tensor of shape []
            value for the lowest knot on the x axis
        log_xmax: tf.tensor of shape []
            value for the greatest knot on the x axis
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
        log_q2min: tf.tensor of shape []
            value for the lowest knot on the q2 axis
            (current subgrid)
        log_q2max: tf.tensor of shape []
            value for the greatest knot on the q2 axis
            (current subgrid)
        padded_q2: tf.tensor of shape [None]
            value for all the knots on the q2 axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when querying points near boundaries
        s_q2: tf.tensor of shape []
            size of q2 knots tensor without padding
        actual_padded: tf.tensor of shape [None,None]
            pdf values: first axis is the flattened padded (q2,x) grid,
            second axis is needed pid column (dimension depends on the query)
    """
    corn_x = padded_x[1:3]
    corn_q2 = tf.stack([padded_q2[1], padded_q2[1]], 0)

    f = interpolate(
        tf.concat([corn_x, corn_x], 0),
        tf.concat([corn_q2, 1.01 * corn_q2], 0),
        log_xmin,
        log_xmax,
        padded_x,
        s_x,
        log_q2min,
        log_q2max,
        padded_q2,
        s_q2,
        actual_padded,
    )

    fq2Min = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[:1], f[1:2])

    fq2Min1 = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[2:3], f[3:])

    a_q2 = tf.expand_dims(tf.math.exp(a_q2), 1)
    corn_q2 = tf.math.exp(corn_q2[0])

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(mask, tf.maximum(float_me(-2.5), (fq2Min1 - fq2Min) / fq2Min / 0.01), fone)

    factor = tf.math.pow(a_q2 / corn_q2, anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2)

    return fq2Min * factor
