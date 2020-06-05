"""
Contains the extrapolation and interpolation functions wrappers for the different regions
"""

import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, float_me, fone
from pdfflow.neighbour_knots import four_neighbour_knots
from pdfflow.interpolations import default_bicubic_interpolation
from pdfflow.interpolations import extrapolate_linear

INTERPOLATE_SIGNATURE = [
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPEINT),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPEINT),
    tf.TensorSpec(shape=[None, None], dtype=DTYPE),
]


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def interpolate(
    a_x, a_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
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
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
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
    x_bins, q2_bins, corn_x, corn_q2, pdf_vals = four_neighbour_knots(
        a_x, a_q2, padded_x, padded_q2, actual_padded
    )

    return default_bicubic_interpolation(
        a_x, a_q2, x_bins, q2_bins, corn_x, corn_q2, pdf_vals, s_x, s_q2
    )


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowx_extrapolation(
    a_x, a_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
):
    """
    Extrapolation in low x regime

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
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
        padded_x,
        s_x,
        padded_q2,
        s_q2,
        actual_padded,
    )

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], y[:s], y[s:])


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowq2_extrapolation(
    a_x, a_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
):
    """
    Extrapolation in low q2 regime

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
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
        padded_x,
        s_x,
        padded_q2,
        s_q2,
        actual_padded,
    )

    fq2Min1 = fq2Min[s:]
    fq2Min = fq2Min[:s]

    a_q2 = tf.math.exp(a_q2)
    corn_q2 = tf.math.exp(corn_q2[:1])

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(
        mask, tf.maximum(float_me(-2.5), (fq2Min1 - fq2Min) / fq2Min / 0.01), fone
    )
    corn_q2 = tf.expand_dims(corn_q2, 1)
    a_q2 = tf.expand_dims(a_q2, 1)

    return fq2Min * tf.math.pow(
        a_q2 / corn_q2, anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2
    )


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def highq2_extrapolation(
    a_x, a_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
):
    """
    Extrapolation in high q2 regime

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
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
        padded_x,
        s_x,
        padded_q2,
        s_q2,
        actual_padded,
    )

    return extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], y[:s], y[s:])


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowx_highq2_extrapolation(
    a_x, a_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
):
    """
    Extrapolation in high q2, low x regime

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
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
        padded_x,
        s_x,
        padded_q2,
        s_q2,
        actual_padded,
    )

    fxMin = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[:1], f[2:3])

    fxMin1 = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[1:2], f[3:])

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], fxMin, fxMin1)


@tf.function(input_signature=INTERPOLATE_SIGNATURE)
def lowx_lowq2_extrapolation(
    a_x, a_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
):
    """
    Extrapolation in low q2, low x regime

    Parameters
    ----------
        a_x: tf.tensor of shape [None]
            query of values of log(x)
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        padded_x: tf.tensor of shape [None]
            value for all the knots on the x axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when queryingpoints near boundaries
        s_x: tf.tensor of shape []
            size of x knots tensor without padding
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
        padded_x,
        s_x,
        padded_q2,
        s_q2,
        actual_padded,
    )

    fq2Min = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[:1], f[1:2])

    fq2Min1 = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[2:3], f[3:])

    a_q2 = tf.expand_dims(tf.math.exp(a_q2), 1)
    corn_q2 = tf.math.exp(corn_q2[0])

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(
        mask, tf.maximum(float_me(-2.5), (fq2Min1 - fq2Min) / fq2Min / 0.01), fone
    )

    factor = tf.math.pow(a_q2 / corn_q2, anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2)

    return fq2Min * factor
