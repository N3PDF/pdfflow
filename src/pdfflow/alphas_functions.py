"""
    This module contains the different alphas grids (first, last and inner) wrapper functions
    when compiled by tensorflow they all take alphas_GRID_FUNCTION_SIGNATURE as input
    which is defined in :py:module:`ppdfflow.alphas_subgrid`
    and will be compiled once they are linked to a specific subgrid.

    The function in this module apply different masks to the input to generate
    the different interpolation zones:

    (0) = log_q2min <= a_q2 <= log_q2max (log cubic interpolation)
    (1) = a_q2 > log_q2max (high q2, freezing the last grid value)
    (2) = a_q2 < log_q2max (low q2, logarithmic extrapolation)

    The input values defining the query are
        shape, a_q2
    while the rest of the input define the subgrid.
    The points are selected by a boolean mask

    and the functions to call depending on the zone are:
    alphas_interpolate: (0)
    alphas_lowq2_extrapolation: (2)

"""
import tensorflow as tf
from pdfflow.configflow import DTYPE, int_me
from pdfflow.alphas_region_interpolator import alphas_interpolate
from pdfflow.functions import _condition_to_idx

def alphas_inner_subgrid(
    shape,
    a_q2,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """
    Inner (non-first and non-last) alphas subgrid interpolation
    Calls
    alphas_interpolate (basic interpolation) (0)

    Parameters
    ----------
        shape: tf.tensor of shape [None]
            final output shape to scatter points into
        For other parameters refer to subgrid.py:alphas_interpolate

    Returns
    ----------
        tf.tensor of shape `shape`
        alphas interpolated values for each query point
    """
    res = tf.zeros(shape, dtype=DTYPE)

    # --------------------------------------------------------------------
    # normal interpolation

    stripe, f_idx = _condition_to_idx(a_q2 >= log_q2min, a_q2 < log_q2max)
    if tf.math.equal(f_idx, 0) is not None:
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = alphas_interpolate(in_q2, padded_q2, s_q2, actual_padded)
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)
    return res


def alphas_first_subgrid(
    shape,
    a_q2,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """
    First subgrid interpolation
    Calls
    alphas_interpolate (basic interpolation) (0)

    Parameters
    ----------
        shape: tf.tensor of shape [None]
            final output shape to scatter points into

        For other parameters refer to subgrid.py:alphas_interpolate

    Returns
    ----------
        tf.tensor of shape `shape`
        alphas interpolated values for each query point
    """
    res = tf.zeros(shape, dtype=DTYPE)

    # --------------------------------------------------------------------
    # normal interpolation

    stripe, f_idx = _condition_to_idx(a_q2 >= log_q2min, a_q2 < log_q2max)
    if tf.math.equal(f_idx, 0) is not None:
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = alphas_interpolate(in_q2, padded_q2, s_q2, actual_padded)
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # lowq2

    stripe = a_q2 < log_q2min
    f_idx = int_me(tf.where(stripe))
    if tf.math.equal(f_idx, 0) is not None:
        in_q2 = tf.boolean_mask(a_q2, stripe)
        m = tf.math.log(actual_padded[2]/actual_padded[1])\
            /(padded_q2[2] - padded_q2[1])
        
        ff_f = actual_padded[1] * tf.math.pow(
                            tf.math.exp(in_q2)/tf.math.exp(padded_q2[1]),
                            m)
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    return res


def alphas_last_subgrid(
    shape,
    a_q2,
    log_q2min,
    log_q2max,
    padded_q2,
    s_q2,
    actual_padded,
):
    """
    Last subgrid interpolation.
    Calls
    alphas_interpolate: (0)

    Parameters
    ----------
        shape: tf.tensor of shape [None]
            final output shape to scatter points into

        For other parameters see :py:func:`pdfflow.alphas_region_interpolator.alphas_interpolate`

    Returns
    ----------
        tf.tensor, shape: `shape`
            alphas interpolated values for each query point
    """
    # Generate all conditions for all stripes
    res = tf.zeros(shape, dtype=DTYPE)

    # --------------------------------------------------------------------
    # normal interpolation
    stripe, f_idx = _condition_to_idx(a_q2 >= log_q2min, a_q2 <= log_q2max)
    if tf.math.equal(f_idx, 0) is not None:
        # Check whether there are any points in this region
        # if there are, execute normal_interpolation
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = alphas_interpolate(in_q2, padded_q2, s_q2, actual_padded)
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # high q2
    stripe = a_q2 > log_q2max
    f_idx = int_me(tf.where(stripe))
    if tf.math.equal(f_idx, 0) is not None:
        ff_f = tf.ones_like(f_idx[:,0], dtype=DTYPE)*actual_padded[-2]
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    return res
