"""
    This module contains the different grids (first, last and inner) wrapper functions
    when compiled by tensorflow they all take GRID_FUNCTION_SIGNATURE as input
    which is defined in :py:module:`ppdfflow.subgrid`
    and will be compiled once they are linked to a specific subgrid.

    The function in this module apply different masks to the input to generate
    the different interpolation zones:

    (0) = log_xmin <= a_x <= log_xmax
    (1) = log_q2min <= a_q2 <= log_q2max
    (2) = a_x < log_xmin (low x)
    (3) = a_q2 > log_q2max (high q2)
    (4) = a_q2 < log_q2max (low q2)

    The input values defining the query are
        u, shape, a_x, a_q2
    while the rest of the input define the subgrid.
    The points are selected by a boolean mask

    and the functions to call depending on the zone are:
    interpolate: (0) && (1)
    lowx_extrapolation: (1) && (2)
    highq2_extrapolation: (0) && (3)
    lowq2_extrapolation: (0) && (4)
    low_x_highq2_extrapolation: (2) && (3)
    lowx_lowq2_extrapolation: (2) && (4)

"""
import tensorflow as tf
from pdfflow.configflow import DTYPE, int_me
from pdfflow.region_interpolator import interpolate
from pdfflow.region_interpolator import lowx_extrapolation
from pdfflow.region_interpolator import lowq2_extrapolation
from pdfflow.region_interpolator import lowx_lowq2_extrapolation
from pdfflow.region_interpolator import highq2_extrapolation
from pdfflow.region_interpolator import lowx_highq2_extrapolation

# Auxiliary functions
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=bool),
        tf.TensorSpec(shape=[None], dtype=bool),
    ]
)
def _condition_to_idx(cond1, cond2):
    """ Take two boolean masks and returns the indexes in which both are true """
    full_condition = tf.logical_and(cond1, cond2)
    return full_condition, int_me(tf.where(full_condition))


def inner_subgrid(
    shape,
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
    Inner (non-first and non-last) subgrid interpolation
    Calls
    interpolate (basic interpolation) (0) && (1)
    lowx_extrapolation (1) && (2)

    Parameters
    ----------
        u: tf.tensor of shape [None]
            query of pids
        shape: tf.tensor of shape [None,None]
            final output shape to scatter points into
        For other parameters refer to subgrid.py:interpolate

    Returns
    ----------
        tf.tensor of shape [None,None]
        pdf interpolated values for each query point and quey pids
    """
    stripe_0 = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    stripe_1 = tf.math.logical_and(a_q2 >= log_q2min, a_q2 < log_q2max)
    stripe_2 = a_x < log_xmin

    res = tf.zeros(shape, dtype=DTYPE)

    # --------------------------------------------------------------------
    # normal interpolation
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_1)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,)
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # lowx
    stripe, f_idx = _condition_to_idx(stripe_1, stripe_2)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(
            in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
        )
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    return res


def first_subgrid(
    shape,
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
    First subgrid interpolation
    Calls
    interpolate (basic interpolation) (0) && (1)
    lowx_extrapolation (1) && (2)
    lowq2_extrapolation (0) && (4)
    lowx_lowq2_extrapolation (2) && (4)

    Parameters
    ----------
        u: tf.tensor(int)
            list of pids to query
        ...
        shape: tf.tensor(int, int)
            final output shape to scatter points into

        For other parameters refer to subgrid.py:interpolate

    Returns
    ----------
        tf.tensor of shape `shape`
        pdf interpolated values for each query point and quey pids
    """
    stripe_0 = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    stripe_1 = tf.math.logical_and(a_q2 >= log_q2min, a_q2 < log_q2max)
    stripe_2 = a_x < log_xmin
    stripe_4 = a_q2 < log_q2min

    res = tf.zeros(shape, dtype=DTYPE)

    # --------------------------------------------------------------------
    # normal interpolation
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_1)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,)
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # lowx
    stripe, f_idx = _condition_to_idx(stripe_1, stripe_2)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(
            in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
        )
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------
    # low q2
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_4)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowq2_extrapolation(
            in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
        )
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # low x low q2
    stripe, f_idx = _condition_to_idx(stripe_2, stripe_4)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_lowq2_extrapolation(
            in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
        )
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    return res


def last_subgrid(
    shape,
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
    Last subgrid interpolation.
    Calls
    interpolate: (0) && (1)
    lowx_extrapolation: (1) && (2)
    highq2_extrapolation: (0) && (3)

    Parameters
    ----------
        u: tf.tensor, rank-1
            grid of pid being queried
        shape: tf.tensor, rank-1 shape: (2,)
            final output shape to scatter points into

        For other parameters see :py:func:`pdfflow.region_interpolator.interpolate`

    Returns
    ----------
        tf.tensor, rank-2, shape: shape
            pdf interpolated values for each query point and quey pids
    """
    # Generate all conditions for all stripes
    stripe_0 = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    stripe_1 = tf.math.logical_and(a_q2 >= log_q2min, a_q2 <= log_q2max)
    stripe_2 = a_x < log_xmin
    stripe_3 = a_q2 > log_q2max

    res = tf.zeros(shape, dtype=DTYPE)

    # --------------------------------------------------------------------
    # normal interpolation
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_1)
    if tf.size(f_idx) != 0:
        # Check whether there are any points in this region
        # if there are, execute normal_interpolation
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,)
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # lowx
    stripe, f_idx = _condition_to_idx(stripe_1, stripe_2)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(
            in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
        )
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # high q2
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_3)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = highq2_extrapolation(
            in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
        )
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    # --------------------------------------------------------------------
    # low x high q2
    stripe, f_idx = _condition_to_idx(stripe_2, stripe_3)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_highq2_extrapolation(
            in_x, in_q2, padded_x, s_x, padded_q2, s_q2, actual_padded,
        )
        res = tf.tensor_scatter_nd_update(res, f_idx, ff_f)

    return res
