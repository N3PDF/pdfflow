"""
    This module contains the different grids (first, last and inner) wrapper functions
    when compiled by tensorflow they all take GRID_FUNCTION_SIGNATURE as input.

    Parameters
    ----------
        GRID_FUNCTION_SIGNATURE: list
        [
                 tf.TensorSpec(shape=[None], dtype=DTYPEINT),
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
                 tf.TensorSpec(shape=[2], dtype=DTYPEINT
        ]
"""
import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, fzero, int_me
from pdfflow.region_interpolator import interpolate
from pdfflow.region_interpolator import lowx_extrapolation
from pdfflow.region_interpolator import lowq2_extrapolation
from pdfflow.region_interpolator import lowx_lowq2_extrapolation
from pdfflow.region_interpolator import highq2_extrapolation
from pdfflow.region_interpolator import lowx_highq2_extrapolation

AUTOGRAPH_OPT = tf.autograph.experimental.Feature.ALL

GRID_FUNCTION_SIGNATURE = [
    tf.TensorSpec(shape=[None], dtype=DTYPEINT), # u
    tf.TensorSpec(shape=[2], dtype=DTYPEINT), # shape
    tf.TensorSpec(shape=[None], dtype=DTYPE), # a_x
    tf.TensorSpec(shape=[None], dtype=DTYPE), # a_q2
    tf.TensorSpec(shape=[], dtype=DTYPE), # xmin
    tf.TensorSpec(shape=[], dtype=DTYPE), # xmax
    tf.TensorSpec(shape=[None], dtype=DTYPE), # padded_x
    tf.TensorSpec(shape=[], dtype=DTYPEINT), # s_x
    tf.TensorSpec(shape=[], dtype=DTYPE), # q2min
    tf.TensorSpec(shape=[], dtype=DTYPE), # q2max
    tf.TensorSpec(shape=[None], dtype=DTYPE), # padded_q2
    tf.TensorSpec(shape=[], dtype=DTYPEINT), #s_q2
    tf.TensorSpec(shape=[None, None], dtype=DTYPE), # grid
]

OPT = {
        'experimental_autograph_options' : AUTOGRAPH_OPT,
        'input_signature' : GRID_FUNCTION_SIGNATURE
        }

# Auxiliary functions
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=bool), tf.TensorSpec(shape=[None], dtype=bool)])
def _condition_to_idx(cond1, cond2):
    """ Take two boolean masks and returns the indexes in which both are true """
    full_condition = tf.logical_and(cond1, cond2)
    return full_condition, int_me(tf.where(full_condition))


@tf.function(**OPT)
def inner_subgrid(
    u,
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
    padded_grid,
):
    """
    Inner (non-first and non-last) subgrid interpolation
    Selects query points by a boolean mask
    Calls interpolate (basic interpolation)
    Calls lowx_extrapolation

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

    actual_padded = tf.gather(padded_grid, u, axis=-1)

    stripe_0 = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    stripe_1 = tf.math.logical_and(a_q2 >= log_q2min, a_q2 < log_q2max)
    stripe_2 = a_x < log_xmin

    res = fzero

    # --------------------------------------------------------------------
    # normal interpolation
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_1)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    # --------------------------------------------------------------------
    # lowx
    stripe, f_idx = _condition_to_idx(stripe_1, stripe_2)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    return res


@tf.function(**OPT)
def first_subgrid(
    u,
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
    padded_grid,
):
    """ 
    First subgrid interpolation
    Selects query points by a boolean mask
    Calls interpolate (basic interpolation)
    Calls lowx_extrapolation
    Calls lowq2_extrapolation
    Calls lowx_lowq2_extrapolation

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

    actual_padded = tf.gather(padded_grid, u, axis=-1)

    stripe_0 = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    stripe_1 = tf.math.logical_and(a_q2 >= log_q2min, a_q2 < log_q2max)
    stripe_2 = a_x < log_xmin
    stripe_3 = a_q2 < log_q2min

    res = fzero

    # --------------------------------------------------------------------
    # normal interpolation
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_1)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    # --------------------------------------------------------------------
    # lowx
    stripe, f_idx = _condition_to_idx(stripe_1, stripe_2)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    # --------------------------------------
    # low q2
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_3)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowq2_extrapolation(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    # --------------------------------------------------------------------
    # low x low q2
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_3)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_lowq2_extrapolation(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    return res

@tf.function(**OPT)
def last_subgrid(
    u,
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
    padded_grid,
):
    """
    Last subgrid interpolation.
    The values defining the query are
        u, shape, a_x, a_q2
    while the rest of the input define the subgrid
    Selects query points by a boolean mask

    The conditions in this interpolation are
    (0) = log_xmin <= a_x <= log_xmax
    (1) = log_q2min <= a_q2 <= log_q2max
    (2) = a_x < log_xmin
    (3) = a_q2 > log_q2max

    and the functions to call are
    interpolate: (0) && (1)
    lowx_extrapolation: (1) && (2)
    highq2_extrapolation: (0) && (3)
    low_x_highq2_extrapolation: (2) && (3)


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
    actual_padded = tf.gather(padded_grid, u, axis=-1)

    # Generate all conditions for all stripes
    stripe_0 = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    stripe_1 = tf.math.logical_and(a_q2 >= log_q2min, a_q2 <= log_q2max)
    stripe_2 = a_x < log_xmin
    stripe_3 = a_q2 > log_q2max

    res = fzero

    # --------------------------------------------------------------------
    # normal interpolation
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_1)
    if tf.size(f_idx) != 0:
        # Check whether there are any points in this region
        # if there are, execute normal_interpolation
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    # --------------------------------------------------------------------
    # lowx
    stripe, f_idx = _condition_to_idx(stripe_1, stripe_2)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)
    # --------------------------------------------------------------------
    # high q2
    stripe, f_idx = _condition_to_idx(stripe_0, stripe_3)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = highq2_extrapolation(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)
    # --------------------------------------------------------------------
    # low x high q2
    stripe, f_idx = _condition_to_idx(stripe_2, stripe_3)
    if tf.size(f_idx) != 0:
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_highq2_extrapolation(
            in_x,
            in_q2,
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
        res += tf.scatter_nd(f_idx, ff_f, shape)

    return res
