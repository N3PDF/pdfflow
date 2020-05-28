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
from pdfflow.configflow import DTYPE, DTYPEINT, fzero
from pdfflow.subgrid import interpolate
from pdfflow.subgrid import lowx_extrapolation
from pdfflow.subgrid import lowq2_extrapolation
from pdfflow.subgrid import lowx_lowq2_extrapolation
from pdfflow.subgrid import highq2_extrapolation
from pdfflow.subgrid import lowx_highq2_extrapolation

empty_fn = lambda: fzero

GRID_FUNCTION_SIGNATURE = [tf.TensorSpec(shape=[None], dtype=DTYPEINT),
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
                 tf.TensorSpec(shape=[2], dtype=DTYPEINT)]


@tf.function(input_signature=GRID_FUNCTION_SIGNATURE)
def inner_subgrid(u, a_x, a_q2,
                  log_xmin, log_xmax, padded_x, s_x,
                  log_q2min, log_q2max, padded_q2, s_q2,
                  padded_grid, shape):
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

    # --------------------------------------------------------------------
    # normal interpolation
    stripe = tf.math.logical_and(stripe_0, stripe_1)
    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(in_x, in_q2,
                           log_xmin, log_xmax, padded_x, s_x,
                           log_q2min, log_q2max, padded_q2, s_q2,
                           actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    res = tf.cond(idx0 == 0, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # lowx
    stripe = tf.math.logical_and(a_x < log_xmin, stripe_1)
    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, gen_stripe)
        in_q2 = tf.boolean_mask(a_q2, gen_stripe)
        ff_f = lowx_extrapolation(in_x, in_q2,
                                  log_xmin, log_xmax, padded_x, s_x,
                                  log_q2min, log_q2max,  padded_q2, s_q2,
                                  actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    return res + tf.cond(idx0 == 0, empty_fn, gen_fun)

@tf.function(input_signature=GRID_FUNCTION_SIGNATURE)
def first_subgrid(u, a_x, a_q2,
                  log_xmin, log_xmax, padded_x, s_x,
                  log_q2min, log_q2max, padded_q2, s_q2,
                  padded_grid, shape):
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

    # --------------------------------------------------------------------
    # normal interpolation
    stripe = tf.math.logical_and(stripe_0, stripe_1)
    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(in_x, in_q2,
                           log_xmin, log_xmax, padded_x, s_x,
                           log_q2min, log_q2max, padded_q2, s_q2,
                           actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    res = tf.cond(idx0 == 0, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # lowx
    stripe = tf.math.logical_and(stripe_2, stripe_1)
    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(in_x, in_q2,
                                  log_xmin, log_xmax, padded_x, s_x,
                                  log_q2min, log_q2max,  padded_q2, s_q2,
                                  actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    res += tf.cond(idx0 == 0, empty_fn, gen_fun)
    
    #--------------------------------------
    # low q2
    stripe_3 = a_q2 < log_q2min
    stripe = tf.math.logical_and(stripe_0, stripe_3)

    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowq2_extrapolation(in_x, in_q2,
                                   log_xmin, log_xmax, padded_x, s_x,
                                   log_q2min, log_q2max, padded_q2, s_q2,
                                   actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    res += tf.cond(idx0 == 0, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # low x low q2
    stripe = tf.math.logical_and(stripe_2, stripe_3)
    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_lowq2_extrapolation(in_x, in_q2,
                                        log_xmin, log_xmax, padded_x, s_x,
                                        log_q2min, log_q2max, padded_q2, s_q2,
                                        actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    return res + tf.cond(idx0 == 0, empty_fn, gen_fun)

@tf.function(input_signature=GRID_FUNCTION_SIGNATURE)
def last_subgrid(u, a_x, a_q2,
                 log_xmin, log_xmax, padded_x, s_x,
                 log_q2min, log_q2max, padded_q2, s_q2,
                 padded_grid, shape):
    """ 
    Last subgrid interpolation
    Selects query points by a boolean mask
    Calls interpolate (basic interpolation)
    Calls lowx_extrapolation
    Calls highq2_extrapolation
    Calls low_x_highq2_extrapolation

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
    stripe_1 = tf.math.logical_and(a_q2 >= log_q2min, a_q2 <= log_q2max)
    stripe_2 = a_x < log_xmin

    # --------------------------------------------------------------------
    # normal interpolation
    stripe = tf.math.logical_and(stripe_0, stripe_1)
    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = interpolate(in_x, in_q2,
                           log_xmin, log_xmax, padded_x, s_x,
                           log_q2min, log_q2max, padded_q2, s_q2,
                           actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    res = tf.cond(idx0 == 0, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # lowx
    stripe = tf.math.logical_and(stripe_2, stripe_1)
    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)


    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_extrapolation(in_x, in_q2,
                                  log_xmin, log_xmax, padded_x, s_x,
                                  log_q2min, log_q2max,  padded_q2, s_q2,
                                  actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    res += tf.cond(idx0 == 0, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # high q2
    stripe_3 = a_q2 > log_q2max
    stripe = tf.math.logical_and(stripe_0, stripe_3)

    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = highq2_extrapolation(in_x, in_q2,
                                    log_xmin, log_xmax, padded_x, s_x,
                                    log_q2min, log_q2max, padded_q2, s_q2,
                                    actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    res += tf.cond(idx0 == 0, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # low x high q2
    stripe_4 = a_q2 > log_q2max
    stripe = tf.math.logical_and(stripe_2, stripe_4)

    f_idx = tf.cast(tf.where(stripe), dtype=DTYPEINT)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_highq2_extrapolation(in_x, in_q2,
                                         log_xmin, log_xmax, padded_x, s_x,
                                         log_q2min, log_q2max, padded_q2, s_q2,
                                         actual_padded)
        return tf.scatter_nd(f_idx, ff_f, shape)

    idx0 = tf.size(f_idx, out_type=DTYPEINT)
    return res + tf.cond(idx0 == 0, empty_fn, gen_fun)
