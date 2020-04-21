import tensorflow as tf
from pdfflow.subgrid import act_on_empty
from pdfflow.subgrid import interpolate
from pdfflow.subgrid import lowx_extrapolation
from pdfflow.subgrid import lowq2_extrapolation
from pdfflow.subgrid import lowx_lowq2_extrapolation
from pdfflow.subgrid import last_interpolation
from pdfflow.subgrid import highq2_extrapolation
from pdfflow.subgrid import lowx_highq2_extrapolation

float64 = tf.float64
int64 = tf.int64

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[None, None], dtype=float64),
                 tf.TensorSpec(shape=[2], dtype=int64)])
def inner_subgrid(u, a_x, a_q2,
                  log_x, log_xmin, log_xmax,
                  log_q2, log_q2min, log_q2max,
                  values, shape):
    """Inner subgrid interpolation"""
    #print('retrace inner subgrid')
    empty_fn = lambda: tf.constant(0.0, dtype=float64)

    valid = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    stripe = tf.math.logical_and(a_q2 >= log_q2min, a_q2 < log_q2max)
    in_stripe = tf.math.logical_and(valid, stripe)
    lowx_stripe = tf.math.logical_and(a_x < log_xmin, stripe)

    in_f_idx = tf.where(in_stripe)
    lowx_f_idx = tf.where(lowx_stripe)

    def in_fun():
        in_x = tf.boolean_mask(a_x, in_stripe)
        in_q2 = tf.boolean_mask(a_q2, in_stripe)
        ff_f = interpolate(u, in_x, in_q2,
                           log_x, log_xmin, log_xmax,
                           log_q2, log_q2min, log_q2max,
                           values)
        return tf.scatter_nd(in_f_idx, ff_f, shape)

    def lowx_fun():
        in_x = tf.boolean_mask(a_x, lowx_stripe)
        in_q2 = tf.boolean_mask(a_q2, lowx_stripe)
        ff_f = lowx_extrapolation(u, in_x, in_q2,
                                  log_x, log_xmin, log_xmax,
                                  log_q2, log_q2min, log_q2max,
                                  values)
        return tf.scatter_nd(lowx_f_idx, ff_f, shape)

    inside = act_on_empty(in_f_idx, empty_fn, in_fun)

    lowx = act_on_empty(lowx_f_idx, empty_fn, lowx_fun)

    return inside + lowx

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[None, None], dtype=float64),
                 tf.TensorSpec(shape=[2], dtype=int64)])
def first_subgrid(u, a_x, a_q2,
                  log_x, log_xmin, log_xmax,
                  log_q2, log_q2min, log_q2max,
                  values, shape):
    """First subgrid interpolation"""
    #print('retrace first subgrid')

    empty_fn = lambda: tf.constant(0.0, dtype=float64)
    # --------------------------------------------------------------------
    # exploit inner subgrid
    stripe = tf.math.logical_and(a_q2 >= log_q2min, a_q2 < log_q2max)

    f_idx = tf.where(stripe)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        size = tf.shape(in_x)
        shape_ = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)

        ff_f = inner_subgrid(u, in_x, in_q2,
                             log_x, log_xmin, log_xmax,
                             log_q2, log_q2min, log_q2max,
                             values, shape_)
        return tf.scatter_nd(f_idx, ff_f, shape)

    res = act_on_empty(f_idx, empty_fn, gen_fun)
    # --------------------------------------------------------------------
    # low q2
    x_stripe = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    q2_stripe = a_q2 < log_q2min
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    f_idx = tf.where(stripe)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowq2_extrapolation(u, in_x, in_q2,
                                   log_x, log_xmin, log_xmax,
                                   log_q2, log_q2min, log_q2max,
                                   values)
        return tf.scatter_nd(f_idx, ff_f, shape)

    res += act_on_empty(f_idx, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # low x low q2
    stripe = tf.math.logical_and(a_x < log_xmin, a_q2 < log_q2min)
    f_idx = tf.where(stripe)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_lowq2_extrapolation(u, in_x, in_q2,
                                        log_x, log_xmin, log_xmax,
                                        log_q2, log_q2min, log_q2max,
                                        values)
        return tf.scatter_nd(f_idx, ff_f, shape)

    res += act_on_empty(f_idx, empty_fn, gen_fun)

    return res

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[None], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[], dtype=float64),
                 tf.TensorSpec(shape=[None, None], dtype=float64),
                 tf.TensorSpec(shape=[2], dtype=int64)])
def last_subgrid(u, a_x, a_q2,
                 log_x, log_xmin, log_xmax,
                 log_q2, log_q2min, log_q2max,
                 values, shape):
    """Last subgrid interpolation"""
    #print('retrace last subgrid')

    empty_fn = lambda: tf.constant(0.0, dtype=float64)
    # --------------------------------------------------------------------
    # exploit inner subgrid
    stripe = tf.math.logical_and(a_q2 >= log_q2min, a_q2 < log_q2max)

    f_idx = tf.where(stripe)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        size = tf.shape(in_x)
        shape_ = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)
        ff_f = inner_subgrid(u, in_x, in_q2,
                             log_x, log_xmin, log_xmax,
                             log_q2, log_q2min, log_q2max,
                             values, shape_)
        return tf.scatter_nd(f_idx, ff_f, shape)

    res = act_on_empty(f_idx, empty_fn, gen_fun)
    # --------------------------------------------------------------------
    # last interpolation
    stripe = a_q2 == log_q2max

    f_idx = tf.where(stripe)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = last_interpolation(u, in_x, in_q2,
                                  log_x, log_xmin, log_xmax,
                                  log_q2, log_q2min, log_q2max,
                                  values)
        return tf.scatter_nd(f_idx, ff_f, shape)

    res += act_on_empty(f_idx, empty_fn, gen_fun)

    # --------------------------------------------------------------------
    # high q2
    x_stripe = tf.math.logical_and(a_x >= log_xmin, a_x <= log_xmax)
    q2_stripe = a_q2 > log_q2max
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    f_idx = tf.where(stripe)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = highq2_extrapolation(u, in_x, in_q2,
                                    log_x, log_xmin, log_xmax,
                                    log_q2, log_q2min, log_q2max,
                                    values)
        return tf.scatter_nd(f_idx, ff_f, shape)

    res += act_on_empty(f_idx, empty_fn, gen_fun)
    # --------------------------------------------------------------------
    # low x high q2
    stripe = tf.math.logical_and(a_x < log_xmin, a_q2 > log_q2max)

    f_idx = tf.where(stripe)

    def gen_fun():
        in_x = tf.boolean_mask(a_x, stripe)
        in_q2 = tf.boolean_mask(a_q2, stripe)
        ff_f = lowx_highq2_extrapolation(u, in_x, in_q2,
                                         log_x, log_xmin, log_xmax,
                                         log_q2, log_q2min, log_q2max,
                                         values)
        return tf.scatter_nd(f_idx, ff_f, shape)

    res += act_on_empty(f_idx, empty_fn, gen_fun)

    return res
