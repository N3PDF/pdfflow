import tensorflow as tf
import numpy as np
from pdfflow.selection import *
from pdfflow.neighbour_knots import *
from pdfflow.interpolations import *

float64 = tf.float64
int64 = tf.int64

def act_on_empty(input_tensor, fn_true, fn_false):
    #print('act on empty')
    idx0 = tf.shape(input_tensor)[0]
    return tf.cond(idx0 == 0, fn_true, fn_false)

class Subgrid:
    """
    Wrapper class around subgrdis.
    This class reads the LHAPDF grid and stores useful information:

    - log(x)
    - log(Q2)
    - Q2
    - values
    """

    def __init__(self, grid=None):
        """
        Init
        """

        if grid is None:
            raise ValueError("Subgrids need a grid to be generated from")

        self.flav = tf.cast(grid[2], dtype=int64)

        xarr = grid[0]
        self.log_x = tf.cast(tf.math.log(xarr), dtype=float64)
        self.log_xmin = tf.reduce_min(self.log_x)
        self.log_xmax = tf.reduce_max(self.log_x)

        qarr = grid[1]
        q2arr = tf.constant(pow(qarr, 2), dtype=float64)
        self.log_q2 = tf.math.log(q2arr)
        self.log_q2max = tf.reduce_max(self.log_q2)
        self.log_q2min = tf.reduce_min(self.log_q2)

        self.grid_values = tf.constant(grid[3], dtype=float64)

        #self.flag = tf.constant(1, dtype=int64)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def ledge_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Interpolation to use near the border in x

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('ledge inter')
    a2, a3, a4 = l_four_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                        actual_values)
    result = left_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def redge_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Interpolation to use near the border in x

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('redge inter')
    a2, a3, a4 = r_four_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                        actual_values)
    result = right_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def uedge_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Interpolation to use near the upper border in q2.
    Use central and backward difference to compute q2 derivatives

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('uedge inter')
    a2, a3, a4 = u_four_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                        actual_values)
    result = upper_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def dedge_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Interpolation to use near the lower border in x
    Use forward and central difference to compute q2 derivatives

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('dedge inter')
    a2, a3, a4 = d_four_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                        actual_values)
    result = lower_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def c0_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('c0 inter')
    a2, a3, a4 = c0_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                    actual_values)
    result = c0_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def c1_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('c1 inter')
    a2, a3, a4 = c1_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                    actual_values)
    result = c1_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def c2_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('c2 inter')
    a2, a3, a4 = c2_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                    actual_values)
    result = c2_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def c3_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('c3 inter')
    a2, a3, a4 = c3_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                    actual_values)
    result = c3_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def default_interpolation(a_x, a_q2, log_x, log_q2, actual_values):
    """ 
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('default inter')
    a2, a3, a4 = four_neighbour_knots(a_x, a_q2, log_x, log_q2,
                                        actual_values)
    result = default_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def interpolate(u, a_x, a_q2,
                log_x, log_xmin, log_xmax,
                log_q2, log_q2min, log_q2max,
                values):
    """ 
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('inter')
    actual_values = tf.gather(values, u, axis=-1)
    size = tf.shape(a_x)
    shape = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)
    empty_fn = lambda: tf.constant(0.0, dtype=float64)

    l_x, l_q2, l_index = select_left_stripe(a_x, a_q2, log_x, log_q2)
    r_x, r_q2, r_index = select_right_stripe(a_x, a_q2, log_x, log_q2)
    u_x, u_q2, u_index = select_upper_stripe(a_x, a_q2, log_x, log_q2)
    d_x, d_q2, d_index = select_lower_stripe(a_x, a_q2, log_x, log_q2)
    in_x, in_q2, in_index = select_inside(a_x, a_q2, log_x, log_q2)
    # order of corners: anticlockwise starting from low x, low q2 corner
    c0_x, c0_q2, c0_index = select_c0(a_x, a_q2, log_x, log_q2)
    c1_x, c1_q2, c1_index = select_c1(a_x, a_q2, log_x, log_q2)
    c2_x, c2_q2, c2_index = select_c2(a_x, a_q2, log_x, log_q2)
    c3_x, c3_q2, c3_index = select_c3(a_x, a_q2, log_x, log_q2)

    def ledge_fn():
        l_f = ledge_interpolation(l_x, l_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(l_index, -1), l_f, shape)

    def redge_fn():
        r_f = redge_interpolation(r_x, r_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(r_index, -1), r_f, shape)

    def uedge_fn():
        u_f = uedge_interpolation(u_x, u_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(u_index, -1), u_f, shape)

    def dedge_fn():
        d_f = dedge_interpolation(d_x, d_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(d_index, -1), d_f, shape)

    def insi_fn():
        in_f = default_interpolation(in_x, in_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(in_index, -1), in_f, shape)

    def c0_fn():
        c0_f = c0_interpolation(c0_x, c0_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(c0_index, -1), c0_f, shape)

    def c1_fn():
        c1_f = c1_interpolation(c1_x, c1_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(c1_index, -1), c1_f, shape)

    def c2_fn():
        c2_f = c2_interpolation(c2_x, c2_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(c2_index, -1), c2_f, shape)

    def c3_fn():
        c3_f = c3_interpolation(c3_x, c3_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(c3_index, -1), c3_f, shape)

    ledge_res = act_on_empty(l_x, empty_fn, ledge_fn)
    redge_res = act_on_empty(r_x, empty_fn, redge_fn)
    uedge_res = act_on_empty(u_x, empty_fn, uedge_fn)
    dedge_res = act_on_empty(d_x, empty_fn, dedge_fn)
    insi_res = act_on_empty(in_x, empty_fn, insi_fn)
    c0_res = act_on_empty(c0_x, empty_fn, c0_fn)
    c1_res = act_on_empty(c1_x, empty_fn, c1_fn)
    c2_res = act_on_empty(c2_x, empty_fn, c2_fn)
    c3_res = act_on_empty(c3_x, empty_fn, c3_fn)
    
    return ledge_res + redge_res + uedge_res + dedge_res + insi_res\
           + c0_res + c1_res + c2_res + c3_res

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def lowx_extrapolation(u, a_x, a_q2,
                       log_x, log_xmin, log_xmax,
                       log_q2, log_q2min, log_q2max,
                       values):
    """ 
    Extrapolation in low x regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('lowx extra')
    x_id = tf.constant([0, 1], dtype=int64)
    corn_x = tf.gather(log_x, x_id)

    x, q2 = tf.meshgrid(corn_x, a_q2, indexing="ij")

    yl = interpolate(u, x[0], a_q2,
                     log_x, log_xmin, log_xmax,
                     log_q2, log_q2min, log_q2max,
                     values)
    yh = interpolate(u, x[1], a_q2,
                     log_x, log_xmin, log_xmax,
                     log_q2, log_q2min, log_q2max,
                     values)

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], yl, yh)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def lowq2_extrapolation(u, a_x, a_q2,
                        log_x, log_xmin, log_xmax,
                        log_q2, log_q2min, log_q2max,
                        values):
    #print('lowq2 extra')
    q2_id = tf.constant([0], dtype=int64)
    corn_q2 = tf.gather(log_q2, q2_id)

    x, q2 = tf.meshgrid(a_x, corn_q2)

    fq2Min = interpolate(u, x[0], q2[0],
                         log_x, log_xmin, log_xmax,
                         log_q2, log_q2min, log_q2max,
                         values)
    fq2Min1 = interpolate(u, x[0], q2[0] * 1.01,
                          log_x, log_xmin, log_xmax,
                          log_q2, log_q2min, log_q2max,
                          values)

    a_q2 = tf.math.exp(a_q2)
    corn_q2 = tf.math.exp(corn_q2)

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(mask,
                    tf.maximum(tf.constant(-2.5, dtype=float64),
                               (fq2Min1 - fq2Min) / fq2Min / 0.01),
                    tf.constant(1, dtype=float64))
    corn_q2 = tf.expand_dims(corn_q2,1)
    a_q2 = tf.expand_dims(a_q2,1)
    #print('anom',anom.shape)
    #print('a_q2',a_q2.shape)
    #print('corn_q2',corn_q2.shape)
    res = fq2Min * tf.math.pow(a_q2 / corn_q2,
                               anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2)
    #print('res',res.shape)
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
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def last_interpolation(u, a_x, a_q2,
                       log_x, log_xmin, log_xmax,
                       log_q2, log_q2min, log_q2max,
                       values):
    """ 
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('last inter')
    empty_fn = lambda: tf.constant(0.0, dtype=float64)
    size = tf.shape(a_x)
    shape = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)
    actual_values = tf.gather(values, u, axis=-1)


    stripe = a_x < log_xmin
    ll_x, ll_q2, ll_index = select_last(a_x, a_q2, stripe)

    stripe = tf.math.logical_and(a_x >= log_x[0], a_x < log_x[1])
    l_x, l_q2, l_index = select_last(a_x, a_q2, stripe)

    stripe = tf.math.logical_and(a_x >= log_x[1], a_x < log_x[-2])
    c_x, c_q2, c_index = select_last(a_x, a_q2, stripe)

    stripe = a_x >= log_x[-2]
    r_x, r_q2, r_index = select_last(a_x, a_q2, stripe)

    def ll_fn():
        x_id = tf.constant([0, 1], dtype=int64)
        corn_x = tf.gather(log_x, x_id)

        x, q2 = tf.meshgrid(corn_x, ll_q2, indexing="ij")

        yl = c3_interpolation(x[0], ll_q2, log_x, log_q2, actual_values)

        yh = uedge_interpolation(x[1], ll_q2, log_x, log_q2, actual_values)

        ll_f = extrapolate_linear(ll_x, corn_x[0], corn_x[1], yl, yh)

        return tf.scatter_nd(tf.expand_dims(ll_index, -1), ll_f, shape)

    def l_fn():
        l_f = c3_interpolation(l_x, l_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(l_index, -1), l_f, shape)

    def c_fn():
        c_f = uedge_interpolation(c_x, c_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(c_index, -1), c_f, shape)

    def r_fn():
        r_f = c2_interpolation(r_x, r_q2, log_x, log_q2, actual_values)
        return tf.scatter_nd(tf.expand_dims(r_index, -1), r_f, shape)

    ll_res = act_on_empty(ll_x, empty_fn, ll_fn)
    l_res = act_on_empty(l_x, empty_fn, l_fn)
    c_res = act_on_empty(c_x, empty_fn, c_fn)
    r_res = act_on_empty(r_x, empty_fn, r_fn)

    return ll_res + l_res + c_res + r_res

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def highq2_extrapolation(u, a_x, a_q2,
                         log_x, log_xmin, log_xmax,
                         log_q2, log_q2min, log_q2max,
                         values):
    """ 
    Extrapolation in hihg q2 regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('highq2 extr')
    s = tf.size(log_q2, out_type=int64)
    q2_id = tf.stack([s - 1, s - 2])

    corn_q2 = tf.gather(log_q2, q2_id)

    x, q2 = tf.meshgrid(a_x, corn_q2)

    yl = last_interpolation(u, a_x, q2[0],
                     log_x, log_xmin, log_xmax,
                     log_q2, log_q2min, log_q2max,
                     values)
    yh = interpolate(u, a_x, q2[1],
                     log_x, log_xmin, log_xmax,
                     log_q2, log_q2min, log_q2max,
                     values)

    return extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], yl, yh)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def lowx_highq2_extrapolation(u, a_x, a_q2,
                              log_x, log_xmin, log_xmax,
                              log_q2, log_q2min, log_q2max,
                              values):
    """ 
    Extrapolation in high q2, low x regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('lowx highq2 extr')
    x_id = tf.constant([0, 1], dtype=int64)
    s = tf.size(log_q2, out_type=int64)
    q2_id = tf.stack([s - 1, s - 2])

    corn_x = tf.gather(log_x, x_id)
    corn_q2 = tf.gather(log_q2, q2_id)

    x, q2 = tf.meshgrid(corn_x, corn_q2, indexing="ij")

    f0 = last_interpolation(u, x[:,0], q2[:,0],
                    log_x, log_xmin, log_xmax,
                    log_q2, log_q2min, log_q2max,
                    values)
    f1 = interpolate(u, x[:,1], q2[:,1],
                    log_x, log_xmin, log_xmax,
                    log_q2, log_q2min, log_q2max,
                    values)

    f = tf.concat([f0,f1],0)

    fxMin = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[:1], f[2:3])

    fxMin1 = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[1:2], f[3:])

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], fxMin, fxMin1)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def lowx_lowq2_extrapolation(u, a_x, a_q2,
                             log_x, log_xmin, log_xmax,
                             log_q2, log_q2min, log_q2max,
                             values):
    """ 
    Extrapolation in high q2, low x regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #print('lowx lowq2 extr')
    x_id = tf.constant([0, 1], dtype=int64)
    q2_id = tf.stack([0, 0])

    corn_x = tf.gather(log_x, x_id)
    corn_q2 = tf.gather(log_q2, q2_id)

    f = interpolate(u, tf.concat([corn_x, corn_x], 0),
                    tf.concat([corn_q2, 1.01 * corn_q2], 0),
                    log_x, log_xmin, log_xmax,
                    log_q2, log_q2min, log_q2max,
                    values)
    fq2Min = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[:1], f[1:2])

    fq2Min1 = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[2:3], f[3:])

    #fq2Min = tf.squeeze(fq2Min)
    #fq2Min1 = tf.squeeze(fq2Min1)

    a_q2 = tf.expand_dims(tf.math.exp(a_q2),1)
    corn_q2 = tf.math.exp(corn_q2[0])

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(mask,
                    tf.maximum(tf.constant(-2.5, dtype=float64),
                               (fq2Min1 - fq2Min) / fq2Min / 0.01),
                    tf.constant(1, dtype=float64))

    res = fq2Min * tf.math.pow(a_q2 / corn_q2,
                                anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2)

    return res
