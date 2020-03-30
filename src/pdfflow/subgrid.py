import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pdfflow.selection import *
from pdfflow.neighbour_knots import *
from pdfflow.interpolations import *

float64 = tf.float64
int64 = tf.int64

def act_on_empty(input_tensor, fn_true, fn_false):
    idx0 = tf.shape(input_tensor)[0]
    return tf.cond(idx0 == 0, fn_true, fn_false)
'''
def linear_interpolation(x, xl, xh, yl, yh):
    x = tf.expand_dims(x,1)
    xl = tf.expand_dims(xl,1)
    xh = tf.expand_dims(xh,1)

    return yl + (x - xl) / (xh - xl) * (yh - yl)

def cubic_interpolation(T, VL, VDL, VH, VDH):
    t2 = T*T
    t3 = t2*T

    p0 = (2*t3 - 3*t2 + 1)*VL
    m0 = (t3 - 2*t2 + T)*VDL

    p1 = (-2*t3 + 3*t2)*VH
    m1 = (t3 - t2)*VDH

    return p0 + m0 + p1 + m1

def select_left_stripe(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first and last bin in the logx array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = a_x < log_x[1]
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[1], a_q2 < log_q2[-2])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_right_stripe(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first and last bin in the logx array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = a_x >= log_x[-2]
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[1], a_q2 < log_q2[-2])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_upper_stripe(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the last bin in the logq2 array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = tf.math.logical_and(a_x >= log_x[1], a_x < log_x[-2])
    q2_stripe = a_q2 >= log_q2[-2]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_lower_stripe(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first bin in the logq2 array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = tf.math.logical_and(a_x >= log_x[1], a_x < log_x[-2])
    q2_stripe = a_q2 < log_q2[1]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_inside(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first bin in the logq2 array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = tf.math.logical_and(a_x >= log_x[1], a_x < log_x[-2])
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[1], a_q2 < log_q2[-2])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_c0(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first and last bin in the logx array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = a_x < log_x[1]
    q2_stripe = a_q2 < log_q2[1]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_c1(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first and last bin in the logx array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = a_x >= log_x[-2]
    q2_stripe = a_q2 < log_q2[1]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_c2(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first and last bin in the logx array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = a_x >= log_x[-2]
    q2_stripe = a_q2 >= log_q2[-2]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def select_c3(a_x, a_q2, log_x, log_q2):
    """
    Find the values in the first and last bin in the logx array

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
        logx: tf.tensor
            array of values of log(x) of the subgrid
        logq2: tf.tensor
            array of values of log(q2) of the subgrid


    """
    x_stripe = a_x < log_x[1]
    q2_stripe = a_q2 >= log_q2[-2]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

def df_dx_func(corn_x, A):
    #just two kind of derivatives are useful in the x direction if we are interpolating in the [-1,2]x[-1,2] square:
    #four derivatives in x = 0 for all Qs (:,0,:)
    #four derivatives in x = 1 for all Qs (:,1,:)
    #derivatives are returned in a tensor with shape (#draws,2,4)

    lddx = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)
    rddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    left = (lddx+rddx)/2

    lddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    rddx = (A[3] - A[2]) / tf.expand_dims(tf.expand_dims(corn_x[3] - corn_x[2],0),-1)
    right = (lddx+rddx)/2
    return tf.stack([left, right], 0)

def l_df_dx_func(corn_x, A):
    left = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)

    lddx = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)
    rddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    right = (lddx+rddx)/2
    return tf.stack([left, right], 0)

def r_df_dx_func(corn_x, A):
    lddx = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)
    rddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    left = (lddx+rddx)/2

    right = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    return tf.stack([left, right], 0)

def bilinear_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    f_ql = linear_interpolation(a_x, corn_x[0], corn_x[1], A[0,0], A[1,0])
    f_qh = linear_interpolation(a_x, corn_x[0], corn_x[1], A[0,1], A[1,1])
    return linear_interpolation(a_Q2, corn_Q2[0], corn_Q2[1], f_ql, f_qh)


def default_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    dlogq_2 = tf.expand_dims(corn_Q2[3] - corn_Q2[2],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)
    vdl = ((vh - vl)/dlogq_1 + (vl - vll)/dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[1,3], df_dx[0,3]*dlogx_1, A[2,3], df_dx[1,3]*dlogx_1)
    vdh = ((vh - vl)/dlogq_1 + (vhh - vh)/dlogq_2) / 2

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def left_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = l_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[1] - corn_x[0]
    tlogx = tf.expand_dims((a_x - corn_x[0])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    dlogq_2 = tf.expand_dims(corn_Q2[3] - corn_Q2[2],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[0,1], df_dx[0,1]*dlogx_1, A[1,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[0,2], df_dx[0,2]*dlogx_1, A[1,2], df_dx[1,2]*dlogx_1)

    vll = cubic_interpolation(tlogx, A[0,0], df_dx[0,0]*dlogx_1, A[1,0], df_dx[1,0]*dlogx_1)
    vdl = ((vh - vl)/dlogq_1 + (vl - vll)/dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[0,3], df_dx[0,3]*dlogx_1, A[1,3], df_dx[1,3]*dlogx_1)
    vdh = ((vh - vl)/dlogq_1 + (vhh - vh)/dlogq_2) / 2

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def right_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = r_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    dlogq_2 = tf.expand_dims(corn_Q2[3] - corn_Q2[2],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)
    vdl = ((vh - vl)/dlogq_1 + (vl - vll)/dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[1,3], df_dx[0,3]*dlogx_1, A[2,3], df_dx[1,3]*dlogx_1)
    vdh = ((vh - vl)/dlogq_1 + (vhh - vh)/dlogq_2) / 2

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def upper_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vdh = (vh - vl) / dlogq_1
    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)

    vdl = (vdh + (vl - vll)/dlogq_0) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def lower_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_1 = corn_Q2[1] - corn_Q2[0]
    dlogq_2 = tf.expand_dims(corn_Q2[2] - corn_Q2[1],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[0]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,0], df_dx[0,1]*dlogx_1, A[2,0], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,1], df_dx[0,2]*dlogx_1, A[2,1], df_dx[1,2]*dlogx_1)
    
    vdl = (vh - vl) / dlogq_1

    vhh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)
    vdh = (vdl + (vhh - vh)/dlogq_2) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c0_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = l_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_1 = corn_Q2[1] - corn_Q2[0]
    dlogq_2 = tf.expand_dims(corn_Q2[2] - corn_Q2[1],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[0]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,0], df_dx[0,1]*dlogx_1, A[2,0], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,1], df_dx[0,2]*dlogx_1, A[2,1], df_dx[1,2]*dlogx_1)
    
    vdl = (vh - vl) / dlogq_1

    vhh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)
    vdh = (vdl + (vhh - vh)/dlogq_2) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c1_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = r_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_1 = corn_Q2[1] - corn_Q2[0]
    dlogq_2 = tf.expand_dims(corn_Q2[2] - corn_Q2[1],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[0]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,0], df_dx[0,1]*dlogx_1, A[2,0], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,1], df_dx[0,2]*dlogx_1, A[2,1], df_dx[1,2]*dlogx_1)
    
    vdl = (vh - vl) / dlogq_1

    vhh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)
    vdh = (vdl + (vhh - vh)/dlogq_2) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c2_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = r_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vdh = (vh - vl) / dlogq_1
    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)

    vdl = (vdh + (vl - vll)/dlogq_0) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c3_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = l_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vdh = (vh - vl) / dlogq_1
    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)

    vdl = (vdh + (vl - vll)/dlogq_0) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)




# Utility functions
@tf.function
def two_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    #knot indeces of the [0,0] point in the square
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2), dtype=int64)

    #corner coordinates

    corn_x_id = tf.stack([x_id, x_id+1],0)
    corn_Q2_id = tf.stack([Q2_id, Q2_id+1],0) 
    
    corn_x = tf.gather(log_x, corn_x_id)
    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+Q2_id, x+Q2_id+1])
    b = tf.stack([x+Q2_id+s, x+Q2_id+s+1])

    A_id = tf.stack([a,b])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

@tf.function
def four_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id-1, x_id, x_id+1, x_id+2],0)
    corn_Q2_id = tf.stack([Q2_id-1, Q2_id, Q2_id+1, Q2_id+2],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+Q2_id-s-1, x+Q2_id-s, x+Q2_id-s+1, x+Q2_id-s+2])
    b = tf.stack([x+Q2_id-1, x+Q2_id, x+Q2_id+1, x+Q2_id+2])
    c = tf.stack([x+Q2_id+s-1, x+Q2_id+s, x+Q2_id+s+1, x+Q2_id+s+2])
    d = tf.stack([x+Q2_id+2*s-1, x+Q2_id+2*s, x+Q2_id+2*s+1, x+Q2_id+2*s+2])

    A_id = tf.stack([a,b,c,d])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

@tf.function
def l_four_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id, x_id+1, x_id+2],0)
    corn_Q2_id = tf.stack([Q2_id-1, Q2_id, Q2_id+1, Q2_id+2],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    b = tf.stack([x+Q2_id-1, x+Q2_id, x+Q2_id+1, x+Q2_id+2])
    c = tf.stack([x+Q2_id+s-1, x+Q2_id+s, x+Q2_id+s+1, x+Q2_id+s+2])
    d = tf.stack([x+Q2_id+2*s-1, x+Q2_id+2*s, x+Q2_id+2*s+1, x+Q2_id+2*s+2])

    A_id = tf.stack([b,c,d])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

@tf.function
def r_four_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id-1, x_id, x_id+1],0)
    corn_Q2_id = tf.stack([Q2_id-1, Q2_id, Q2_id+1, Q2_id+2],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+Q2_id-s-1, x+Q2_id-s, x+Q2_id-s+1, x+Q2_id-s+2])
    b = tf.stack([x+Q2_id-1, x+Q2_id, x+Q2_id+1, x+Q2_id+2])
    c = tf.stack([x+Q2_id+s-1, x+Q2_id+s, x+Q2_id+s+1, x+Q2_id+s+2])

    A_id = tf.stack([a,b,c])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

@tf.function
def u_four_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id-1, x_id, x_id+1, x_id+2],0)
    corn_Q2_id = tf.stack([Q2_id-1, Q2_id, Q2_id+1],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+Q2_id-s-1, x+Q2_id-s, x+Q2_id-s+1])
    b = tf.stack([x+Q2_id-1, x+Q2_id, x+Q2_id+1])
    c = tf.stack([x+Q2_id+s-1, x+Q2_id+s, x+Q2_id+s+1])
    d = tf.stack([x+Q2_id+2*s-1, x+Q2_id+2*s, x+Q2_id+2*s+1])

    A_id = tf.stack([a,b,c,d])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

#.function
def d_four_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id-1,x_id, x_id+1, x_id+2],0)
    corn_Q2_id = tf.stack([Q2_id, Q2_id+1, Q2_id+2],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+Q2_id-s, x+Q2_id-s+1, x+Q2_id-s+2])
    b = tf.stack([x+Q2_id, x+Q2_id+1, x+Q2_id+2])
    c = tf.stack([x+Q2_id+s, x+Q2_id+s+1, x+Q2_id+s+2])
    d = tf.stack([x+Q2_id+2*s, x+Q2_id+2*s+1, x+Q2_id+2*s+2])

    A_id = tf.stack([a,b,c,d])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

def c0_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id, x_id+1, x_id+2],0)
    corn_Q2_id = tf.stack([Q2_id, Q2_id+1, Q2_id+2],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    b = tf.stack([x+Q2_id, x+Q2_id+1, x+Q2_id+2])
    c = tf.stack([x+Q2_id+s, x+Q2_id+s+1, x+Q2_id+s+2])
    d = tf.stack([x+Q2_id+2*s, x+Q2_id+2*s+1, x+Q2_id+2*s+2])

    A_id = tf.stack([b,c,d])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

def c1_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id-1, x_id, x_id+1],0)
    corn_Q2_id = tf.stack([Q2_id, Q2_id+1, Q2_id+2],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+Q2_id-s, x+Q2_id-s+1, x+Q2_id-s+2])
    b = tf.stack([x+Q2_id, x+Q2_id+1, x+Q2_id+2])
    c = tf.stack([x+Q2_id+s, x+Q2_id+s+1, x+Q2_id+s+2])

    A_id = tf.stack([a,b,c])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

def c2_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id-1, x_id, x_id+1],0)
    corn_Q2_id = tf.stack([Q2_id-1, Q2_id, Q2_id+1],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+Q2_id-s-1, x+Q2_id-s, x+Q2_id-s+1])
    b = tf.stack([x+Q2_id-1, x+Q2_id, x+Q2_id+1])
    c = tf.stack([x+Q2_id+s-1, x+Q2_id+s, x+Q2_id+s+1])

    A_id = tf.stack([a,b,c])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A

def c3_neighbour_knots(a_x, a_q2, log_x, log_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        log_x: tf.tensor
            values of log(x) of the grid
        log_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    """
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x, name='find_bins_logx'), dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2, name='find_bins_logQ2'), dtype=int64)

    corn_x_id = tf.stack([x_id, x_id+1, x_id+2],0)
    corn_Q2_id = tf.stack([Q2_id-1, Q2_id, Q2_id+1],0)       

    corn_x = tf.gather(log_x, corn_x_id)

    corn_Q2 = tf.gather(log_q2, corn_Q2_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    b = tf.stack([x+Q2_id-1, x+Q2_id, x+Q2_id+1])
    c = tf.stack([x+Q2_id+s-1, x+Q2_id+s, x+Q2_id+s+1])
    d = tf.stack([x+Q2_id+2*s-1, x+Q2_id+2*s, x+Q2_id+2*s+1])

    A_id = tf.stack([b,c,d])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, corn_Q2, A
'''

class Subgrid:
    """
    Wrapper class around subgrdis.
    This class reads the LHAPDF grid and stores useful information:

    - log(x)
    - log(Q2)
    - Q2
    - values
    """
    def __init__(self, grid = None):
        if grid is None:
            raise ValueError("Subrids need a grid to generate from")

        self.flav = tf.cast(grid[2], dtype=int64)

        xarr = grid[0]
        self.log_x = tf.cast(tf.math.log(xarr), dtype=float64)

        qarr = grid[1]
        q2arr = tf.constant(pow(qarr,2), dtype=float64)
        self.log_q2 = tf.math.log(q2arr)
        self.log_q2max = tf.reduce_max(self.log_q2)
        self.log_q2min = tf.reduce_min(self.log_q2)

        self.grid_values = tf.constant(grid[3], dtype=float64)

    def ledge_interpolation(self, a_x, a_q2, actual_values):
        """ 
        Interpolation to use near the border in x

        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        #a2, a3, a4 = two_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        #result = bilinear_interpolation(a_x, a_q2, a2, a3, a4)
        a2, a3, a4 = l_four_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = left_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result
    def redge_interpolation(self, a_x, a_q2, actual_values):
        """ 
        Interpolation to use near the border in x

        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        #a2, a3, a4 = two_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        #result = bilinear_interpolation(a_x, a_q2, a2, a3, a4)
        a2, a3, a4 = r_four_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = right_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def uedge_interpolation(self, a_x, a_q2, actual_values):
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
        a2, a3, a4 = u_four_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = upper_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def dedge_interpolation(self, a_x, a_q2, actual_values):
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
        a2, a3, a4 = d_four_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = lower_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def c0_interpolation(self, a_x, a_q2, actual_values):
        """ 
        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        a2, a3, a4 = c0_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = c0_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def c1_interpolation(self, a_x, a_q2, actual_values):
        """ 
        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        a2, a3, a4 = c1_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = c1_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def c2_interpolation(self, a_x, a_q2, actual_values):
        """ 
        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        a2, a3, a4 = c2_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = c2_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def c3_interpolation(self, a_x, a_q2, actual_values):
        """ 
        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        a2, a3, a4 = c3_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = c3_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def default_interpolation(self, a_x, a_q2, actual_values):
        """ 
        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        a2, a3, a4 = four_neighbour_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        result = default_bicubic_interpolation(a_x, a_q2, a2, a3, a4)
        return result

    def lowx_extrapolation(self, a_x, a_q2, actual_values):
        """ 
        Extrapolation in low x regime 

        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        a2, a4 = lowx_extra_knots(a_x, a_q2, self.log_x, self.log_q2, actual_values)
        print(a_x.shape)
        print(a2[0].shape)
        print(a2[1].shape)
        print(a4[0].shape)
        print(a4[1].shape)

        mask = tf.math.logical_and(a4[0] > 1E3, a4[1] > 1E3)

        def true_mask():
            x = a_x
            xl = a2[0]
            xh = a2[1]
            yl = tf.math.log(a4[0])
            yh = tf.math.log(a4[1])
            return tf.math.exp(linear_interpolation(x,xl,xh,yl,yh))
        def false_mask():
            return linear_interpolation(a_x, a2[0], a2[1], a4[0], a4[1])
        
        return tf.where(mask, true_mask(), false_mask())

    def interpolate(self, u, a_x, a_Q2):
        """ 
        Find which points are near the edges and linear interpolate between them
        otherwise use bicubic interpolation
        
        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        actual_values = tf.gather(self.grid_values, u, axis = -1)
        size = tf.shape(a_x)
        shape = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)
        empty_fn = lambda: tf.constant(0.0, dtype=float64)

        l_x, l_q2, l_index = select_left_stripe(a_x, a_Q2, self.log_x, self.log_q2)
        r_x, r_q2, r_index = select_right_stripe(a_x, a_Q2, self.log_x, self.log_q2)
        u_x, u_q2, u_index = select_upper_stripe(a_x, a_Q2, self.log_x, self.log_q2)
        d_x, d_q2, d_index = select_lower_stripe(a_x, a_Q2, self.log_x, self.log_q2)
        in_x, in_q2, in_index = select_inside(a_x, a_Q2, self.log_x, self.log_q2)
        #order of corners: anticlockwise starting from low x, low q2 corner
        c0_x, c0_q2, c0_index = select_c0(a_x, a_Q2, self.log_x, self.log_q2)
        c1_x, c1_q2, c1_index = select_c1(a_x, a_Q2, self.log_x, self.log_q2)
        c2_x, c2_q2, c2_index = select_c2(a_x, a_Q2, self.log_x, self.log_q2)
        c3_x, c3_q2, c3_index = select_c3(a_x, a_Q2, self.log_x, self.log_q2)
        ex_x, ex_q2, ex_index = select_extra_stripe(a_x, a_Q2, self.log_x, self.log_q2)

        #print('l' ,l_x.shape, l_q2.shape, l_index.shape)
        #print('r' ,r_x.shape, r_q2.shape, r_index.shape)
        #print('u' ,u_x.shape, u_q2.shape, u_index.shape)
        #print('d' ,d_x.shape, d_q2.shape, d_index.shape)
        #print('in' ,in_x.shape, in_q2.shape, in_index.shape)
        #print('c0' ,c0_x.shape, c0_q2.shape, c0_index.shape)
        #print('c1' ,c1_x, c1_q2, c1_index)
        #print('c2' ,c2_x.shape, c2_q2.shape, c2_index.shape)
        #print('c3' ,c3_x.shape, c3_q2.shape, c3_index.shape)



        def ledge_fn():
            l_f = self.ledge_interpolation(l_x, l_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(l_index,-1), l_f, shape)

        def redge_fn():
            r_f = self.redge_interpolation(r_x, r_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(r_index,-1), r_f, shape)

        def uedge_fn():
            u_f = self.uedge_interpolation(u_x, u_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(u_index,-1), u_f, shape)

        def dedge_fn():
            d_f = self.dedge_interpolation(d_x, d_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(d_index,-1), d_f, shape)

        def insi_fn():
            in_f = self.default_interpolation(in_x, in_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(in_index,-1), in_f, shape)

        def c0_fn():
            c0_f = self.c0_interpolation(c0_x, c0_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(c0_index,-1), c0_f, shape)

        def c1_fn():
            c1_f = self.c1_interpolation(c1_x, c1_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(c1_index,-1), c1_f, shape)

        def c2_fn():
            c2_f = self.c2_interpolation(c2_x, c2_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(c2_index,-1), c2_f, shape)

        def c3_fn():
            c3_f = self.c3_interpolation(c3_x, c3_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(c3_index,-1), c3_f, shape)

        def ex_fn():
            ex_f = self.lowx_extrapolation(ex_x, ex_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(ex_index,-1), ex_f, shape)

        ledge_res = act_on_empty(l_x, empty_fn, ledge_fn)
        redge_res = act_on_empty(r_x, empty_fn, redge_fn)
        uedge_res = act_on_empty(u_x, empty_fn, uedge_fn)
        dedge_res = act_on_empty(d_x, empty_fn, dedge_fn)
        insi_res = act_on_empty(in_x, empty_fn, insi_fn)
        c0_res = act_on_empty(c0_x, empty_fn, c0_fn)
        c1_res = act_on_empty(c1_x, empty_fn, c1_fn)
        c2_res = act_on_empty(c2_x, empty_fn, c2_fn)
        c3_res = act_on_empty(c3_x, empty_fn, c3_fn)
        ex_res = act_on_empty(ex_x, empty_fn, ex_fn)

        return ledge_res + redge_res + uedge_res\
               + dedge_res + insi_res \
               + c0_res + c1_res + c2_res + c3_res\
               + ex_res