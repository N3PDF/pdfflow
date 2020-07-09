"""
    Basic low-level interpolation functions
"""

import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
    ]
)
def linear_interpolation(x, xl, xh, yl, yh):
    """Linear extrapolation itself"""
    # print('lin interp')
    x = tf.expand_dims(x, 1)
    return yl + (x - xl) / (xh - xl) * (yh - yl)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
    ]
)
def extrapolate_linear(x, xl, xh, yl, yh):
    """
    Selects by a mask which point has yl and yh greater or lower than a threshold:
    for lower points a linear extrapolation is performed
    for greater points a log-linear extrapolation is performed
    Returns
    ----------
        tf.tensor of shape [None,None]
        (Log)Linear Extrapolated points, with all pids queried

    """
    mask = tf.math.logical_and(yl > 1e-3, yh > 1e-3)
    a = tf.where(mask, tf.math.log(yl), yl)
    b = tf.where(mask, tf.math.log(yh), yh)
    res = linear_interpolation(x, xl, xh, a, b)
    return tf.where(mask, tf.math.exp(res), res)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
    ]
)
def cubic_interpolation(T, VL, VDL, VH, VDH):
    """Cubic extrapolation itself"""
    # print('cubic int')
    t2 = T * T
    t3 = t2 * T

    p0 = (2 * t3 - 3 * t2 + 1) * VL
    m0 = (t3 - 2 * t2 + T) * VDL

    p1 = (-2 * t3 + 3 * t2) * VH
    m1 = (t3 - t2) * VDH

    return p0 + m0 + p1 + m1


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
        tf.TensorSpec(shape=[4, None], dtype=DTYPE),
        tf.TensorSpec(shape=[4, 4, None, None], dtype=DTYPE),
    ]
)
def df_dx_func(x_id, s_x, corn_x, A):
    """
    Computes derivatives to make the bicubic interpolation
    When a query point is in the left or rightmost bin of the x axis, it
    automatically ignores the knots that would have gone outside array
    boundaries in the computation (this is done by a mask and tf.where,
    exploiting the x_id variable)
    """
    # print('df_dx func')
    # just two kind of derivatives are useful in the x direction
    # if we are interpolating in the [-1,2]x[-1,2] square:
    # four derivatives in x = 0 for all Qs (:,0,:)
    # four derivatives in x = 1 for all Qs (:,1,:)
    # derivatives are returned in a tensor with shape (2,4,#draws)
    edge = (A[2] - A[1]) / tf.reshape(corn_x[2] - corn_x[1], (1, -1, 1))

    lddx = (A[1] - A[0]) / tf.reshape(corn_x[1] - corn_x[0], (1, -1, 1))
    default_l = (lddx + edge) / 2

    rddx = (A[3] - A[2]) / tf.reshape(corn_x[3] - corn_x[2], (1, -1, 1))
    default_r = (edge + rddx) / 2

    mask_l = tf.reshape(x_id == 1, (1, -1, 1))
    left = tf.where(mask_l, edge, default_l)

    mask_r = tf.reshape(x_id == s_x - 1, (1, -1, 1))
    right = tf.where(mask_r, edge, default_r)

    return tf.stack([left, right], 0)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[4, None], dtype=DTYPE),
        tf.TensorSpec(shape=[4, None], dtype=DTYPE),
        tf.TensorSpec(shape=[4, 4, None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
    ]
)
def default_bicubic_interpolation(
    a_x, a_q2, x_id, q2_id, corn_x, corn_q2, A, s_x, s_q2
):
    """
    Makes the bicubic interpolation: when a query point is in the lower
    or uppermost bin of the q2 axis, it automatically ignores the knots
    that would have gone outside array boundaries in the computation
    (this is done by a mask and tf.where, exploiting the q2_id variable)
    Returns
    ----------
        tf.tensor of shape [None,None]
        LogBicubic Interpolated points, with all pids queried
    """
    # print('def bic int')
    dlogx_1 = corn_x[2] - corn_x[1]
    dlogq_1 = corn_q2[2] - corn_q2[1]

    tlogq = tf.expand_dims((a_q2 - corn_q2[1]) / dlogq_1, 1)
    tlogx = tf.expand_dims((a_x - corn_x[1]) / dlogx_1, 1)

    dlogq_0 = tf.expand_dims(corn_q2[1] - corn_q2[0], 1)
    dlogx_1 = tf.expand_dims(dlogx_1, 1)
    dlogq_1 = tf.expand_dims(dlogq_1, 1)
    dlogq_2 = tf.expand_dims(corn_q2[3] - corn_q2[2], 1)

    df_dx = df_dx_func(x_id, s_x, corn_x, A) * dlogx_1

    # TODO: can the 4 cubc interpolation be done in one go looking at the 4 dimension as the batch dimension?
    vl = cubic_interpolation(tlogx, A[1, 1], df_dx[0, 1], A[2, 1], df_dx[1, 1])
    vh = cubic_interpolation(tlogx, A[1, 2], df_dx[0, 2], A[2, 2], df_dx[1, 2])
    edge_vals = (vh - vl) / dlogq_1

    vll = cubic_interpolation(tlogx, A[1, 0], df_dx[0, 0], A[2, 0], df_dx[1, 0])
    default_l = (edge_vals + (vl - vll) / dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[1, 3], df_dx[0, 3], A[2, 3], df_dx[1, 3])
    default_r = (edge_vals + (vhh - vh) / dlogq_2) / 2

    mask_l = tf.reshape(q2_id == 1, [-1, 1])
    vdl = tf.where(mask_l, edge_vals, default_l)

    mask_r = tf.reshape(q2_id == s_q2 - 1, [-1, 1])
    vdh = tf.where(mask_r, edge_vals, default_r)

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)
