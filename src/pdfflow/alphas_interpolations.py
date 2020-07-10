"""
    Basic low-level interpolation functions
"""

import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, FMAX


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
    ]
)
def cubic_interpolation(T, VL, VDL, VH, VDH):
    """Cubic extrapolation itself"""
    #print('cubic int')
    t2 = T * T
    t3 = t2 * T

    p0 = (2 * t3 - 3 * t2 + 1) * VL
    m0 = (t3 - 2 * t2 + T) * VDL

    p1 = (-2 * t3 + 3 * t2) * VH
    m1 = (t3 - t2) * VDH

    res = p0 + m0 + p1 + m1

    return tf.where(tf.math.abs(res)<2., res, tf.ones_like(res)*FMAX)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
        tf.TensorSpec(shape=[4, None], dtype=DTYPE),
        tf.TensorSpec(shape=[4, None], dtype=DTYPE),
    ]
)
def daS_dq2_func(q2_id, s_q2, corn_q2, A):
    """
    Computes derivatives to make the alphas cubic interpolation
    When a query point is in the left or rightmost bin of the q2 axis, it
    automatically ignores the knots that would have gone outside array
    boundaries in the computation (this is done by a mask and tf.where,
    exploiting the q2_id variable)
    """
    # print('alphas_df_dx func')
    # just two kind of derivatives are usually useful
    # if we are interpolating in the [-1,2] interval:
    # left derivatives concerning q2 within [-1,0,1]
    # right derivatives concerning q2 within [0,1,2]
    # derivatives are returned in a tensor with shape (2,#draws)
    # Note: there shouldn't never be a division by zero

    #print('das dq2')
    diff_A = A[1:]-A[:-1]
    diff_q2 = corn_q2[1:] - corn_q2[:-1]

    forward = diff_A[1:]/diff_q2[1:]
    backward = diff_A[:-1]/diff_q2[:-1]
    central = (forward + backward)/2

    left = tf.where(q2_id == 1, forward[0], central[0])

    right = tf.where(q2_id == s_q2 - 1, backward[1], central[1])

    return tf.stack([left, right], 0)



@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[4, None], dtype=DTYPE),
        tf.TensorSpec(shape=[4, None], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
    ]
)
def alphas_cubic_interpolation(
    a_q2, q2_id, corn_q2, A, s_q2
):
    """
    Makes the alphas cubic interpolation: when a query point is in the lower
    or uppermost bin of the q2 axis, it automatically ignores the knots
    that would have gone outside array boundaries in the computation
    (this is done by a mask and tf.where, exploiting the q2_id variable)
    Returns
    ----------
        tf.tensor of shape [None]
        LogCubic Interpolated points
    """
    #print('alphas bic int')
    dlogq2 = corn_q2[2] - corn_q2[1]
    tlogq2 = (a_q2 - corn_q2[1])/dlogq2

    daS_dq2 = daS_dq2_func(q2_id, s_q2, corn_q2, A)

    return cubic_interpolation(tlogq2, A[1], daS_dq2[0]*dlogq2,
                               A[2], daS_dq2[1]*dlogq2)
