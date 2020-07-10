"""
Contains the extrapolation and interpolation functions wrappers for the different regions
"""

import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, float_me, fone
from pdfflow.neighbour_knots import alphas_neighbour_knots
from pdfflow.alphas_interpolations import alphas_cubic_interpolation

alphas_INTERPOLATE_SIGNATURE = [
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPEINT),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
]


@tf.function(input_signature=alphas_INTERPOLATE_SIGNATURE)
def alphas_interpolate(
    a_q2, padded_q2, s_q2, actual_padded,
):
    """
    Basic Cubic Interpolation inside the alphas subgrid
    Four Neighbour Knots selects grid knots around each query point to
    make the interpolation: 4 knots on the q2 axis are needed for each point,
    plus the pdf fvalues there.
    Default bicubic interpolation performs the interpolation itself

    Parameters
    ----------
        a_q2: tf.tensor of shape [None]
            query of values of log(q2)
        padded_q2: tf.tensor of shape [None]
            value for all the knots on the q2 axis
            padded with one zero at the beginning and one at the end to
            avoid out of range errors when querying points near boundaries
        s_q2: tf.tensor of shape []
            size of q2 knots tensor without padding
        actual_padded: tf.tensor of shape [None]
            alphas values: contains the padded (q2) grid
    """
    #print('alphas interpolate')
    q2_bins, corn_q2, alphas_vals = alphas_neighbour_knots(
        a_q2, padded_q2, actual_padded
    )

    return alphas_cubic_interpolation(
        a_q2, q2_bins, corn_q2, alphas_vals, s_q2
    )
