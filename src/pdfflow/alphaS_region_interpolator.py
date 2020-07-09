"""
Contains the extrapolation and interpolation functions wrappers for the different regions
"""

import tensorflow as tf
from pdfflow.configflow import DTYPE, DTYPEINT, float_me, fone
from pdfflow.neighbour_knots import alphaS_neighbour_knots
from pdfflow.alphaS_interpolations import alphaS_cubic_interpolation

ALPHAS_INTERPOLATE_SIGNATURE = [
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPEINT),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
]


@tf.function(input_signature=ALPHAS_INTERPOLATE_SIGNATURE)
def alphaS_interpolate(
    a_q2, padded_q2, s_q2, actual_padded,
):
    """
    Basic Cubic Interpolation inside the alphaS subgrid
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
            alphaS values: contains the padded (q2) grid
    """
    q2_bins, corn_q2, alphaS_vals = alphaS_neighbour_knots(
        a_q2, padded_q2, actual_padded
    )

    return alphaS_cubic_interpolation(
        a_q2, q2_bins, corn_q2, alphaS_vals, s_q2
    )