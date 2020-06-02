import tensorflow as tf
import tensorflow_probability as tfp
from pdfflow.configflow import DTYPE, DTYPEINT


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
    ]
)
def four_neighbour_knots(a_x, a_q2, padded_x, padded_q2, actual_values):
    """
    Parameters
    ----------
        a_x: tf.tensor
            tensor of values of x
        a_q2: tf.tensor
            tensor of values of q2
        padded_x: tf.tensor
            values of log(x) of the grid
        padded_q2: tf.tensor
            values of log(q2) of the grid
        actual_values: tf.tensor
            values of the grid
    Returns
    ----------
        x_id: tf.tensor of shape [None]
            x bin for each query point
        q2_id: tf.tensor of shape [None]
            q2 bin for each query point
        corn_x: tf.tensor of shape [4,None]
            x values of the 4 knots around the query point
        corn_q2: tf.tensor of shape [4,None]
            q2 values of the 4 knots around the query point
        A: tf.tensor of shape [4,4,None,None]
            pdf values of the 4*4 grid knots around the query point
            (first None is for query points, second None is for query pids)
    """
    # print('nk')
    x_id = tfp.stats.find_bins(a_x, padded_x[1:-1], dtype=DTYPEINT) + 1
    q2_id = tfp.stats.find_bins(a_q2, padded_q2[1:-1], dtype=DTYPEINT) + 1

    piu = tf.reshape(tf.range(-1, 3, dtype=DTYPEINT), (4, 1))
    corn_x_id = tf.repeat(tf.reshape(x_id, (1, -1)), 4, axis=0) + piu
    corn_q2_id = tf.repeat(tf.reshape(q2_id, (1, -1)), 4, axis=0) + piu

    corn_x = tf.gather(padded_x, corn_x_id, name="fnk_1")
    corn_q2 = tf.gather(padded_q2, corn_q2_id, name="fnk_2")

    s = tf.size(padded_q2, out_type=DTYPEINT)
    x = x_id * s

    pdf_idx = tf.reshape(x_id * s + q2_id, (1, -1))

    a = tf.repeat(pdf_idx - s, 4, axis=0) + piu
    b = tf.repeat(pdf_idx, 4, axis=0) + piu
    c = tf.repeat(pdf_idx + s, 4, axis=0) + piu
    d = tf.repeat(pdf_idx + 2 * s, 4, axis=0) + piu

    A_id = tf.stack([a, b, c, d])
    A = tf.gather(actual_values, A_id, name="fnk_3")

    return x_id, q2_id, corn_x, corn_q2, A
