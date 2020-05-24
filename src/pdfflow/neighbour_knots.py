import tensorflow as tf
import tensorflow_probability as tfp
from pdfflow.interpolations import float64
from pdfflow.interpolations import int64
#float64 = tf.float64
#int64 = tf.int64


@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None,None], dtype=float64)])
def four_neighbour_knots(a_x, a_q2, log_x, log_q2, padded_x, padded_q2, actual_values):
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
    #print('nk')
    x_id = tf.cast(tfp.stats.find_bins(a_x, log_x,
                                       name='find_bins_logx'),
                   dtype=int64) + 1

    q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2,
                                        name='find_bins_logq2'),
                    dtype=int64) + 1

    corn_x_id = tf.stack([x_id-1, x_id, x_id+1, x_id+2],0)
    corn_q2_id = tf.stack([q2_id-1, q2_id, q2_id+1, q2_id+2],0)

    corn_x = tf.gather(padded_x, corn_x_id)
    corn_q2 = tf.gather(padded_q2, corn_q2_id)


    s = tf.size(padded_q2, out_type=tf.int64)
    x = x_id * s

    a = tf.stack([x+q2_id-s-1, x+q2_id-s, x+q2_id-s+1, x+q2_id-s+2])
    b = tf.stack([x+q2_id-1, x+q2_id, x+q2_id+1, x+q2_id+2])
    c = tf.stack([x+q2_id+s-1, x+q2_id+s, x+q2_id+s+1, x+q2_id+s+2])
    d = tf.stack([x+q2_id+2*s-1, x+q2_id+2*s, x+q2_id+2*s+1, x+q2_id+2*s+2])

    A_id = tf.stack([a,b,c,d])
    A = tf.gather(actual_values, A_id)
    
    return x_id, q2_id, corn_x, corn_q2, A
