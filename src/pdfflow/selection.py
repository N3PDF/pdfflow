import tensorflow as tf

float64 = tf.float64

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select l s')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = a_x < log_x[1]
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[1], a_q2 < log_q2[-2])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select r s')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = a_x >= log_x[-2]
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[1], a_q2 < log_q2[-2])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select u s')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = tf.math.logical_and(a_x >= log_x[1], a_x < log_x[-2])
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[-2], a_q2 < log_q2[-1])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select low s')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = tf.math.logical_and(a_x >= log_x[1], a_x < log_x[-2])
    q2_stripe = a_q2 < log_q2[1]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select in')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = tf.math.logical_and(a_x >= log_x[1], a_x < log_x[-2])
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[1], a_q2 < log_q2[-2])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select c0')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = a_x < log_x[1]
    q2_stripe = a_q2 < log_q2[1]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select c1')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = a_x >= log_x[-2]
    q2_stripe = a_q2 < log_q2[1]
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select c2')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = a_x >= log_x[-2]
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[-2], a_q2 < log_q2[-1])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64)])
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
    #print('select c3')
    valid = tf.logical_and(a_x >= log_x[0], a_x < log_x[-1])
    x_stripe = a_x < log_x[1]
    q2_stripe = tf.math.logical_and(a_q2 >= log_q2[-2], a_q2 < log_q2[-1])
    stripe = tf.math.logical_and(x_stripe, q2_stripe)
    stripe = tf.math.logical_and(valid, stripe)

    out_x = tf.boolean_mask(a_x, stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return out_x, out_q2, out_index

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=bool)])
def select_last(a_x, a_q2, stripe):
    #print('select last')
    out_x = tf.boolean_mask(a_x,stripe)
    out_q2 = tf.boolean_mask(a_q2, stripe)
    out_index = tf.squeeze(tf.where(stripe),-1)

    return  out_x, out_q2, out_index