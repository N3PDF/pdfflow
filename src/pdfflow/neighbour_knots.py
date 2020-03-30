import tensorflow as tf


#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
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

#@tf.function
def lowx_extra_knots(a_x, a_q2, log_x, log_q2, actual_values):
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
    x_id = tf.constant([0,1], dtype=int64)
    Q2_id = tf.cast(tfp.stats.find_bins(a_q2, log_q2), dtype=int64)

    #corner coordinates

    corn_x = tf.gather(log_x, x_id)

    s = tf.size(log_q2, out_type=tf.int64)
    x = x_id * s

    A_id = tf.stack([x[0]+Q2_id, x[1]+Q2_id+s])
    A = tf.gather(actual_values, A_id)
    
    return corn_x, A