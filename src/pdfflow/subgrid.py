import tensorflow as tf
import numpy as np
from pdfflow.neighbour_knots import four_neighbour_knots
from pdfflow.interpolations import default_bicubic_interpolation
from pdfflow.interpolations import extrapolate_linear
from pdfflow.interpolations import float64
from pdfflow.interpolations import int64
#float64 = tf.float64
#int64 = tf.int64

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
        self.padded_x = tf.concat([tf.expand_dims(self.log_xmin*0.99, 0),
                                   self.log_x,
                                   tf.expand_dims(self.log_xmax*1.01, 0)],
                                  axis=0)
        self.s_x = tf.size(self.log_x, out_type=int64)

        qarr = grid[1]
        q2arr = tf.constant(pow(qarr, 2), dtype=float64)
        self.log_q2 = tf.math.log(q2arr)
        self.log_q2max = tf.reduce_max(self.log_q2)
        self.log_q2min = tf.reduce_min(self.log_q2)
        self.padded_q2 = tf.concat([tf.expand_dims(self.log_q2min*0.99, 0),
                                    self.log_q2,
                                    tf.expand_dims(self.log_q2max*1.01, 0)],
                                   axis=0)
        self.s_q2 = tf.size(self.log_q2, out_type=int64)

        self.grid_values = tf.constant(grid[3], dtype=float64)

        a = tf.reshape(self.grid_values, [self.s_x, self.s_q2,-1])
        a = tf.pad(a, tf.constant([[1,1],[1,1],[0,0]]))
        self.padded_grid = tf.reshape(a, [(self.s_x+2)*(self.s_q2+2),-1])

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def interpolate(a_x, a_q2,
                log_xmin, log_xmax, padded_x, s_x,
                log_q2min, log_q2max, padded_q2, s_q2,
                actual_padded):
    """ 
    Basic Interpolation inside the subgrid
    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    #actual_padded = tf.gather(padded_grid, u, axis=-1)
    #size = tf.shape(a_x)
    #shape = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)

    a0, a1, a2, a3, a4 = four_neighbour_knots(a_x, a_q2,
                                              padded_x, padded_q2,
                                              actual_padded)
    
    return default_bicubic_interpolation(a_x, a_q2,
                                        a0, a1, a2, a3, a4,
                                        s_x, s_q2)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[None,None], dtype=float64)])
def lowx_extrapolation(a_x, a_q2,
                       log_xmin, log_xmax, padded_x, s_x,
                       log_q2min, log_q2max, padded_q2, s_q2,
                       actual_padded):
    """ 
    Extrapolation in low x regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    corn_x = padded_x[1:3]
    s = tf.size(a_x, out_type=int64)

    x, q2 = tf.meshgrid(corn_x, a_q2, indexing="ij")

    y = interpolate(tf.reshape(x, [-1]), tf.reshape(q2, [-1]),
                     log_xmin, log_xmax, padded_x, s_x,
                     log_q2min, log_q2max, padded_q2, s_q2,
                     actual_padded)

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], y[:s], y[s:])

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def lowq2_extrapolation(a_x, a_q2,
                        log_xmin, log_xmax, padded_x, s_x,
                        log_q2min, log_q2max, padded_q2, s_q2,
                        actual_padded):
    corn_q2 = tf.stack([padded_q2[1], 1.01*padded_q2[1]],0)

    x, q2 = tf.meshgrid(a_x, corn_q2)

    s = tf.size(a_x, out_type=int64)

    fq2Min = interpolate(tf.reshape(x,[-1]), tf.reshape(q2, [-1]),
                         log_xmin, log_xmax, padded_x, s_x,
                         log_q2min, log_q2max, padded_q2, s_q2,
                         actual_padded)
    
    fq2Min1 = fq2Min[s:]
    fq2Min = fq2Min[:s]

    a_q2 = tf.math.exp(a_q2)
    corn_q2 = tf.math.exp(corn_q2[:1])

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(mask,
                    tf.maximum(tf.constant(-2.5, dtype=float64),
                               (fq2Min1 - fq2Min) / fq2Min / 0.01),
                    tf.constant(1, dtype=float64))
    corn_q2 = tf.expand_dims(corn_q2,1)
    a_q2 = tf.expand_dims(a_q2,1)

    return fq2Min * tf.math.pow(a_q2 / corn_q2,
                                anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def highq2_extrapolation(a_x, a_q2,
                         log_xmin, log_xmax, padded_x, s_x,
                         log_q2min, log_q2max, padded_q2, s_q2,
                         actual_padded):
    """ 
    Extrapolation in hihg q2 regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    corn_q2 = padded_q2[-2:-4:-1]

    x, q2 = tf.meshgrid(a_x, corn_q2)
    s = tf.size(a_x,out_type=int64)

    y = interpolate(tf.reshape(x, [-1]), tf.reshape(q2, [-1]),
                     log_xmin, log_xmax, padded_x, s_x,
                     log_q2min, log_q2max, padded_q2, s_q2,
                     actual_padded)

    return extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], y[:s], y[s:])

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def lowx_highq2_extrapolation(a_x, a_q2,
                              log_xmin, log_xmax, padded_x, s_x,
                              log_q2min, log_q2max, padded_q2, s_q2,
                              actual_padded):
    """ 
    Extrapolation in high q2, low x regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """

    corn_x = padded_x[1:3]
    corn_q2 = padded_q2[-2:-4:-1]

    x, q2 = tf.meshgrid(corn_x, corn_q2)

    f = interpolate(tf.reshape(x, [-1]), tf.reshape(q2, [-1]),
                    log_xmin, log_xmax, padded_x, s_x,
                    log_q2min, log_q2max, padded_q2, s_q2,
                    actual_padded)


    fxMin = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[:1], f[2:3])

    fxMin1 = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[1:2], f[3:])

    return extrapolate_linear(a_x, corn_x[0], corn_x[1], fxMin, fxMin1)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=float64),
                              tf.TensorSpec(shape=[None], dtype=float64),
                              tf.TensorSpec(shape=[], dtype=int64),
                              tf.TensorSpec(shape=[None,None],dtype=float64)])
def lowx_lowq2_extrapolation(a_x, a_q2,
                             log_xmin, log_xmax, padded_x, s_x,
                             log_q2min, log_q2max, padded_q2, s_q2,
                             actual_padded):
    """ 
    Extrapolation in high q2, low x regime 

    Parameters
    ----------
        a_x: tf.tensor
            query of values of log(x)
        a_q2: tf.tensor
            query of values of log(q2)
    """
    corn_x = padded_x[1:3]
    corn_q2 = tf.stack([padded_q2[1],padded_q2[1]],0)

    f = interpolate(tf.concat([corn_x, corn_x], 0),
                    tf.concat([corn_q2, 1.01 * corn_q2], 0),
                    log_xmin, log_xmax, padded_x, s_x,
                    log_q2min, log_q2max, padded_q2, s_q2,
                    actual_padded)

    fq2Min = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[:1], f[1:2])

    fq2Min1 = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[2:3], f[3:])

    a_q2 = tf.expand_dims(tf.math.exp(a_q2),1)
    corn_q2 = tf.math.exp(corn_q2[0])

    mask = tf.math.abs(fq2Min) >= 1e-5
    anom = tf.where(mask,
                    tf.maximum(tf.constant(-2.5, dtype=float64),
                               (fq2Min1 - fq2Min) / fq2Min / 0.01),
                    tf.constant(1, dtype=float64))

    return fq2Min * tf.math.pow(a_q2 / corn_q2,
                                anom * a_q2 / corn_q2 + 1.0 - a_q2 / corn_q2)
