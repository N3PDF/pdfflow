import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
from pdfflow.selection import *
from pdfflow.neighbour_knots import *
from pdfflow.interpolations import *

float64 = tf.float64
int64 = tf.int64

def act_on_empty(input_tensor, fn_true, fn_false):
    idx0 = tf.shape(input_tensor)[0]
    return tf.cond(idx0 == 0, fn_true, fn_false)

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
        #print('a2', tf.math.exp(a2))
        #print('a3', tf.math.sqrt(tf.math.exp(a3)))
        #print('a4', a4)
        #exit()
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

        mask = tf.math.logical_and(a4[0] > 1E-3, a4[1] > 1E-3)

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
        '''
        print('l', l_x.shape)
        print('u ', u_x.shape)
        print('d ', d_x.shape)
        print('in ', in_x.shape)
        print('c0 ', c0_x.shape)
        print('c1 ', c1_x.shape)
        print('c2 ', c2_x.shape)
        print('c3 ', c3_x.shape)
        print('ex ', ex_x.shape)
        '''

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