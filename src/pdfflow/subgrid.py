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
        self.log_xmin = tf.reduce_min(self.log_x)
        self.log_xmax = tf.reduce_max(self.log_x)

        qarr = grid[1]
        q2arr = tf.constant(pow(qarr,2), dtype=float64)
        self.log_q2 = tf.math.log(q2arr)
        self.log_q2max = tf.reduce_max(self.log_q2)
        self.log_q2min = tf.reduce_min(self.log_q2)

        self.grid_values = tf.constant(grid[3], dtype=float64)
        #a positive flag says that this is the last subgrid in the pdf file
        self.flag = tf.constant(-1, dtype=int64)

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

    def last_interpolation(self,a_x, a_q2, actual_values,u):
        """ 
        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        empty_fn = lambda: tf.constant(0.0, dtype=float64)
        size = tf.shape(a_x)
        shape = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)
        stripe = tf.math.logical_and(a_x < self.log_x[1], a_q2 == self.log_q2max)
        l_x, l_q2, l_index = select_last(a_x, a_q2, stripe)
        
        stripe = tf.math.logical_and(a_x >= self.log_x[1], a_x <= self.log_x[-2])
        stripe = tf.math.logical_and(a_q2 == self.log_q2max,stripe)
        c_x, c_q2, c_index = select_last(a_x, a_q2, stripe)
        
        stripe = tf.math.logical_and(a_x >= self.log_x[-2], a_q2 == self.log_q2max)
        r_x, r_q2, r_index = select_last(a_x, a_q2, stripe)

        def l_fn():
            l_f = self.c3_interpolation(l_x, l_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(l_index,-1), l_f, shape)

        def c_fn():
            c_f = self.uedge_interpolation(c_x, c_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(c_index,-1), c_f, shape)

        def r_fn():
            r_f = self.c2_interpolation(r_x, r_q2, actual_values)
            return tf.scatter_nd(tf.expand_dims(r_index,-1), r_f, shape)
        
        l_res = act_on_empty(l_x, empty_fn, l_fn)
        c_res = act_on_empty(c_x, empty_fn, c_fn)
        r_res = act_on_empty(r_x, empty_fn, r_fn)


        return l_res + c_res + r_res




    def interpolate(self, u, a_x, a_q2):
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

        l_x, l_q2, l_index = select_left_stripe(a_x, a_q2, self.log_x, self.log_q2)
        r_x, r_q2, r_index = select_right_stripe(a_x, a_q2, self.log_x, self.log_q2)
        u_x, u_q2, u_index = select_upper_stripe(a_x, a_q2, self.log_x, self.log_q2)
        d_x, d_q2, d_index = select_lower_stripe(a_x, a_q2, self.log_x, self.log_q2)
        in_x, in_q2, in_index = select_inside(a_x, a_q2, self.log_x, self.log_q2)
        #order of corners: anticlockwise starting from low x, low q2 corner
        c0_x, c0_q2, c0_index = select_c0(a_x, a_q2, self.log_x, self.log_q2)
        c1_x, c1_q2, c1_index = select_c1(a_x, a_q2, self.log_x, self.log_q2)
        c2_x, c2_q2, c2_index = select_c2(a_x, a_q2, self.log_x, self.log_q2)
        c3_x, c3_q2, c3_index = select_c3(a_x, a_q2, self.log_x, self.log_q2)

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

        ledge_res = act_on_empty(l_x, empty_fn, ledge_fn)
        redge_res = act_on_empty(r_x, empty_fn, redge_fn)
        uedge_res = act_on_empty(u_x, empty_fn, uedge_fn)
        dedge_res = act_on_empty(d_x, empty_fn, dedge_fn)
        insi_res = act_on_empty(in_x, empty_fn, insi_fn)
        c0_res = act_on_empty(c0_x, empty_fn, c0_fn)
        c1_res = act_on_empty(c1_x, empty_fn, c1_fn)
        c2_res = act_on_empty(c2_x, empty_fn, c2_fn)
        c3_res = act_on_empty(c3_x, empty_fn, c3_fn)
        #interpolation for the upper bound of the grid
        last_fn = lambda: self.last_interpolation(a_x, a_q2, actual_values, u)
        last_empty_fn = lambda: tf.zeros(shape, dtype=float64)
        ll_res = tf.cond(self.flag>0, last_fn, last_empty_fn)
        

        return ledge_res + redge_res + uedge_res\
               + dedge_res + insi_res\
               + c0_res + c1_res + c2_res + c3_res\
               + ll_res

    def lowx_extrapolation(self, u, a_x, a_q2):
        """ 
        Extrapolation in low x regime 

        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        x_id = tf.constant([0,1], dtype=int64)
        corn_x = tf.gather(self.log_x, x_id)

        x, q2 = tf.meshgrid(corn_x, a_q2, indexing='ij')

        yl = self.interpolate(u, x[0],a_q2)
        yh = self.interpolate(u, x[1],a_q2)

        return extrapolate_linear(a_x, corn_x[0], corn_x[1],yl,yh)

    def lowq2_extrapolation(self, u, a_x, a_q2):
        q2_id = tf.constant([0], dtype=int64)
        corn_q2 = tf.gather(self.log_q2, q2_id)

        x, q2 = tf.meshgrid(a_x, corn_q2)

        fq2Min = tf.squeeze(self.interpolate(u, x[0], q2[0]))
        fq2Min1 = tf.squeeze(self.interpolate(u, x[0], q2[0]*1.01))
        
        a_q2 = tf.math.exp(a_q2)
        corn_q2 = tf.math.exp(corn_q2)

        mask = tf.math.abs(fq2Min) >= 1e-5
        anom = tf.where(mask,\
                       tf.maximum(tf.constant(-2.5,dtype=float64),\
                        (fq2Min1 - fq2Min) / fq2Min / 0.01),\
                       tf.constant(1, dtype=float64)\
                       )
        
        res = fq2Min * tf.math.pow(a_q2/corn_q2, anom*a_q2/corn_q2 + 1.0 - a_q2/corn_q2)
        return tf.expand_dims(res,1)

    def highq2_extrapolation(self, u, a_x, a_q2):
        """ 
        Extrapolation in hihg q2 regime 

        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        s = tf.size(self.log_q2, out_type=int64)
        q2_id = tf.stack([s-1,s-2])

        corn_q2 = tf.gather(self.log_q2, q2_id)

        x, q2 = tf.meshgrid(a_x, corn_q2)
        
        yl = self.interpolate(u, a_x,q2[0])
        yh = self.interpolate(u, a_x,q2[1])

        return extrapolate_linear(a_q2, corn_q2[0], corn_q2[1],yl,yh)

    def lowx_highq2_extrapolation(self, u, a_x, a_q2):
        """ 
        Extrapolation in high q2, low x regime 

        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        x_id = tf.constant([0,1], dtype=int64)
        s = tf.size(self.log_q2, out_type=int64)
        q2_id = tf.stack([s-1,s-2])

        corn_x = tf.gather(self.log_x, x_id)
        corn_q2 = tf.gather(self.log_q2, q2_id)

        x, q2 = tf.meshgrid(corn_x, corn_q2, indexing='ij')

        #maybe this is not necessary
        #maybe just access the pdf knot values
        f = self.interpolate(u,tf.reshape(x,[-1]), tf.reshape(q2,[-1]))

        fxMin = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[:1], f[1:2])

        fxMin1 = extrapolate_linear(a_q2, corn_q2[0], corn_q2[1], f[2:3], f[3:])

        return extrapolate_linear(a_x, corn_x[0], corn_x[1], fxMin, fxMin1)

    def lowx_lowq2_extrapolation(self, u, a_x, a_q2):
        """ 
        Extrapolation in high q2, low x regime 

        Parameters
        ----------
            a_x: tf.tensor
                query of values of log(x)
            a_q2: tf.tensor
                query of values of log(q2)
        """
        x_id = tf.constant([0,1], dtype=int64)
        q2_id = tf.stack([0,0])

        corn_x = tf.gather(self.log_x, x_id)
        corn_q2 = tf.gather(self.log_q2, q2_id)
        
        f = self.interpolate(u,\
                            tf.concat([corn_x,corn_x],0),\
                            tf.concat([corn_q2,1.01*corn_q2],0)\
                            )
        fq2Min = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[:1], f[1:2])

        fq2Min1 = extrapolate_linear(a_x, corn_x[0], corn_x[1], f[2:3], f[3:])

        fq2Min = tf.squeeze(fq2Min)
        fq2Min1 = tf.squeeze(fq2Min1)

        a_q2 = tf.math.exp(a_q2)
        corn_q2 = tf.math.exp(corn_q2[0])

        mask = tf.math.abs(fq2Min) >= 1e-5
        anom = tf.where(mask,\
                       tf.maximum(-2.5, (fq2Min1 - fq2Min) / fq2Min / 0.01),\
                       tf.constant(1, dtype=float64)\
                       )
        
        res = fq2Min * tf.math.pow(a_q2/corn_q2, anom*a_q2/corn_q2 + 1.0 - a_q2/corn_q2)

        return tf.expand_dims(res,1)