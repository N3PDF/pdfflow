import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
#import matplotlib.pyplot as plt
#import p

float64 = tf.float64

def linear_interpolation(x, xl, xh, yl, yh):
    x = tf.expand_dims(x,1)
    xl = tf.expand_dims(xl,1)
    xh = tf.expand_dims(xh,1)

    return yl + (x - xl) / (xh - xl) * (yh - yl)

def cubic_interpolation(T, VL, VDL, VH, VDH):
    t2 = T*T
    t3 = t2*T

    p0 = (2*t3 - 3*t2 + 1)*VL
    m0 = (t3 - 2*t2 + T)*VDL

    p1 = (-2*t3 + 3*t2)*VH
    m1 = (t3 - t2)*VDH

    return p0 + m0 + p1 + m1

def remove_edge_stripes(a_x, a_Q2, logx, logQ2):

    x_stripe = tf.math.logical_or(a_x < logx[1], a_x >= logx[-2])

    out_x = tf.boolean_mask(a_x, x_stripe)
    out_Q2 = tf.boolean_mask(a_Q2, x_stripe)
    in_x = tf.boolean_mask(a_x, ~x_stripe)
    in_Q2 = tf.boolean_mask(a_Q2, ~x_stripe)


    #remove Q2 stripes and concatenate those points with the previous ones
    Q_stripe = tf.math.logical_or(in_Q2 < logQ2[1], in_Q2 >= logQ2[-2])


    out_x = tf.concat([out_x, tf.boolean_mask(in_x, Q_stripe)],0)
    out_Q2 = tf.concat([out_Q2, tf.boolean_mask(in_Q2, Q_stripe)],0)
    in_x = tf.boolean_mask(in_x, ~Q_stripe)
    in_Q2 = tf.boolean_mask(in_Q2, ~Q_stripe)
    return in_x, in_Q2, out_x, out_Q2


def df_dx_func(corn_x, A):
    #just two kind of derivatives are useful in the x direction if we are interpolating in the [-1,2]x[-1,2] square:
    #four derivatives in x = 0 for all Qs (:,0,:)
    #four derivatives in x = 1 for all Qs (:,1,:)
    #derivatives are returned in a tensor with shape (#draws,2,4)

    lddx = (A[:,:,1,:] - A[:,:,0,:]) / tf.expand_dims(tf.expand_dims(corn_x[:,1] - corn_x[:, 0],1),-1)
    rddx = (A[:,:,2,:] - A[:,:,1,:]) / tf.expand_dims(tf.expand_dims(corn_x[:,2] - corn_x[:,1],1),-1)
    left = (lddx+rddx)/2

    lddx = (A[:,:,2,:] - A[:,:,1,:]) / tf.expand_dims(tf.expand_dims(corn_x[:,2] - corn_x[:, 1],1),-1)
    rddx = (A[:,:,3,:] - A[:,:,2,:]) / tf.expand_dims(tf.expand_dims(corn_x[:,3] - corn_x[:,2],1),-1)
    right = (lddx+rddx)/2
    return tf.stack([left, right], 2)

def bilinear_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    f_ql = linear_interpolation(a_x, corn_x[0], corn_x[1], A[:,:,0,0], A[:,:,1,0])
    f_qh = linear_interpolation(a_x, corn_x[0], corn_x[1], A[:,:,0,1], A[:,:,1,1])
    return linear_interpolation(a_Q2, corn_Q2[0], corn_Q2[1], f_ql, f_qh)


def bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = df_dx_func(corn_x, A)

    dlogx_1 = corn_x[:,2] - corn_x[:,1]
    tlogx = tf.expand_dims((a_x - corn_x[:,1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[:,1] - corn_Q2[:,0],1)
    dlogq_1 = corn_Q2[:,2] - corn_Q2[:,1]
    dlogq_2 = tf.expand_dims(corn_Q2[:,3] - corn_Q2[:,2],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[:,1]) / dlogq_1,1)


    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)


    vl = cubic_interpolation(tlogx, A[:,:,1,1], df_dx[:,:,0,1]*dlogx_1, A[:,:,2,1], df_dx[:,:,1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[:,:,1,2], df_dx[:,:,0,2]*dlogx_1, A[:,:,2,2], df_dx[:,:,1,2]*dlogx_1)

    vll = cubic_interpolation(tlogx, A[:,:,1,0], df_dx[:,:,0,0]*dlogx_1, A[:,:,2,0], df_dx[:,:,1,0]*dlogx_1)
    vdl = ((vh - vl)/dlogq_1 + (vl - vll)/dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[:,:,1,3], df_dx[:,:,0,3]*dlogx_1, A[:,:,2,3], df_dx[:,:,1,3]*dlogx_1)
    vdh = ((vh - vl)/dlogq_1 + (vhh - vh)/dlogq_2) / 2

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)








class subgrid:
    def __init__(self, grid=None):

        self.x = tf.constant(grid[0], dtype=float64)
        self.Q = tf.constant(grid[1], dtype=float64)
        self.Q2 = tf.pow(self.Q, 2)
        self.flav = grid[2]
        self.values = tf.constant(grid[3], dtype=float64)

        self.logx = tf.math.log(self.x)
        self.logQ2 = tf.math.log(self.Q2)

        self.xmin = np.amin(grid[0])
        self.xmax = np.amax(grid[0])
        self.Qmin = np.amin(grid[1])
        self.Qmax = np.amax(grid[1])
        self.Q2min = self.Qmin**2
        self.Q2max = self.Qmax**2



    def print_summary(self):
        print('\n----------- Pdf Summary ----------------')
        print('| Size of the pdf grid: ',self.values.shape, '   |')
        print('| x is in [%2e, %2e] |'%(self.xmin, self.xmax))
        print('| Q2 is in [%2e, %2e]|'%(self.Q2min, self.Q2max))
        print('----------------------------------------\n')

    def get_value(self, a_x_id, a_Q2_id):
        '''
        Returns the pdf value in a knot point
        Parameters:
            a_x_id, tensor of x indeces, shape: [#draws]
            a_Q2_id, tensor of Q2 indeces, shape: [#draws]
        '''
        idx = a_x_id * self.Q2.shape[0] + a_Q2_id
        return tf.gather(self.values, idx)


    #@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=float64), tf.TensorSpec(shape=None, dtype=float64)])
    #def interpolate_fn(self, a_x, a_Q2):
    #    print('\nTracing interpolate with : a_x,  shape' + str(a_x.shape) + '; a_Q2, shape' + str(a_Q2.shape))
    #    return self.interpolate(a_x, a_Q2)




    def two_neighbour_knots(self, a_x, a_Q2):
        #knot indeces of the [0,0] point in the square
        x_id = tf.cast(tfp.stats.find_bins(a_x, self.logx), dtype=tf.int64)
        Q2_id = tf.cast(tfp.stats.find_bins(a_Q2, self.logQ2), dtype=tf.int64)

        #corner coordinates
        corn_x = [tf.gather(self.logx, x_id)] + [tf.gather(self.logx, x_id + 1)]
        corn_Q2 = [tf.gather(self.logQ2, Q2_id)] + [tf.gather(self.logQ2, Q2_id + 1)]

        #pdf values
        f00 = self.get_value(x_id, Q2_id)
        f01 = self.get_value(x_id, Q2_id+1)
        f10 = self.get_value(x_id+1, Q2_id)
        f11 = self.get_value(x_id+1, Q2_id+1)

        A1 = tf.stack([f00,f01], axis=2)
        A2 = tf.stack([f10,f11], axis=2)
        A = tf.stack([A1,A2], axis=2)
        return corn_x, corn_Q2, A


    def four_neighbour_knots(self, a_x, a_Q2):
        x_id = tf.cast(tfp.stats.find_bins(a_x, self.logx, name='find_bins_logx'), dtype=tf.int64)
        Q2_id = tf.cast(tfp.stats.find_bins(a_Q2, self.logQ2, name='find_bins_logQ2'), dtype=tf.int64)

        corn_x = [tf.gather(self.logx, (x_id-1))] + [tf.gather(self.logx, x_id)] +[tf.gather(self.logx, (x_id+1))] + [tf.gather(self.logx, (x_id+2))]
        corn_Q2 = [tf.gather(self.logQ2, (Q2_id-1))] + [tf.gather(self.logQ2, Q2_id)] + [tf.gather(self.logQ2, (Q2_id+1))] + [tf.gather(self.logQ2, (Q2_id+2))]


        f0 = tf.stack([self.get_value(x_id-1, Q2_id-1), self.get_value(x_id-1, Q2_id), self.get_value(x_id-1, Q2_id+1), self.get_value(x_id-1, Q2_id+2)], -1)
        f1 = tf.stack([self.get_value(x_id, Q2_id-1), self.get_value(x_id, Q2_id), self.get_value(x_id, Q2_id+1), self.get_value(x_id, Q2_id+2)], -1)
        f2 = tf.stack([self.get_value(x_id+1, Q2_id-1), self.get_value(x_id+1, Q2_id), self.get_value(x_id+1, Q2_id+1), self.get_value(x_id+1, Q2_id+2)], -1)
        f3 = tf.stack([self.get_value(x_id+2, Q2_id-1), self.get_value(x_id+2, Q2_id), self.get_value(x_id+2, Q2_id+1), self.get_value(x_id+2, Q2_id+2)], -1)
        A = tf.stack([f0,f1,f2,f3], 2)

        return tf.stack(corn_x,1), tf.stack(corn_Q2,1), A

    def interpolate(self, a_x, a_Q2):

        #assert tf.math.reduce_all(tf.math.logical_and(self.xmin <= a_x, a_x <= self.xmax))

        #find which points are near the edges and linear interpolate between them
        #otherwise use bicubic interpolation
        #remove x stripes and put those points in out_x and out_Q2

        in_x, in_Q2, out_x, out_Q2 = remove_edge_stripes(a_x, a_Q2, self.logx, self.logQ2)
        #print(self.x)
        #print('in x: ',in_x)
        #print('in_Q2: ', in_Q2)
        #print('out_x: ', out_x)
        #print('out_Q2', out_Q2)
        #exit()
        a2,a3,a4 = self.two_neighbour_knots(out_x, out_Q2)
        out_f = bilinear_interpolation(out_x,out_Q2,a2,a3,a4)

        a2,a3,a4 = self.four_neighbour_knots(in_x, in_Q2)
        in_f = bicubic_interpolation(in_x, in_Q2,a2,a3,a4)

        final_x = tf.concat([in_x,out_x],0)
        final_Q2 = tf.concat([in_Q2, out_Q2],0)
        final_f = tf.concat([in_f, out_f],0)

        #final_x = out_x
        #final_Q2 = out_Q2
        #final_f = out_f


        return final_x, final_Q2, final_f