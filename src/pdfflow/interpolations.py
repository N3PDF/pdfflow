import tensorflow as tf

def linear_interpolation(x, xl, xh, yl, yh):
    x = tf.expand_dims(x,1)
    xl = tf.expand_dims(xl,0)
    xh = tf.expand_dims(xh,0)
    return yl + (x - xl) / (xh - xl) * (yh - yl)

def extrapolate_linear(x, xl, xh, yl, yh):
    mask = tf.math.logical_and(yl > 1E-3, yh > 1E-3)
    #print(mask)
    def true_mask():
        a = tf.math.log(yl)
        b = tf.math.log(yh)
        #print('res',linear_interpolation(x, xl, xh, a, b))
        return tf.math.exp(linear_interpolation(x, xl, xh, a, b))
    def false_mask():
        return linear_interpolation(x, xl, xh, yl, yh)

    return tf.where(mask, true_mask(), false_mask())


def cubic_interpolation(T, VL, VDL, VH, VDH):
    t2 = T*T
    t3 = t2*T

    p0 = (2*t3 - 3*t2 + 1)*VL
    m0 = (t3 - 2*t2 + T)*VDL

    p1 = (-2*t3 + 3*t2)*VH
    m1 = (t3 - t2)*VDH

    return p0 + m0 + p1 + m1


def df_dx_func(corn_x, A):
    #just two kind of derivatives are useful in the x direction if we are interpolating in the [-1,2]x[-1,2] square:
    #four derivatives in x = 0 for all Qs (:,0,:)
    #four derivatives in x = 1 for all Qs (:,1,:)
    #derivatives are returned in a tensor with shape (#draws,2,4)

    lddx = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)
    rddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    left = (lddx+rddx)/2

    lddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    rddx = (A[3] - A[2]) / tf.expand_dims(tf.expand_dims(corn_x[3] - corn_x[2],0),-1)
    right = (lddx+rddx)/2
    return tf.stack([left, right], 0)

def l_df_dx_func(corn_x, A):
    left = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)

    lddx = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)
    rddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    right = (lddx+rddx)/2
    return tf.stack([left, right], 0)

def r_df_dx_func(corn_x, A):
    lddx = (A[1] - A[0]) / tf.expand_dims(tf.expand_dims(corn_x[1] - corn_x[0],0),-1)
    rddx = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    left = (lddx+rddx)/2

    right = (A[2] - A[1]) / tf.expand_dims(tf.expand_dims(corn_x[2] - corn_x[1],0),-1)
    return tf.stack([left, right], 0)

def bilinear_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    f_ql = linear_interpolation(a_x, corn_x[0], corn_x[1], A[0,0], A[1,0])
    f_qh = linear_interpolation(a_x, corn_x[0], corn_x[1], A[0,1], A[1,1])
    return linear_interpolation(a_Q2, corn_Q2[0], corn_Q2[1], f_ql, f_qh)


def default_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    dlogq_2 = tf.expand_dims(corn_Q2[3] - corn_Q2[2],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)
    vdl = ((vh - vl)/dlogq_1 + (vl - vll)/dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[1,3], df_dx[0,3]*dlogx_1, A[2,3], df_dx[1,3]*dlogx_1)
    vdh = ((vh - vl)/dlogq_1 + (vhh - vh)/dlogq_2) / 2

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def left_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = l_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[1] - corn_x[0]
    tlogx = tf.expand_dims((a_x - corn_x[0])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    dlogq_2 = tf.expand_dims(corn_Q2[3] - corn_Q2[2],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[0,1], df_dx[0,1]*dlogx_1, A[1,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[0,2], df_dx[0,2]*dlogx_1, A[1,2], df_dx[1,2]*dlogx_1)

    vll = cubic_interpolation(tlogx, A[0,0], df_dx[0,0]*dlogx_1, A[1,0], df_dx[1,0]*dlogx_1)
    vdl = ((vh - vl)/dlogq_1 + (vl - vll)/dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[0,3], df_dx[0,3]*dlogx_1, A[1,3], df_dx[1,3]*dlogx_1)
    vdh = ((vh - vl)/dlogq_1 + (vhh - vh)/dlogq_2) / 2

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def right_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = r_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    dlogq_2 = tf.expand_dims(corn_Q2[3] - corn_Q2[2],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)
    vdl = ((vh - vl)/dlogq_1 + (vl - vll)/dlogq_0) / 2

    vhh = cubic_interpolation(tlogx, A[1,3], df_dx[0,3]*dlogx_1, A[2,3], df_dx[1,3]*dlogx_1)
    vdh = ((vh - vl)/dlogq_1 + (vhh - vh)/dlogq_2) / 2

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def upper_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vdh = (vh - vl) / dlogq_1
    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)

    vdl = (vdh + (vl - vll)/dlogq_0) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def lower_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_1 = corn_Q2[1] - corn_Q2[0]
    dlogq_2 = tf.expand_dims(corn_Q2[2] - corn_Q2[1],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[0]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    
    vdl = (vh - vl) / dlogq_1

    vhh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)
    vdh = (vdl + (vhh - vh)/dlogq_2) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c0_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = l_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[1] - corn_x[0]
    tlogx = tf.expand_dims((a_x - corn_x[0])/dlogx_1,1)
    dlogq_1 = corn_Q2[1] - corn_Q2[0]
    dlogq_2 = tf.expand_dims(corn_Q2[2] - corn_Q2[1],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[0]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[0,0], df_dx[0,0]*dlogx_1, A[1,0], df_dx[1,0]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[0,1], df_dx[0,1]*dlogx_1, A[1,1], df_dx[1,1]*dlogx_1)

    vdl = (vh - vl) / dlogq_1

    vhh = cubic_interpolation(tlogx, A[0,2], df_dx[0,2]*dlogx_1, A[1,2], df_dx[1,2]*dlogx_1)
    vdh = (vdl + (vhh - vh)/dlogq_2) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c1_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = r_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_1 = corn_Q2[1] - corn_Q2[0]
    dlogq_2 = tf.expand_dims(corn_Q2[2] - corn_Q2[1],1)
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[0]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    
    vdl = (vh - vl) / dlogq_1

    vhh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)
    vdh = (vdl + (vhh - vh)/dlogq_2) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c2_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = r_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[2] - corn_x[1]
    tlogx = tf.expand_dims((a_x - corn_x[1])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[1,1], df_dx[0,1]*dlogx_1, A[2,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[1,2], df_dx[0,2]*dlogx_1, A[2,2], df_dx[1,2]*dlogx_1)

    vdh = (vh - vl) / dlogq_1
    vll = cubic_interpolation(tlogx, A[1,0], df_dx[0,0]*dlogx_1, A[2,0], df_dx[1,0]*dlogx_1)

    vdl = (vdh + (vl - vll)/dlogq_0) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)

def c3_bicubic_interpolation(a_x, a_Q2, corn_x, corn_Q2, A):
    df_dx = l_df_dx_func(corn_x, A)

    dlogx_1 = corn_x[1] - corn_x[0]
    tlogx = tf.expand_dims((a_x - corn_x[0])/dlogx_1,1)
    dlogq_0 = tf.expand_dims(corn_Q2[1] - corn_Q2[0],1)
    dlogq_1 = corn_Q2[2] - corn_Q2[1]
    tlogq = tf.expand_dims((a_Q2 - corn_Q2[1]) / dlogq_1,1)

    dlogx_1 = tf.expand_dims(dlogx_1,1)
    dlogq_1 = tf.expand_dims(dlogq_1,1)

    vl = cubic_interpolation(tlogx, A[0,1], df_dx[0,1]*dlogx_1, A[1,1], df_dx[1,1]*dlogx_1)
    vh = cubic_interpolation(tlogx, A[0,2], df_dx[0,2]*dlogx_1, A[1,2], df_dx[1,2]*dlogx_1)

    vdh = (vh - vl) / dlogq_1
    vll = cubic_interpolation(tlogx, A[0,0], df_dx[0,0]*dlogx_1, A[1,0], df_dx[1,0]*dlogx_1)

    vdl = (vdh + (vl - vll)/dlogq_0) / 2.0

    vdl *= dlogq_1
    vdh *= dlogq_1
    return cubic_interpolation(tlogq, vl, vdl, vh, vdh)