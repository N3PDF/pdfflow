"""
    Spinor calculations
"""

import numpy as np
from pdfflow.configflow import fone, fzero
import tensorflow as tf
from parameters import TFLOAT4, s_in

zi = tf.complex(fzero, fone)


@tf.function(input_signature=[TFLOAT4])
def calc_zt2(pa):
    """ Transverse momentum squared along the y axis:
        returns px^2 + pz^2
    """
    bb = tf.square(pa[1, :]) + tf.square(pa[3, :])
    return bb


@tf.function(input_signature=[TFLOAT4])
def calc_ap(pa):
    """ compute py + E """
    at2 = calc_zt2(pa)
    ap = pa[2, :] + pa[0, :]
    conditional_p = at2 / (pa[0, :] - pa[2, :])
    ap = tf.where(ap < pa[0, :] / 2.0, conditional_p, ap)
    return tf.complex(ap, fzero)


@tf.function(input_signature=[TFLOAT4, TFLOAT4, tf.TensorSpec(shape=[], dtype=bool)])
def zA(pa, pb, cross=False):  # cross == when only one of (pa,pb) is initial-state
    """ <ab> spinor """
    ap = calc_ap(pa)
    bp = calc_ap(pb)
    ra = tf.complex(pa[1, :], -pa[3, :]) * bp
    rb = tf.complex(pb[1, :], -pb[3, :]) * ap
    zval = zi * (ra - rb) / tf.sqrt(ap * bp)
    if not cross:
        return zval
    return zval * zi


@tf.function(input_signature=[TFLOAT4, TFLOAT4, tf.TensorSpec(shape=[], dtype=bool)])
def zB(pa, pb, cross=False):
    """ [ab] spinor """
    return tf.math.conj(zA(pa, pb, cross=cross))


@tf.function(input_signature=[TFLOAT4] * 2)
def dot_product(par, pbr):
    pa = tf.transpose(par)
    pb = tf.transpose(pbr)
    ener = pa[:, 0] * pb[:, 0]
    mome = tf.keras.backend.batch_dot(pa[:, 1:4], pb[:, 1:4])[:, 0]
    return ener - mome


@tf.function(input_signature=[TFLOAT4, TFLOAT4])
def zprod(pa, pb):
    return tf.math.real(zA(pa, pb) * zB(pa, pb))


@tf.function(input_signature=[TFLOAT4, TFLOAT4])
def sprod(pa, pb):
    pp = pa + pb
    return dot_product(pp, pp)


if __name__ == "__main__":
    from phase_space import psgen_2to3, psgen_2to4

    nevents = 10
    for n in [2, 3]:
        if n == 2:
            print("Generate a tree level phase space point")
            random_r = np.random.rand(nevents, 12)
            pa, pb, p1, p2, p3, x1, x2, _ = psgen_2to4(random_r)
        elif n == 3:
            print("Generate a real level phase space point")
            random_lo = np.random.rand(nevents, 9)
            pa, pb, p1, p2, x1, x2, _ = psgen_2to3(random_lo)
        # Ensure that (pa+pb)^2 is shat in different ways
        shat = s_in * x1 * x2
        shat_sprod = sprod(pa, pb)
        shat_zprod = zprod(pa, pb)
        zA_ab = zA(pa, pb)
        zB_ab = zB(pa, pb)
        zhat = tf.math.real(zA_ab * zB_ab)
        print("Testing II")
        np.testing.assert_allclose(shat, shat_sprod)
        np.testing.assert_allclose(shat, shat_zprod)
        np.testing.assert_allclose(shat, zhat)
        # Check sprod and zprod do the same for several cases
        print("Testing IF")
        np.testing.assert_allclose(zprod(pa, p1), sprod(pa, p1))
        np.testing.assert_allclose(zprod(pa, p2), sprod(pa, p2))
        np.testing.assert_allclose(zprod(pb, p1), sprod(pb, p1))
        np.testing.assert_allclose(zprod(pb, p2), sprod(pb, p2))
        if n == 3:
            np.testing.assert_allclose(zprod(pa, p3), sprod(pa, p3))
            np.testing.assert_allclose(zprod(pb, p3), sprod(pb, p3))
        print("Testing FF")
        np.testing.assert_allclose(zprod(p1, p2), sprod(p1, p2))
        if n == 3:
            np.testing.assert_allclose(zprod(p1, p3), sprod(p1, p3))
            np.testing.assert_allclose(zprod(p2, p3), sprod(p2, p3))
