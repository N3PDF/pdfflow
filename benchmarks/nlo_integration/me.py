"""
    Matrix Element functions for VFH calculation
"""

import numpy as np
from pdfflow.configflow import float_me
import tensorflow as tf
from parameters import TFLOAT4, stw, mw, gw, TFLOAT1
from phase_space import psgen_2to3, psgen_2to4
from spinors import dot_product, zA, zB, sprod


@tf.function(input_signature=[TFLOAT1])
def propagator_w(s):
    """ Compute the propagator of the W boson:
    (s - w)^2 + (m*Gamma_w)^2
    """
    t1 = tf.square(s - tf.square(mw))
    t2 = tf.square(mw * gw)
    return t1 + t2


@tf.function(input_signature=4 * [TFLOAT4])
def partial_lo(pa, pb, p1, p2):
    """ Computes the LO q Q -> Q q H (WW->H) """
    # Compute the propagators
    sa1 = -2.0 * dot_product(pa, p1)
    sb2 = -2.0 * dot_product(pb, p2)

    prop = propagator_w(sa1) * propagator_w(sb2)
    coup = tf.square(mw / tf.pow(stw, 1.5))
    rmcom = coup / prop

    # Compute the amplitude
    # W-boson, so only Left-Left
    amplitude = zA(pa, pb) * zB(p1, p2)
    amp2 = tf.math.real(amplitude * tf.math.conj(amplitude))

    me_res = 2.0 * amp2 * rmcom
    return me_res


# Leading Order matrix element
factor_lo = float_me(1.0702411577062499e-4)  # there is no alpha_s, alpha_ew computed at Mz val


@tf.function(input_signature=4 * [TFLOAT4])
def qq_h_lo(pa, pb, p1, p2):
    """ Computes the LO q Q -> Q q H (WW->H) """
    return factor_lo * partial_lo(pa, pb, p1, p2)


@tf.function(input_signature=[TFLOAT4] * 5)
def partial_qq_h_qQg(pa, pb, p1, p2, p3):
    """ Gluon radiated from leg pa-p1 """
    pa13 = pa - (p1 + p3)
    sa13 = dot_product(pa13, pa13)
    sb2 = -2.0 * dot_product(pb, p2)
    prop = propagator_w(sa13) * propagator_w(sb2)
    coup = tf.square(mw / tf.pow(stw, 1.5))
    rmcom = coup / prop

    # compute the amplitude
    zUp = zB(pa, p1, cross=True) * zA(p2, p1) + zB(pa, p3, cross=True) * zA(p2, p3)
    zFp = zB(pb, pa)  # (zB(p1,p3)*zB(pa,p3,cross=True))
    zAp = zFp * zUp

    zUm = zA(pa, p1, cross=True) * zB(pa, pb) + zB(pb, p3, cross=True) * zA(p1, p3)
    zFm = zA(p1, p2)  # (zA(p1,p3)*zA(pa,p3,cross=True))
    zAm = zFm * zUm

    s13 = 2.0 * dot_product(p1, p3)
    sa3 = 2.0 * dot_product(pa, p3)

    zamp2 = zAp * tf.math.conj(zAp) + zAm * tf.math.conj(zAm)
    amp = 2.0 * tf.math.real(zamp2) / s13 / sa3

    return amp * rmcom


factor_re = float_me(4.0397470069216974e-004)  # TODO compute alphas


@tf.function(input_signature=[TFLOAT4] * 5)
def qq_h_r(pa, pb, p1, p2, p3):
    """ Computes q Q -> Q q g H (WW -> H)
    Q = p1
    q = p2
    g = p3
    """
    r1 = partial_qq_h_qQg(pa, pb, p1, p2, p3)
    r2 = partial_qq_h_qQg(pb, pa, p2, p1, p3)
    return (r1 + r2) * factor_re


@tf.function(input_signature=[TFLOAT4] * 3)
def antenna_qgq(p1, p2, p3):
    """ Dipole for a q-g-q where p1 is an initial particle
    and p2 is the gluon """
    s12 = sprod(p1, p2)
    s13 = sprod(p1, p3)
    s23 = sprod(p2, p3)
    s123 = s12 + s13 + s23

    FullAnt = s12 / s23 + s23 / s12 + 2.0 * s13 * s123 / s12 / s23
    return FullAnt / s123


if __name__ == "__main__":
    nevents = 10
    print("Generate a tree level matrix element")
    random_lo = np.random.rand(nevents, 9)
    pa, pb, p1, p2, _, _, _ = psgen_2to3(random_lo)
    tree_level_res = qq_h_lo(pa, pb, p1, p2)
    random_lo = np.random.rand(nevents, 12)
    pa, pb, p1, p2, p3, _, _, _ = psgen_2to4(random_lo)
    real_level_res = qq_h_r(pa, pb, p1, p2, p3)
