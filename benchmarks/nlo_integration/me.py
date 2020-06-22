"""
    Matrix Element functions for VFH calculation
"""

import numpy as np
from pdfflow.configflow import fone, fzero, float_me
import tensorflow as tf
from parameters import TFLOAT4, stw, mw, gw, stw, TFLOAT1
from phase_space import psgen_2to3
from spinors import dot_product, zA, zB


@tf.function(input_signature=[TFLOAT1])
def propagator_w(s):
    t1 = tf.square(s - tf.square(mw))
    t2 = tf.square(mw * gw)
    return t1 + t2


# Leading Order matrix element
factor_lo = float_me(
    1.0702411577062499e-4
)  # there is no alpha_s, alpha_ew computed at Mz val


@tf.function(input_signature=4*[TFLOAT4])
def qq_h_lo(pa, pb, p1, p2):
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
    return factor_lo * me_res

if __name__ == "__main__":
    nevents = 10
    print("Generate a tree level matrix element")
    random_lo = np.random.rand(nevents, 9)
    pa, pb, p1, p2, x1, x2, _ = psgen_2to3(random_lo)
    tree_level_res = qq_h_lo(pa, pb, p1, p2)
