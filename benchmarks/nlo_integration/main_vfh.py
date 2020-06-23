#!/usr/bin/env python3
"""
Implementation of the Vector Boson Fusion Higgs production
using the hep-flow suite: pdfflow and vegasflow
"""
from argparse import ArgumentParser
import subprocess as sp
import numpy as np

from pdfflow.pflow import mkPDF
from pdfflow.configflow import float_me, fone, fzero, DTYPE
from pdfflow.functions import _condition_to_idx
from vegasflow.vflow import vegas_wrapper

import tensorflow as tf

from parameters import *
import phase_space
import me

pdfset = "NNPDF31_nnlo_as_0118/0"
# Instantiate the PDF
DIRNAME = (
    sp.run(
        ["lhapdf-config", "--datadir"], stdout=sp.PIPE, universal_newlines=True
    ).stdout.strip("\n")
    + "/"
)
pdf = mkPDF(pdfset, DIRNAME)
# TODO: get alpha val

##### PDF calculation
@tf.function(input_signature=[TFLOAT1, TFLOAT1])
def luminosity(x1, x2):
    """ Returns f(x1)*f(x2) """
    q2array = muR2 * tf.ones_like(x1)
    utype = pdf.xfxQ2([2, 4], x1, q2array)
    dtype = pdf.xfxQ2([1, 3], x2, q2array)
    lumi = tf.reduce_sum(utype * dtype, axis=-1)
    return lumi / x1 / x2


### Main functions
@tf.function
def vfh_production_leading_order(xarr, **kwargs):
    """ Wrapper for LO VFH calculation """
    # Compute the phase space point
    pa, pb, p1, p2, x1, x2, wgt = phase_space.psgen_2to3(xarr)
    # Apply cuts
    stripe, idx = phase_space.pt_cut_2of2(p1, p2)
    pa = tf.boolean_mask(pa, stripe, axis=1)
    pb = tf.boolean_mask(pb, stripe, axis=1)
    p1 = tf.boolean_mask(p1, stripe, axis=1)
    p2 = tf.boolean_mask(p2, stripe, axis=1)
    wgt = tf.boolean_mask(wgt, stripe, axis=0)
    x1 = tf.boolean_mask(x1, stripe, axis=0)
    x2 = tf.boolean_mask(x2, stripe, axis=0)
    # Compute luminosity
    lumi = luminosity(x1, x2)
    me_lo = me.qq_h_lo(pa, pb, p1, p2)
    res = lumi * me_lo * wgt
    final_result = res * flux / x1 / x2
    return tf.scatter_nd(idx, final_result, shape=xarr.shape[0:1])


@tf.function
def vfh_production_real(xarr, **kwargs):
    """ Wrapper for R VFH calculation """
    # Compute the phase space point
    pa, pb, p1, p2, p3, x1, x2, wgt = phase_space.psgen_2to4(xarr)

    # Input a PS point from NNLOJET
#     pa = np.zeros_like(pa)
#     pb = np.zeros_like(pb)
#     p1 = np.zeros_like(p1)
#     p2 = np.zeros_like(p2)
#     p3 = np.zeros_like(p3)
#     x1 = np.ones_like(x1)
#     x2 = np.ones_like(x2)
# 
#     x1 *= 0.51306089926047227
#     x2 *= 5.2644661467767841E-002
# 
#     pb[1,:] =      0.000000
#     pb[2,:] =      0.000000
#     pb[3,:] =   2219.522543
#     pb[0,:] =   2219.522543
#     pa[1,:] =     -0.000000
#     pa[2,:] =     -0.000000
#     pa[3,:] =  -2219.522543
#     pa[0,:] =   2219.522543
#     p3[1,:] =     -6.213524
#     p3[2,:] =     29.208424
#     p3[3,:] =      4.599521
#     p3[0,:] =     30.214160
#     p2[1,:] =   1689.902208
#     p2[2,:] =   -385.974216
#     p2[3,:] =    120.388150
#     p2[0,:] =   1737.595716
#     p1[1,:] =     23.346000
#     p1[2,:] =    356.765793
#     p1[3,:] =    688.767650
#     p1[0,:] =    776.033339
# 

    # Apply cuts
    stripe, idx = phase_space.pt_cut_3of3(p1, p2, p3, True, pa, pb)
    
    pa = tf.boolean_mask(pa, stripe, axis=1)
    pb = tf.boolean_mask(pb, stripe, axis=1)
    p1 = tf.boolean_mask(p1, stripe, axis=1)
    p2 = tf.boolean_mask(p2, stripe, axis=1)
    p3 = tf.boolean_mask(p3, stripe, axis=1)
    wgt = tf.boolean_mask(wgt, stripe, axis=0)
    x1 = tf.boolean_mask(x1, stripe, axis=0)
    x2 = tf.boolean_mask(x2, stripe, axis=0)
    if phase_space.UNIT_PHASE:
        return tf.scatter_nd(idx, wgt, shape=xarr.shape[0:1])

    # Compute luminosity
    lumi = luminosity(x1, x2)
    me_r = me.qq_h_r(pa, pb, p1, p2, p3)
    res = lumi * me_r * wgt
    final_result = res * flux / x1 / x2
    return tf.scatter_nd(idx, final_result, shape=xarr.shape[0:1])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-l", "--level", default="LO", help="Integration level")
    parser.add_argument(
        "-n", "--nevents", type=int, default=int(1e6), help="Number of events"
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=5, help="Number of iterations"
    )
    args = parser.parse_args()

    ncalls = args.nevents
    niter = args.iterations

    if args.level == "LO":
        print("Running Leading Order")
        print(f"ncalls={ncalls}, niter={niter}")
        ndim = 9
        res = vegas_wrapper(
            vfh_production_leading_order, ndim, niter, ncalls, compilable=True
        )
    elif args.level == "R":
        print("Running Real Tree level")
        print(f"ncalls={ncalls}, niter={niter}")
        ndim = 12
        res = vegas_wrapper(vfh_production_real, ndim, niter, ncalls, compilable=True)
