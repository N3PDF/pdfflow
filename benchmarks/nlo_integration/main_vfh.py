#!/usr/bin/env python3
"""
Implementation of the Vector Boson Fusion Higgs production
using the hep-flow suite: pdfflow and vegasflow
"""
from argparse import ArgumentParser
import subprocess as sp

from pdfflow.pflow import mkPDF
from pdfflow.configflow import float_me, fone, fzero, DTYPE
from pdfflow.functions import _condition_to_idx
from vegasflow.vflow import VegasFlow

import tensorflow as tf

from parameters import *
import phase_space
import spinors
import me

pdfset = "NNPDF31_nnlo_as_0118/0"
# Instantiate the PDF
DIRNAME = (
    sp.run(["lhapdf-config", "--datadir"], stdout=sp.PIPE, universal_newlines=True).stdout.strip(
        "\n"
    )
    + "/"
)
pdf = mkPDF(pdfset, DIRNAME)
# TODO: get alpha val

##### PDF calculation
@tf.function(input_signature=[TFLOAT1, TFLOAT1, TFLOAT1])
def luminosity(x1, x2, q2array):
    """ Returns f(x1)*f(x2) """
#     q2array = muR2 * tf.ones_like(x1)
    utype = pdf.xfxQ2([2, 4], x1, q2array)
    dtype = pdf.xfxQ2([1, 3], x2, q2array)
    lumi = tf.reduce_sum(utype * dtype, axis=-1)
    return lumi / x1 / x2

### Main functions
@tf.function
def vfh_production_leading_order(xarr, **kwargs):
    """ Wrapper for LO VFH calculation 

    In commit: 5d68c6becb8372b1baaf908222e5d5e6b0a303c4
    the result was 157.2 +- 0.17 fb (1e6 events, 5 iterations, 0.4s p/it)

    """
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
    pt2s = phase_space.pt2many(tf.stack([p1, p2]))
    max_pt2 = tf.reduce_max(pt2s, axis=0)
    lumi = luminosity(x1, x2, max_pt2)

    me_lo = me.qq_h_lo(pa, pb, p1, p2)
    res = lumi * me_lo * wgt
    final_result = res * flux / x1 / x2
    return tf.scatter_nd(idx, final_result, shape=xarr.shape[0:1])


@tf.function
def vfh_production_real(xarr, **kwargs):
    """ Wrapper for R VFH calculation

    commit: e3628061766225ac4d8a62e6aa4393523d6e8b34
    result: 17.6 +- 0.03 fb (1e7 events, 5 iterations, 4.3s p/it 2 GPU)

    """
    # Compute the phase space point
    pa, pb, p1, p2, p3, x1, x2, wgt = phase_space.psgen_2to4(xarr)

    # Apply cuts
    stripe, idx, max_pt2 = phase_space.pt_cut_3of3(pa, pb, p1, p2, p3)

    pa = tf.boolean_mask(pa, stripe, axis=1)
    pb = tf.boolean_mask(pb, stripe, axis=1)
    p1 = tf.boolean_mask(p1, stripe, axis=1)
    p2 = tf.boolean_mask(p2, stripe, axis=1)
    p3 = tf.boolean_mask(p3, stripe, axis=1)
    wgt = tf.boolean_mask(wgt, stripe, axis=0)
    x1 = tf.boolean_mask(x1, stripe, axis=0)
    x2 = tf.boolean_mask(x2, stripe, axis=0)
    max_pt2 = tf.boolean_mask(max_pt2, stripe, axis=0)
    if phase_space.UNIT_PHASE:
        return tf.scatter_nd(idx, wgt, shape=xarr.shape[0:1])

    # Compute luminosity
    lumi = luminosity(x1, x2, max_pt2)

    me_r = me.qq_h_r(pa, pb, p1, p2, p3)
    res = lumi * me_r * wgt
#     res = lumi * wgt
    final_result = res * flux / x1 / x2
    return tf.scatter_nd(idx, final_result, shape=xarr.shape[0:1])


@tf.function
def vfh_production_nlo(xarr, **kwargs):
    """ Wrapper for R VFH calculation at NLO (2 jets)

    commit: e3628061766225ac4d8a62e6aa4393523d6e8b34
    Result:  20.6 +/- 0.3 (1e7 events, 5 iterations, 6.1s p/it 2 GPU)

    """
    # Compute the phase space point
    pa, pb, p1, p2, p3, x1, x2, wgt = phase_space.psgen_2to4(xarr)

    # Apply cuts
    stripe, idx, max_pt2 = phase_space.pt_cut_2of3(pa, pb, p1, p2, p3)
#     stripe, idx, max_pt2 = phase_space.pt_cut_3of3(pa, pb, p1, p2, p3)

    pa_0 = tf.boolean_mask(pa, stripe, axis=1)
    pb_0 = tf.boolean_mask(pb, stripe, axis=1)
    p1_0 = tf.boolean_mask(p1, stripe, axis=1)
    p2_0 = tf.boolean_mask(p2, stripe, axis=1)
    p3_0 = tf.boolean_mask(p3, stripe, axis=1)
    wgt_0 = tf.boolean_mask(wgt, stripe, axis=0)
    x1_0 = tf.boolean_mask(x1, stripe, axis=0)
    x2_0 = tf.boolean_mask(x2, stripe, axis=0)
    max_pt2 = tf.boolean_mask(max_pt2, stripe, axis=0)

    me_r = me.qq_h_r(pa_0, pb_0, p1_0, p2_0, p3_0)

    # Compute luminosity
    lumi = luminosity(x1_0, x2_0, max_pt2)
    phys_me = lumi*me_r*wgt_0/x1_0/x2_0
    phys_res = tf.scatter_nd(idx, phys_me, shape=xarr.shape[0:1])

    if SUBTRACT:
        # Now we need the subtraction terms for leg 1 and leg 2
        # leg 1, p3 is radiated from pa-p1
        npa, np1 = phase_space.map_3to2(pa, p1, p3)
        # apply cuts on this sbt
        stripe_1, x = phase_space.pt_cut_2of2(np1, p2)

        wgt_1 = tf.where(stripe_1, wgt, fzero)

        pa_1 = tf.boolean_mask(pa, stripe, axis=1)
        npa_1 = tf.boolean_mask(npa, stripe, axis=1)
        pb_1 = tf.boolean_mask(pb, stripe, axis=1)
        p1_1 = tf.boolean_mask(p1, stripe, axis=1)
        np1_1 = tf.boolean_mask(np1, stripe, axis=1)
        p2_1 = tf.boolean_mask(p2, stripe, axis=1)
        p3_1 = tf.boolean_mask(p3, stripe, axis=1)
        wgt_1 = tf.boolean_mask(wgt_1, stripe, axis=0)
        x1_1 = tf.boolean_mask(x1, stripe, axis=0)
        x2_1 = tf.boolean_mask(x2, stripe, axis=0)

        # Compute the dipole
        dip_1 = me.antenna_qgq(pa_1, p3_1, p1_1)
        # Reduced ME
        red_1 = me.partial_lo(npa_1, pb_1, np1_1, p2_1)

        # Compute luminosity of the subtraction
        lumi_1 = luminosity(x1_1, x2_1, max_pt2)
        sub_1 = lumi_1*dip_1*red_1*me.factor_re*wgt_1/x1_1/x2_1

        # Add this to the result
        phys_res -= tf.scatter_nd(idx, sub_1, shape=xarr.shape[0:1])

        # leg 2, p3 is radiated from pb-p2
        npb, np2 = phase_space.map_3to2(pb, p2, p3)
        # apply cuts on this sbt
        stripe_2, x = phase_space.pt_cut_2of2(p1, np2)

        wgt_2 = tf.where(stripe_2, wgt, fzero)

        pa_2 = tf.boolean_mask(pa, stripe, axis=1)
        pb_2 = tf.boolean_mask(pb, stripe, axis=1)
        npb_2 = tf.boolean_mask(npb, stripe, axis=1)
        p1_2 = tf.boolean_mask(p1, stripe, axis=1)
        p2_2 = tf.boolean_mask(p2, stripe, axis=1)
        np2_2 = tf.boolean_mask(np2, stripe, axis=1)
        p3_2 = tf.boolean_mask(p3, stripe, axis=1)
        wgt_2 = tf.boolean_mask(wgt_2, stripe, axis=0)
        x1_2 = tf.boolean_mask(x1, stripe, axis=0)
        x2_2 = tf.boolean_mask(x2, stripe, axis=0)

        # Compute the dipole
        dip_2 = me.antenna_qgq(pb_2, p3_2, p2_2)
        # Reduced ME
        red_2 = me.partial_lo(pa_2, npb_2, p1_2, np2_2)

        # Compute luminosity of the subtraction
        lumi_2 = luminosity(x1_2, x2_2, max_pt2)
        sub_2 = lumi_2*dip_2*red_2*me.factor_re*wgt_2/x1_2/x2_2
        # Add this to the result
        phys_res -= tf.scatter_nd(idx, sub_2, shape=xarr.shape[0:1])

    final_result = phys_res * flux
    return final_result


def vegas_integrate(integrand, ndim, niter, ncalls, events_limit):
    """ Wrapper for running Vegasflow """
    # Instantiate vegasflow
    vflow = VegasFlow(ndim, ncalls, events_limit=events_limit)
    # Compile the integrand
    vflow.compile(integrand, compilable=True)
    # Run and return
    return vflow.run_integration(niter)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-l", "--level", default="LO", help="Integration level")
    parser.add_argument("-n", "--nevents", type=int, default=int(1e6), help="Number of events")
    parser.add_argument("-i", "--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument(
        "-e",
        "--events_limit",
        type=int,
        default=int(1e6),
        help="Max events to be sent to an accelerator device at once",
    )
    args = parser.parse_args()

    ncalls = args.nevents
    niter = args.iterations
    print(f"ncalls={ncalls:2.1e}, niter={niter}, device_limit={args.events_limit:2.1e}")

    if args.level == "LO":
        print("Running Leading Order")
        ndim = 6
        integrand = vfh_production_leading_order
    elif args.level == "R":
        print("Running Real Tree level")
        ndim = 9
        integrand = vfh_production_real
    elif args.level == "NLO":
        print("Running Real NLO")
        ndim = 9
        integrand = vfh_production_nlo

    vegas_integrate(integrand, ndim, niter, ncalls, args.events_limit)
