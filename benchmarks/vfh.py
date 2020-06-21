#!/usr/bin/env python3
"""
Implementation of the Vector Boson Fusion Higgs production
using the hep-flow suite: pdfflow and vegasflow
"""
import subprocess as sp
import numpy as np

from pdfflow.pflow import mkPDF
from pdfflow.configflow import float_me, fone, fzero, DTYPE
from pdfflow.functions import _condition_to_idx
from vegasflow.vflow import vegas_wrapper

import tensorflow as tf

# Settings
tfloat1 = tf.TensorSpec(shape=[None], dtype=DTYPE)
tfloat4 = tf.TensorSpec(shape=[4, None], dtype=DTYPE)
# Integration parameters
ncalls = int(1e8)
niter = 10
ndim = 9
tech_cut = float_me(1e-7)
higgs_mass = float_me(125.0)
muR = tf.square(higgs_mass)
s_in = float_me(pow(8*1000, 2))
shat_min = tf.square(higgs_mass)*(1+tf.sqrt(tech_cut))
mw = float_me(80.379)
gw = float_me(2.085)
stw = float_me(0.22264585341299603)
min_pt = float_me(30) # GeV
# Select PDF
pdfset = "NNPDF31_nnlo_as_0118/0"
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE,  universal_newlines=True ).stdout.strip('\n') + '/'
pdf = mkPDF(pdfset, DIRNAME)
# Math parameters
tfpi = float_me(np.pi)
costhmax = fone
costhmin = -1.0*fone
phimax = float_me(2.0*np.pi)
phimin = fzero
fbGeV2=float_me(389379365600)

unit_phase = False
massive_boson = True
if not massive_boson:
    shat_min = tech_cut*s_in

RUN_LO = False
RUN_R = True

@tf.function(input_signature=[tfloat4])
def calc_zt2(pa):
    """ transverse momentum along the y axis """
    bb = tf.square(pa[1,:]) + tf.square(pa[3,:])
    return bb

@tf.function(input_signature=[tfloat4])
def calc_ap(pa):
    at2 = calc_zt2(pa)
    ap = pa[2,:] + pa[0,:]
    conditional_p = at2/(pa[0,:] - pa[2,:])
    ap = tf.where(ap < pa[0,:]/2.0, conditional_p, ap)
    return tf.complex(ap, fzero)

@tf.function(input_signature=[tfloat4, tfloat4, tf.TensorSpec(shape=[], dtype=bool)])
def zA(pa, pb, cross = False): # cross == when only one of (pa,pb) is initial-state
    ap = calc_ap(pa)
    bp = calc_ap(pb)
    ra = tf.complex(pa[1,:],-pa[3,:])*bp
    rb = tf.complex(pb[1,:],-pb[3,:])*ap
    zval = tf.complex(fzero, fone)*(ra-rb)/tf.sqrt(ap*bp)
    if not cross:
        return zval
    else:
        return zval*tf.complex(fzero, fone)

@tf.function(input_signature=[tfloat4, tfloat4, tf.TensorSpec(shape=[], dtype=bool)])
def zB(pa, pb, cross= False):
    return tf.math.conj(zA(pa, pb, cross=cross))


@tf.function(input_signature=[tfloat1, tfloat1])
def luminosity(x1, x2):
    """ Returns f(x1)*f(x2) """
    q2array = muR * tf.ones_like(x1)
    utype = pdf.xfxQ2([2,4], x1, q2array)
    dtype = pdf.xfxQ2([1,3], x2, q2array)
    lumi = tf.reduce_sum(utype*dtype, axis=1)
    return lumi/x1/x2

# PhaseSpace functions
@tf.function
def pick_within(r, valmin, valmax):
    """ Get a random value between valmin and valmax
    as given by the random number r (batch_size, 1)
    the outputs are val (batch_size, 1) and jac (batch_size, 1)
    """
    delta_val = (valmax-valmin)
    val = valmin + r*delta_val
    return val, delta_val

@tf.function
def log_pick(r, valmin, valmax):
    ratio_val = valmax/valmin
    val = valmin*tf.pow(ratio_val, r)
    jac = val*tf.math.log(ratio_val)
    return val, jac


@tf.function
def get_x1x2(xarr):
    """Receives two random numbers and return the
    value of the invariant mass of the center of mass
    as well as the jacobian of the x1,x2 -> tau-y transformation
    and the values of x1 and x2.

    The xarr array is of shape (batch_size, 2)
    """
    taumin = shat_min/s_in
    taumax = fone
    # Get tau logarithmically
    tau, wgt = log_pick(xarr[:,0], taumin, taumax)
    x1 = tf.pow(tau, xarr[:,1])
    x2 = tau/x1
    wgt *= -1.0*tf.math.log(tau)
    shat = x1*x2*s_in
    return shat, wgt, x1, x2


@tf.function
def sample_linear_all(x, nfspartons = 2):
    """ Receives an array of random numbers and samples the 
    invariant masses of the particles as well as the angles
    of the particles

    Uses 8 random numbers (from index 0 to 7)

    Samples:
    -------
        x1, x2: fraction of momenta of incoming partons
        s12: invariant mass of the system of outgoing massless partons
        cos12: angle theta of scattering between 1 and 2 
        phi12: angle phi of scattering between 1 and 2
    not used but calculated just in case:
        cosg1g2: angle theta of scattering between decay products of the Higgds
        phig1g2: angle phi of scattering between decay products of the Higgs
    """
    # Sample shat
    shat, wgt, x1, x2 = get_x1x2(x[:,0:2])
    if unit_phase:
        shat = s_in*tf.ones_like(shat)
        wgt = tf.ones_like(wgt)
        x1 = tf.ones_like(x1)
        x2 = tf.ones_like(x2)
    smin = tech_cut
    smax = shat
    
    fspartons = []
    # Assume no massive boson
    if massive_boson:
        # Detach the massive boson
        shiggs = tf.square(higgs_mass)
        wgt *= tfpi*(16.0*tfpi)
        # And the angles of its decay
        # (which for now are not to be used, but they 
        # do affect the weight)
        # this consumes random numbers 2, 3, 4
        wgt *= (costhmax-costhmin)
        wgt *= (phimax-phimin)
        wgt *= fone/(2.0*tfpi*32.0*tf.square(tfpi))
        # the remaining mass in the new smax
        roots = tf.sqrt(shat)
        smax = tf.square(roots - higgs_mass)
        # Now loop over the final state partons
        for i in range(1, nfspartons):
            j = i*3 + 2
            prev_smax = smax
            smax, jac = pick_within(x[:,j], smin, prev_smax)
            wgt *= jac
            cos12, jac = pick_within(x[:,j+1], costhmin, costhmax)
            wgt *= jac
            phi12, jac = pick_within(x[:,j+2], phimin, phimax)
            wgt *= jac
            wgt *= fone/(2.0*tfpi)
            fspartons.append((smax, cos12, phi12))
            wgt *=  fone/(32.0*tf.square(tfpi))
            if i > 1:
                wgt *= (prev_smax-smax)/prev_smax
    else:
        # Detach one particle
        s2, jac = pick_within(x[:,2], smin, smax)
        wgt *= jac
        # And the angles of its decay
        costh2, jac = pick_within(x[:,3], costhmin, costhmax)
        wgt *= jac
        phi2, jac = pick_within(x[:,4], phimin, phimax)
        wgt *= jac
        wgt *= fone/2.0/tfpi
        # For the first one there is no lambda to be computed
        wgt *= fone/32.0/tf.square(tfpi)
        # Detach the next particle and the angles of its decay
        s3, jac = pick_within(x[:,5], smin, s2)
        wgt *= jac
        costh3, jac = pick_within(x[:,6], costhmin, costhmax)
        wgt *= jac
        phi3, jac = pick_within(x[:,7], phimin, phimax)
        wgt *= jac
        wgt *= fone/2.0/tfpi
        wgt *= tf.sqrt(dlambda(s2, fzero, s3))/s2/32.0/tf.square(tfpi)
        s12 = s2
        shiggs = s3
        cos12 = costh3
        phi12 = phi3
        fspartons.append((s12, cos12, phi12))
    return x1, x2, shat, shiggs, fspartons , wgt

@tf.function
def dlambda(a, b, c):
    return a*a + b*b + c*c -2.0*(a*b +a*c +b*c)

@tf.function
def pcommon2to2(r, shat, s1, s2):
    """ Receives a random number (r) and shat=(pa+pb)^2
    and the invariant masses of p1 and p2
    and returns pa, pb, p1, p2
    The shape of r is (batch_size, 1)
    all the momenta (p) is of size (4, batch_size)
    """
    roots = tf.sqrt(shat)
    Eab = roots/2.0
    pin = Eab
    E1 = (shat + s1 - s2)/2.0/roots
    E2 = (shat + s2 - s1)/2.0/roots
    pout = tf.sqrt(dlambda(shat,s1,s2))/2.0/roots
    # Pick cosine p1-beam
    ta1min = s1 - 2.0*Eab*E1 - 2.0*pin*pout
    ta1max = s1 - 2.0*Eab*E1 + 2.0*pin*pout
    ta1, wgt = pick_within(r, -ta1max, -ta1min)
    costh = (-ta1 - s1 + 2.0*Eab*E1)/(2.0*pin*pout)
    # Check that the cosine is not greater than 1 at this point
    # nor less than -1
    sinth = tf.sqrt(fone - tf.square(costh))
    wgt *= fone/(16.0*tfpi*tfpi*shat)

    # Since there are rotational symmetry around the beam axis
    # we can set the phi angle to 0.0
    cosphi = 1.0
    sinphi = 0.0
    wgt *= 2.0*tfpi

    # Now generate all the momenta
    zeros = tf.zeros_like(r)
    pa = tf.stack([Eab, zeros, zeros, pin])
    pb = tf.stack([Eab, zeros, zeros, -pin])

    px = pout*sinth*cosphi
    py = zeros # sinphi = 0.0
    pz = pout*costh

    p1 = tf.stack([E1, -px, -py, -pz])
    p2 = tf.stack([E2, px, py, pz])

    return pa, pb, p1, p2, wgt

def pcommon1to2(sin, pin, s1, s2, costh, phi):
    """ Receives the input invariant mass and the momentum in the lab frame
    and generates p1 and p2 in the lab frame.
    Needs also the invariant masses of 1 and 2 as well as the angles of the 12->1,2 decay
    First they are generated in the p12 com frame and then boosted back
    """
    sinth = tf.sqrt(fone - tf.square(costh))
    cosphi = tf.cos(phi)
    sinphi = tf.sin(phi)

    roots = tf.sqrt(sin)
    E1 = (sin + s1 - s2)/2.0/roots
    E2 = (sin + s2 - s1)/2.0/roots
    roots1 = tf.sqrt(s1)
    pp = tf.sqrt( (E1-roots1)*(E1+roots1) )

    px = pp*sinth*cosphi
    py = pp*sinth*sinphi
    pz = pp*costh
    
    p1 = tf.stack([E1, px, py, pz])
    p2 = tf.stack([E2, -px, -py, -pz])

    # Now boost both p1 and p2 back to the lab frame
    # Construct the boosting matrix
    gamma = pin[0,:]/roots
    vx = -pin[1,:]/pin[0,:]
    vy = -pin[2,:]/pin[0,:]
    vz = -pin[3,:]/pin[0,:]
    v2 = vx*vx + vy*vy + vz*vz

    omgdv = (gamma-fone)/v2
    bmatE = tf.stack([gamma, -gamma*vx, -gamma*vy,-gamma*vz])
    bmatx = tf.stack([-gamma*vx, omgdv*vx*vx+fone, omgdv*vx*vy, omgdv*vx*vz])
    bmaty = tf.stack([-gamma*vy, omgdv*vy*vx, omgdv*vy*vy+fone, omgdv*vy*vz])
    bmatz = tf.stack([-gamma*vz, omgdv*vz*vx, omgdv*vz*vy, omgdv*vz*vz+fone])
    bmat = tf.stack([bmatE, bmatx, bmaty, bmatz])

    # Now unboost
    bmatt = tf.transpose(bmat)
    p1t = tf.transpose(p1)
    p2t = tf.transpose(p2)
    up1t = tf.keras.backend.batch_dot(p1t, bmatt)
    up2t = tf.keras.backend.batch_dot(p2t, bmatt)

    return tf.transpose(up1t), tf.transpose(up2t)

def psgen_2to3(xarr):
    """ Generates a 2 -> 4 phase space 
    where the two last particles are the results of the decay of the massive higgs
    for the Matrix Element this is equivalent to pa pb -> p1, p2, pH 
    where p1 and p2 are massless and pH massive

    uses 9 random numbers

    Convention:
        p = (px, py, pz, E)
    """
    x1, x2, shat, sh, fspartons, wgt = sample_linear_all(xarr[:, 0:8], nfspartons = 2)
    s12, cos12, phi12 = fspartons[0]
    if massive_boson:
        pa, pb, ph, p12, jac = pcommon2to2(xarr[:, 8], shat, sh, s12)
        p1, p2 = pcommon1to2(s12, p12, fzero, fzero, cos12, phi12)
    else:
        pa, pb, p1, p2h, jac = pcommon2to2(xarr[:,8], shat, 0.0, s12)
        p2, ph = pcommon1to2(s12, p2h, fzero, s12, cos12, phi12)
    wgt *= jac
    return pa, pb, p1, p2, ph, x1, x2, wgt

def psgen_2to4(xarr):
    x1, x2, shat, sh, fspartons, wgt = sample_linear_all(xarr[:, 0:11], nfspartons = 3)
    if massive_boson:
        s123, cos123, phi123 = fspartons[0]
        pa, pb, ph, p123, jac = pcommon2to2(xarr[:, 11], shat, sh, s123)
        s23, cos23, phi23 = fspartons[1]
        p1, p23 = pcommon1to2(s123, p123, fzero, s23, cos123, phi123)
        p2, p3 = pcommon1to2(s23, p23, fzero, fzero, cos23, phi23)
    else:
        raise NotImplementedError("Not implemented @ psgen_2to4")
    wgt *= jac

    # Check the values that actually pass the cuts
    min_pt2 = tf.square(min_pt)
    p1t2 = tf.greater_equal(calc_pt2(p1), min_pt2)
    p2t2 = tf.greater_equal(calc_pt2(p2), min_pt2)
    p3t2 = tf.greater_equal(calc_pt2(p3), min_pt2)

    stripe, idx = _condition_to_idx(tf.logical_and(p1t2, p2t2), p3t2)
    # Mask all the outputs
    pa = tf.boolean_mask(pa, stripe, axis=1)
    pb = tf.boolean_mask(pb, stripe, axis=1)
    p1 = tf.boolean_mask(p1, stripe, axis=1)
    p2 = tf.boolean_mask(p2, stripe, axis=1)
    p3 = tf.boolean_mask(p3, stripe, axis=1)
    x1 = tf.boolean_mask(x1, stripe)
    x2 = tf.boolean_mask(x2, stripe)
    wgt = tf.boolean_mask(wgt, stripe)
    return pa, pb, p1, p2, p3, ph, x1, x2, wgt, idx

@tf.function(input_signature=[tfloat4]*2)
def dot_product(par, pbr):
    pa = tf.transpose(par)
    pb = tf.transpose(pbr)
    ener = pa[:,0]*pb[:,0]
    mome = tf.keras.backend.batch_dot(pa[:,1:4], pb[:,1:4])[:,0]
    return ener-mome

@tf.function(input_signature=[tfloat1])
def propagator_w(s):
    t1 = tf.square(s - tf.square(mw))
    t2 = tf.square(mw*gw)
    return t1+t2

factor_lo = float_me(1.0702411577062499e-4)
@tf.function
def qq_h_lo(pa, pb, p1, p2):
    """ Computes the LO q Q -> Q q H (WW->H) """
    # Compute the propagators
    sa1 =-2.0*dot_product(pa, p1)
    sb2 =-2.0*dot_product(pb, p2)

    prop = propagator_w(sa1)*propagator_w(sb2)
    coup = tf.square(mw/tf.pow(stw, 1.5))
    rmcom = coup/prop
    
    # Compute the amplitude
    # W-boson, so only Left-Left
    amplitude = zA(pa,pb)*zB(p1,p2)
    amp2 = tf.math.real(amplitude*tf.math.conj(amplitude))

    me_res = 2.0*amp2*rmcom
    return factor_lo*me_res

def zprod(pa,pb):
    return tf.math.real(zA(pa, pb)*zB(pa,pb))

def sprod(pa,pb):
    pp = pa+pb
    return dot_product(pp,pp)

@tf.function(input_signature=[tfloat4]*5)
def partial_qq_h_qQg(pa, pb, p1, p2, p3):
    """ Gluon radiated from leg pa-p1 """
    pa13 = pa - (p1+p3)
    sa13 = dot_product(pa13, pa13)
    sb2 =-2.0*dot_product(pb, p2)
    prop = propagator_w(sa13)*propagator_w(sb2)
    coup = tf.square(mw/tf.pow(stw, 1.5))
    rmcom = coup/prop

    # compute the amplitude
    zUp = (zB(pa,p1, cross=True)*zA(p2,p1) + zB(pa,p3, cross=True)*zA(p2,p3))
    zFp = zB(pb,pa)#(zB(p1,p3)*zB(pa,p3,cross=True))
    zAp = zFp*zUp

    zUm = (zA(pa,p1, cross=True)*zB(pa,pb) + zB(pb,p3, cross=True)*zA(p1,p3))
    zFm = zA(p1,p2)#(zA(p1,p3)*zA(pa,p3,cross=True))
    zAm = zFm*zUm

    s13 = 2.0*dot_product(p1,p3)
    sa3 = 2.0*dot_product(pa,p3)

    zamp2 = (zAp*tf.math.conj(zAp) + zAm*tf.math.conj(zAm))
    amp = tf.math.real(zamp2)/s13/sa3

    return amp*rmcom

factor_re = float_me(4.0397470069216974E-004)
@tf.function
def qq_h_qQg(pa, pb, p1, p2, p3):
    """ Computes q Q -> Q q g H (WW -> H)
    Q = p1
    q = p2
    g = p3
    """
    r1 = partial_qq_h_qQg(pb, pa, p2, p1, p3)
    r2 = partial_qq_h_qQg(pa, pb, p1, p2, p3)
    return (r1+r2)*factor_re

@tf.function
def calc_pt2(p):
    pxpy2 = tf.square(p[1:3, :])
    pt2 = tf.reduce_sum(pxpy2, axis=0)
    return pt2

@tf.function # not used at R
def pt_cut(p, wgt):
    """ Returns wgt when pt > min_pt
    and 0 when pt < min_pt """
    pt2 = calc_pt2(p)
    comp = pt2 > tf.square(min_pt)
    return tf.where(comp, wgt, fzero)


@tf.function
def vfh_production_lo(xarr, n_dim = None, **kwars):
    """ Wrapper for the VFH calculation """
    pa, pb, p1, p2, _, x1, x2, wgt = psgen_2to3(xarr)

    lumi = luminosity(x1, x2)
    me_lo = qq_h_lo(pa, pb, p1, p2)
    if unit_phase:
        return wgt
    # set to 0 weights in which pt < ptcut
    wgt = pt_cut(p1, wgt)
    wgt = pt_cut(p2, wgt)
    flux = fbGeV2/2.0/(s_in*x1*x2)
    res = lumi*me_lo*wgt
    return res*flux

@tf.function
def vfh_production_r(xarr, n_dim = None, **kwars):
    pa, pb, p1, p2, p3, _, x1, x2, wgt, idx = psgen_2to4(xarr)
    if unit_phase:
        return wgt
# 
#     x1 = np.ones_like(x1)
#     x2 = np.ones_like(x2)
#     pa = np.zeros_like(pa)
#     pb = np.zeros_like(pb)
#     p1 = np.zeros_like(p1)
#     p2 = np.zeros_like(p2)
#     p3 = np.zeros_like(p3)
# 
# 
#     x1 *=  0.51306089926047227
#     x2 *=  5.2644661467767841E-002
# 
#     pa[  1 ,:] = 0.0000000000000000
#     pb[  1 ,:] =-0.0000000000000000
#     p1[  1 ,:] = 75.238092924633293
#     p2[  1 ,:] = 129.36393855989468
#     p3[  1 ,:] = 372.29394411611491
#     pa[  2 ,:] = 0.0000000000000000
#     pb[  2 ,:] =-0.0000000000000000
#     p1[  2 ,:] =-6.9092298883074905
#     p2[  2 ,:] = 35.399743675152976
#     p3[  2 ,:] =-28.490513786845483
#     pa[  3 ,:] = 657.38776811152911
#     pb[  3 ,:] =-657.38776811152911
#     p1[  3 ,:] = 3.6286971300483559
#     p2[  3 ,:] = 150.92635439952076
#     p3[  3 ,:] =-235.45860221519800
#     pa[  0 ,:] = 657.38776811152911
#     pb[  0 ,:] = 657.38776811152911
#     p1[  0 ,:] = 75.641757828905796
#     p2[  0 ,:] = 201.90823386955890
#     p3[  0 ,:] = 441.42410849262205


    lumi = luminosity(x1, x2)
    me_r = qq_h_qQg(pa, pb, p2, p3, p1)
    # set to 0 weights in which pt < ptcut
    flux = fbGeV2/2.0/(s_in*x1*x2)
    res = wgt*lumi*me_r
    final_result = res*flux
    return tf.scatter_nd(idx, final_result, shape = xarr.shape[0:1])


if __name__ == "__main__":
    print(f"Vegas MC VFH NLO production")
    if RUN_LO:
        print("Running Leading Order")
        print(f"ncalls={ncalls}, niter={niter}")
        res = vegas_wrapper(vfh_production_lo, ndim, niter, ncalls, compilable=True)
    if RUN_R:
        print("Running Real Correction")
        print(f"ncalls={ncalls}, niter={niter}")
        res = vegas_wrapper(vfh_production_r, ndim+3, niter, ncalls, compilable=True)

