#!/usr/bin/env python3
"""
Implementation of the Vector Boson Fusion Higgs production
using the hep-flow suite: pdfflow and vegasflow
"""
import subprocess as sp
import numpy as np

from pdfflow.pflow import mkPDF
from pdfflow.configflow import float_me, fone, fzero
from vegasflow.vflow import vegas_wrapper

import tensorflow as tf

# Settings
# Integration parameters
ncalls = int(1e6)
niter = 10
ndim = 9
tech_cut = float_me(1e-5)
higgs_mass = float_me(125.0)
muR = tf.square(higgs_mass)
s_in = float_me(pow(8*1000, 2))
shat_min = tf.square(higgs_mass) + tech_cut*s_in
mw = float_me(80.379)
gw = float_me(2.085)
stw = float_me(0.22264585341299603)
# Select PDF
pdfset = "NNPDF31_nlo_as_0118/0"
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE,  universal_newlines=True ).stdout.strip('\n') + '/'
pdf = mkPDF(pdfset, DIRNAME)
# Math parameters
tfpi = float_me(np.pi)
costhmax = fone
costhmin = -fone
phimax = float_me(2.0*np.pi)
phimin = fzero
fbGeV2=float_me(389379365600)

unit_phase = True
massive_boson = False
if not massive_boson:
    shat_min = tech_cut*s_in

@tf.function
def luminosity(x1, x2):
    """ Returns f(x1)*f(x2) """
    q2array = tf.fill(x1.shape, muR)
    utype = pdf.xfxQ2([2], x1, q2array)
    dtype = pdf.xfxQ2([1], x2, q2array)
    return utype*dtype/x1/x2

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
def get_x1x2(xarr):
    """Receives two random numbers and return the
    value of the invariant mass of the center of mass
    as well as the jacobian of the x1,x2 -> tau-y transformation
    and the values of x1 and x2.

    The xarr array is of shape (batch_size, 2)
    """
    # building shat
    bmax = tf.sqrt(1 - shat_min / s_in)
    b = bmax * xarr[:, 0]
    onemb2 = 1 - tf.square(b)
    shat = shat_min / onemb2
    tau = shat / s_in
    # building rapidity
    ymax = -0.5 * tf.math.log(tau)
    y = ymax * (2.0 * xarr[:, 1] - 1)
    # building jacobian
    jac = 2.0 * tau * b * bmax / onemb2  # tau
    jac *= 2.0 * ymax  # y
    # building x1 and x2
    sqrttau = tf.sqrt(tau)
    expy = tf.exp(y)
    x1 = sqrttau * expy
    x2 = sqrttau / expy
    return shat, jac, x1, x2

@tf.function
def sample_linear_all(x):
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
    smin = tech_cut*shat
    smax = shat

    # Assume no massive boson
    if not massive_boson:
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
    return x1, x2, shat, shiggs, s12, cos12, phi12, wgt

#     # "detach" the massive boson (the Higgs)
#     if massive_boson:
#         shiggs = tf.square(higgs_mass)
#         wgt *= tf.square(tfpi)*float_me(16.0)
#         # consumes x[:,2]
#         # decay of the Higgs into two photons
#         wgt *= 2.0*(2.0*tfpi)/64.0/tf.pow(tfpi,3)
#         # consumnes x[:,3:5]
#     else:
#         smax = shat - smin
#         shiggs, jac = pick_within(x[:,2], smin, smax)
#         wgt *= jac
#         wgt *= 2.0 # pick cosgamma with x[:,3]
#         wgt *= 2.0*tfpi # pick phigamma with x[:,4]
#         wgt *= fone/2.0/tfpi
#         wgt *= dlambda(shat, 0.0,  
# 
# 
#     # deatch the 1-2 system
#     sqrtshat = tf.sqrt(shat)
#     smax = tf.pow(sqrtshat-tf.sqrt(shiggs), 2)
#     s12, jac = pick_within(x[:,5], smin, smax)
#     wgt *= jac
#     cos12, jac = pick_within(x[:,6], costhmin, costhmax)
#     wgt *= jac
#     phi12, jac = pick_within(x[:,7], phimin, phimax)
#     wgt *= jac
# 
#     # finally the lambda(shat, sh, s12) stuff...
#     wgt *= fone/(float_me(32.0)*tf.square(tfpi))
#     
#     # return everything
#     return x1, x2, shat, shiggs, s12, cos12, phi12, wgt

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
    pa = tf.stack([zeros, zeros, pin, Eab])
    pb = tf.stack([zeros, zeros, -pin, Eab])

    px = pout*sinth*cosphi
    py = zeros # sinphi = 0.0
    pz = pout*costh

    p1 = tf.stack([px, py, pz, E1])
    p2 = tf.stack([-px, -py, -pz, E2])

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
    
    p1 = tf.stack([px, py, pz, E1])
    p2 = tf.stack([-px, -py, -pz, E2])

    # Now boost both p1 and p2 back to the lab frame
    # Construct the boosting matrix
    gamma = pin[3,:]/roots
    vx = -pin[0,:]/pin[3,:]
    vy = -pin[1,:]/pin[3,:]
    vz = -pin[2,:]/pin[3,:]
    v2 = vx*vx + vy*vy + vz*vz

    omgdv = (fone - gamma)/v2
    bmatx = tf.stack([omgdv*vx*vx, omgdv*vx*vy, omgdv*vx*vz, gamma*vx])
    bmaty = tf.stack([omgdv*vy*vx, omgdv*vy*vy, omgdv*vy*vz, gamma*vy])
    bmatz = tf.stack([omgdv*vz*vx, omgdv*vz*vy, omgdv*vz*vz, gamma*vz])
    bmatE = tf.stack([gamma*vx, gamma*vy, gamma*vz, gamma])
    bmat = tf.stack([bmatx, bmaty, bmatz, bmatE])

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
    x1, x2, shat, sh, s12, cos12, phi12, wgt = sample_linear_all(xarr[:, 0:8])
    if massive_boson:
        pa, pb, ph, p12, jac = pcommon2to2(xarr[:, 8], shat, sh, s12)
        p1, p2 = pcommon1to2(s12, p12, fzero, fzero, cos12, phi12)
    else:
        pa, pb, p1, p2h, jac = pcommon2to2(xarr[:,8], shat, 0.0, s12)
        p2, ph = pcommon1to2(s12, p2h, fzero, s12, cos12, phi12)
    wgt *= jac
    return pa, pb, p1, p2, ph, x1, x2, shat, wgt

@tf.function
def propagator_w(s):
    t1 = tf.square(s - tf.square(mw))
    t2 = tf.square(mw*gw)
    return t1+t2

@tf.function
def qq_h_lo(pa, pb, p1, p2, pH):
    """ Computes the LO q Q -> Q q H (WW->H) """
    # Compute the propagator
    pa1 = tf.transpose(pa-p1)
    pb2 = tf.transpose(pb-p2)

    sa1 = tf.keras.backend.batch_dot(pa1, pa1)[:,0]
    sb2 = tf.keras.backend.batch_dot(pa1, pa1)[:,0]

    prop = propagator_w(sa1)*propagator_w(sb2)
    coup = tf.sqrt(tf.square(mw)/tf.pow(stw, 3))
    
    # Compute the amplitude
    # W-boson, so only Left-Left
    pab = tf.transpose(pa+pb)
    sab = tf.keras.backend.batch_dot(pab, pab)[:,0]
    p12 = tf.transpose(p1+p2)
    s12 = tf.keras.backend.batch_dot(p12, p12)[:,0]
    
    res = tf.square(s12*sab)

    return 2.0*res/prop*coup

def vfh_production(xarr, n_dim = None, **kwars):
    """ Wrapper for the VFH calculation """
    pa, pb, p1, p2, pH, x1, x2, shat, wgt = psgen_2to3(xarr)
    lumi = luminosity(x1, x2)
    me_lo = qq_h_lo(pa, pb, p1, p2, pH)
    res = lumi*me_lo*wgt/tf.square(s_in)
    if unit_phase:
        return wgt
    return res*fbGeV2

if __name__ == "__main__":
    print(f"Vegas MC VFH LO production")
    print(f"ncalls={ncalls}, niter={niter}")
    res = vegas_wrapper(vfh_production, ndim, niter, ncalls, compilable=False)
    print("Finished!")
