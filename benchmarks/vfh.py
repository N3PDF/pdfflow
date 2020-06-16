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
ncalls = int(1e5)
niter = 5
ndim = 9
tech_cut = float_me(1e-5)
higgs_mass = float_me(125.0)
muR = tf.square(higgs_mass)
s_in = float_me(pow(7*1000, 2))
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
    y = ymax * (2 * xarr[:, 1] - 1)
    # building jacobian
    jac = 2 * tau * b * bmax / onemb2  # tau
    jac *= 2 * ymax  # y
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
    """
    # Sampele shat
    shat, wgt, x1, x2 = get_x1x2(x[:,0:2])

    # "detach" the massive boson (the Higgs)
    smin = tech_cut*shat
    smax = shat
    shiggs = tf.square(higgs_mass)
    wgt *= tf.square(tfpi)*float_me(16.0)

    # deatch the 1-2 system
    sqrtshat = tf.sqrt(shat)
    smax = tf.pow(sqrtshat-higgs_mass, 2)
    s12, jac = pick_within(x[:,2], smin, smax)
    wgt *= jac
    cos12, jac = pick_within(x[:,3], costhmin, costhmax)
    wgt *= jac
    phi12, jac = pick_within(x[:,4], phimin, phimax)
    wgt *= jac
    
    # finally the lambda(shat, sh, s12) stuff...
    wgt *= fone/(float_me(32.0)*tf.square(tfpi))
    
    # return everything
    return x1, x2, shat, shiggs, s12, cos12, phi12, wgt

@tf.function
def dlambda(a, b, c):
    return a*a + b*b + c*c -2.0*(a*b +a*c +b*c)

@tf.function
def pcommon2to2(r, shat, s1, s2):
    """ Receives a random number (r) and shat=(pa+pb)^2
    and the invariant masses of p1 and p2
    and returns pa, pb, p1, p2.
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
    wgt = fone/(16.0*tfpi*tfpi*shat)

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
    """ Generates a 2 -> 3 phase space 
    where one particle is massive

    Convention:
        p = (px, py, pz, E)
    """
    x1, x2, shat, sh, s12, cos12, phi12, jac = sample_linear_all(xarr[:, 0:5])
    pa, pb, ph, p12, wgt = pcommon2to2(xarr[:, 5], shat, sh, s12)
    p1, p2 = pcommon1to2(s12, p12, fzero, fzero, cos12, phi12)
    return pa, pb, p1, p2, ph, x1, x2, shat, wgt*jac

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
    return tf.reduce_sum(res)*fbGeV2

if __name__ == "__main__":
    print(f"Vegas MC VFH LO production")
    print(f"ncalls={ncalls}, niter={niter}")
    res = vegas_wrapper(vfh_production, ndim, niter, ncalls, compilable=False)
    print("Finished!")
