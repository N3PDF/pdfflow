"""
Phase Space calcuation for VFH integration

The wrapper interface functions are

    psgen_2to3
    psgen_2to4

The convention of the 4 momenta is such that 
    p[0] = E
    p[1:4] = px, py, pz
"""
from pdfflow.configflow import float_me, fone, fzero, DTYPE
from pdfflow.functions import _condition_to_idx
import numpy as np
import tensorflow as tf
from parameters import TFLOAT1, TFLOAT4, shat_min, s_in, higgs_mass, pt2_cut, mjj_cut

# Control flags
UNIT_PHASE = False
MASSIVE_BOSON = True

# Technical settings
TECH_CUT = 1e-7

# Constants
tfpi = float_me(np.pi)
costhmax = fone
costhmin = -1.0 * fone
phimin = fzero
phimax = 2.0 * tfpi

# Cut function
@tf.function
def pt2(fourp):
    """ Returns px^2 + py^2 """
    return tf.square(fourp[1, :]) + tf.square(fourp[2, :])


@tf.function
def pt_cut_2of2(p1, p2):
    """ Ensures that both p1 and p2 pass the pt_cut
    returns a boolean mask and the list of true indices
    """
    p1pass = pt2(p1) > pt2_cut
    p2pass = pt2(p2) > pt2_cut
    stripe, idx = _condition_to_idx(p1pass, p2pass)
    return stripe, idx


# Utility functions
@tf.function
def dlambda(a, b, c):
    """ Computes dlamba(a,b,c) """
    return a * a + b * b + c * c - 2.0 * (a * b + a * c + b * c)


@tf.function
def pick_within(r, valmin, valmax):
    """ Get a random value between valmin and valmax
    as given by the random number r (batch_size, 1)
    the outputs are val (batch_size, 1) and jac (batch_size, 1)

    Linear sampling

    Parameters
    ----------
        r: random val
        valmin: minimum value
        valmax: maximum value
    Returns
    -------
        val: chosen random value
        jac: jacobian of the transformation
    """
    delta_val = valmax - valmin
    val = valmin + r * delta_val
    return val, delta_val


@tf.function
def log_pick(r, valmin, valmax):
    """ Get a random value between valmin and valmax
    as given by the random number r (batch_size, 1)
    the outputs are val (batch_size, 1) and jac (batch_size, 1)
    Logarithmic sampling

    Parameters
    ----------
        r: random val
        valmin: minimum value
        valmax: maximum value
    Returns
    -------
        val: chosen random value
        jac: jacobian of the transformation
    """
    ratio_val = valmax / valmin
    val = valmin * tf.pow(ratio_val, r)
    jac = val * tf.math.log(ratio_val)
    return val, jac


##############################################################


@tf.function
def get_x1x2(xarr):
    """Receives two random numbers and return the
    value of the invariant mass of the center of mass
    as well as the jacobian of the x1,x2 -> tau-y transformation
    and the values of x1 and x2.

    The xarr array is of shape (batch_size, 2)
    """
    taumin = shat_min / s_in
    taumax = fone
    # Get tau logarithmically
    tau, wgt = log_pick(xarr[:, 0], taumin, taumax)
    x1 = tf.pow(tau, xarr[:, 1])
    x2 = tau / x1
    wgt *= -1.0 * tf.math.log(tau)
    shat = x1 * x2 * s_in
    return shat, wgt, x1, x2


@tf.function
def sample_linear_all(x, nfspartons=2):
    """ Receives an array of random numbers and samples the
    invariant masses of the particles as well as the angles
    of the particles

    Uses 3*nfspartons + 2 random numbers (from index 0 to 3*nfspartons + 1)

    Parameters
    ----------
        x: tensor (nevents,)
            Input random numbers from the integration routien
        nfspartons: int
            Number of partons in the final state

    Returns
    -------
        x1: tensor (nevents,)
            Momentum fraction of incoming parton 1
        x2: tensor (nevents,)
            Momentum fraction of incoming parton 1
        shat: tensor (nevents,)
            Incoming invariant mass
        shiggs: tensor (nevents,)
            Invariant mass of the higgs boson
        fspartons: list( list( (nevents, nevents, nevents) ) )
            For each of the detachments returns the invariant mass
            and the angles of the decaya
            i.e., for a 2->3 + H phase space it will return 2 items:
                item 1: [s123, costh1,23, phi1,23]
                item 2: [s23, costh2,3, phi2,3]
    """
    # Sample shat and the incoming momentum fractions
    shat, wgt, x1, x2 = get_x1x2(x[:, 0:2])
    if UNIT_PHASE:
        shat = s_in * tf.ones_like(shat)
        wgt = tf.ones_like(wgt)
        x1 = tf.ones_like(x1)
        x2 = tf.ones_like(x2)

    smin = TECH_CUT
    smax = shat

    fspartons = []
    # Assume no massive boson
    if MASSIVE_BOSON:
        # Detach the massive boson
        shiggs = tf.square(higgs_mass)
        wgt *= tfpi * (16.0 * tfpi)
        # And the angles of its decay
        # (which for now are not to be used, but they
        # do affect the weight)
        # this consumes random numbers 2, 3, 4
        wgt *= costhmax - costhmin
        wgt *= phimax - phimin
        wgt *= fone / (2.0 * tfpi * 32.0 * tf.square(tfpi))
        # the remaining mass in the new smax
        roots = tf.sqrt(shat)
        smax = tf.square(roots - higgs_mass)
        # Now loop over the final state partons
        for i in range(1, nfspartons):
            j = i * 3 + 2
            prev_smax = smax
            smax, jac = pick_within(x[:, j], smin, prev_smax)
            wgt *= jac
            cos12, jac = pick_within(x[:, j + 1], costhmin, costhmax)
            wgt *= jac
            phi12, jac = pick_within(x[:, j + 2], phimin, phimax)
            wgt *= jac
            wgt *= fone / (2.0 * tfpi)
            fspartons.append((smax, cos12, phi12))
            wgt *= fone / (32.0 * tf.square(tfpi))
            if i > 1:
                wgt *= (prev_smax - smax) / prev_smax
    #     else:
    #         # Detach one particle
    #         s2, jac = pick_within(x[:, 2], smin, smax)
    #         wgt *= jac
    #         # And the angles of its decay
    #         costh2, jac = pick_within(x[:, 3], costhmin, costhmax)
    #         wgt *= jac
    #         phi2, jac = pick_within(x[:, 4], phimin, phimax)
    #         wgt *= jac
    #         wgt *= fone / 2.0 / tfpi
    #         # For the first one there is no lambda to be computed
    #         wgt *= fone / 32.0 / tf.square(tfpi)
    #         # Detach the next particle and the angles of its decay
    #         s3, jac = pick_within(x[:, 5], smin, s2)
    #         wgt *= jac
    #         costh3, jac = pick_within(x[:, 6], costhmin, costhmax)
    #         wgt *= jac
    #         phi3, jac = pick_within(x[:, 7], phimin, phimax)
    #         wgt *= jac
    #         wgt *= fone / 2.0 / tfpi
    #         wgt *= tf.sqrt(dlambda(s2, fzero, s3)) / s2 / 32.0 / tf.square(tfpi)
    #         s12 = s2
    #         shiggs = s3
    #         cos12 = costh3
    #         phi12 = phi3
    #         fspartons.append((s12, cos12, phi12))
    return x1, x2, shat, shiggs, fspartons, wgt


@tf.function
def pcommon2to2(r, shat, s1, s2):
    """ Generates a 2 to 2 phase space in the c.o.m. of pa+pb

        pa + pb ----> p1 + p2

    Where:
        (pa + pb)^2 = shat
        p1^2 = s1
        p2^2 = s2

    Parameters
    ----------
        r: tensor (nevents)
            random number to generate a scattering angle
        shat: tensor(nevents)
            invariant mass of the incoming system
        s1: tensor(nevents)
            invariant mass of the outgoing particle 1
        s2: tensor(nevents)
            invariant mass of the outgoing particle 2

    Returns
    -------
        pa: tensor (4, nevents)
            incoming 4-momenta of parton a
        pb: tensor (4, nevents)
            incoming 4-momenta of parton b
        p1: tensor (4, nevents)
            outgoing 4-momenta of parton 1
        p2: tensor (4, nevents)
            outgoing 4-momenta of parton 2
        wgt: tensor (nevents,)
            weight of the generated phase space point
    """
    roots = tf.sqrt(shat)
    Eab = roots / 2.0
    pin = Eab
    E1 = (shat + s1 - s2) / 2.0 / roots
    E2 = (shat + s2 - s1) / 2.0 / roots
    pout = tf.sqrt(dlambda(shat, s1, s2)) / 2.0 / roots
    # Pick cosine p1-beam
    ta1min = s1 - 2.0 * Eab * E1 - 2.0 * pin * pout
    ta1max = s1 - 2.0 * Eab * E1 + 2.0 * pin * pout
    ta1, wgt = pick_within(r, -ta1max, -ta1min)
    costh = (-ta1 - s1 + 2.0 * Eab * E1) / (2.0 * pin * pout)
    # Check that the cosine is not greater than 1 at this point
    # nor less than -1
    sinth = tf.sqrt(fone - tf.square(costh))
    wgt *= fone / (16.0 * tfpi * tfpi * shat)

    # Since there are rotational symmetry around the beam axis
    # we can set the phi angle to 0.0
    #     cosphi = 1.0
    #     sinphi = 0.0
    wgt *= 2.0 * tfpi

    # Now generate all the momenta
    zeros = tf.zeros_like(r)
    pa = tf.stack([Eab, zeros, zeros, pin])
    pb = tf.stack([Eab, zeros, zeros, -pin])

    px = pout * sinth  # cosphi = 1.0
    py = zeros  # sinphi = 0.0
    pz = pout * costh

    p1 = tf.stack([E1, -1.0 * px, -1.0 * py, -1.0 * pz])
    p2 = tf.stack([E2, px, py, pz])

    return pa, pb, p1, p2, wgt


@tf.function
def pcommon1to2(sin, pin, s1, s2, costh, phi):
    """ Generates a 1 -> 2 phase space in the c.o.m. of 1
      
        p_in -> p_1 + p_2

    Parameters
    ----------
        sin: tensor(nevents,)
            ivariant mass of particle in
        pin: tensor(4, nevents,)
            4-momenta of particle in
        s1: tensor(4, nevents,)
            invariant mass of particle 1
        s2: tensor(4, nevents,)
            invariant mass of particle 2
        costh: tensor(nevent,)
            theta angle of the 1->2 decay
        phi: tensor(nevent,)
            phi angle of the 1->2 decay

    Returns
    ------
        p1: tensor(4, nevents)
            4-momenta of particle 1
        p2: tensor(4, nevents)
            4-momenta of particle 2
    """
    sinth = tf.sqrt(fone - tf.square(costh))
    cosphi = tf.cos(phi)
    sinphi = tf.sin(phi)

    roots = tf.sqrt(sin)
    E1 = (sin + s1 - s2) / 2.0 / roots
    E2 = (sin + s2 - s1) / 2.0 / roots
    roots1 = tf.sqrt(s1)
    pp = tf.sqrt((E1 - roots1) * (E1 + roots1))

    px = pp * sinth * cosphi
    py = pp * sinth * sinphi
    pz = pp * costh

    p1 = tf.stack([E1, px, py, pz])
    p2 = tf.stack([E2, -px, -py, -pz])

    # Now boost both p1 and p2 back to the lab frame
    # Construct the boosting matrix
    gamma = pin[0, :] / roots
    vx = -pin[1, :] / pin[0, :]
    vy = -pin[2, :] / pin[0, :]
    vz = -pin[3, :] / pin[0, :]
    v2 = vx * vx + vy * vy + vz * vz

    omgdv = (gamma - fone) / v2
    bmatE = tf.stack([gamma, -gamma * vx, -gamma * vy, -gamma * vz])
    bmatx = tf.stack(
        [-gamma * vx, omgdv * vx * vx + fone, omgdv * vx * vy, omgdv * vx * vz]
    )
    bmaty = tf.stack(
        [-gamma * vy, omgdv * vy * vx, omgdv * vy * vy + fone, omgdv * vy * vz]
    )
    bmatz = tf.stack(
        [-gamma * vz, omgdv * vz * vx, omgdv * vz * vy, omgdv * vz * vz + fone]
    )
    bmat = tf.stack([bmatE, bmatx, bmaty, bmatz])

    # Now unboost
    bmatt = tf.transpose(bmat)
    p1t = tf.transpose(p1)
    p2t = tf.transpose(p2)
    up1t = tf.keras.backend.batch_dot(p1t, bmatt)
    up2t = tf.keras.backend.batch_dot(p2t, bmatt)

    return tf.transpose(up1t), tf.transpose(up2t)


##### PS wrapper functions
@tf.function
def psgen_2to3(xarr):  # tree level phase space
    """ Generates a 2 -> H + 2j phase space

        a + b -> H + 1 + 2

        where 1 and 2 are massless and H is the Higgs boson.

        Uses 9 random numbers
    """
    x1, x2, shat, sh, fspartons, wgt = sample_linear_all(xarr[:, 0:8], nfspartons=2)
    s12, cos12, phi12 = fspartons[0]
    if MASSIVE_BOSON:
        pa, pb, ph, p12, jac = pcommon2to2(xarr[:, 8], shat, sh, s12)
        p1, p2 = pcommon1to2(s12, p12, fzero, fzero, cos12, phi12)
    else:
        pa, pb, p1, p2h, jac = pcommon2to2(xarr[:, 8], shat, 0.0, s12)
        p2, ph = pcommon1to2(s12, p2h, fzero, s12, cos12, phi12)
    wgt *= jac
    return pa, pb, p1, p2, x1, x2, wgt


def psgen_2to4(xarr):  # Real radiation phase space
    x1, x2, shat, sh, fspartons, wgt = sample_linear_all(xarr[:, 0:11], nfspartons=3)
    if massive_boson:
        s123, cos123, phi123 = fspartons[0]
        pa, pb, ph, p123, jac = pcommon2to2(xarr[:, 11], shat, sh, s123)
        s23, cos23, phi23 = fspartons[1]
        p1, p23 = pcommon1to2(s123, p123, fzero, s23, cos123, phi123)
        p2, p3 = pcommon1to2(s23, p23, fzero, fzero, cos23, phi23)
    else:
        raise NotImplementedError("Not implemented @ psgen_2to4")
    wgt *= jac

    # Apply the cuts
    # Check the values that actually pass the cuts
    min_pt2 = tf.square(min_pt)
    p1t2 = tf.greater_equal(calc_pt2(p1), min_pt2)
    p2t2 = tf.greater_equal(calc_pt2(p2), min_pt2)
    p3t2 = tf.greater_equal(calc_pt2(p3), min_pt2)
    if njets == 3:
        # ask for 3 jets with pt > ptmin
        ptpass = tf.logical_and(tf.logical_and(p1t2, p2t2), p3t2)

        stripe, idx = _condition_to_idx(tf.logical_and(p1t2, p2t2), p3t2)
    elif njets == 2:
        # ask for at least 2 jets with pt > ptmin
        stripe, idx = _condition_to_idx(
            tf.logical_or(p1t2, p2t2), tf.logical_or(p1t2, p3t2)
        )

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


if __name__ == "__main__":
    nevents = 10
    print("Generate a tree level phase space point")
    random_lo = np.random.rand(nevents, 9)
    momentum_set = psgen_2to3(random_lo)
