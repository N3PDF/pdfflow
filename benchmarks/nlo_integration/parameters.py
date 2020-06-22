from pdfflow.configflow import float_me, fone, fzero, DTYPE
import tensorflow as tf

# Settings
TFLOAT1 = tf.TensorSpec(shape=[None], dtype=DTYPE)
TFLOAT4 = tf.TensorSpec(shape=[4, None], dtype=DTYPE)

# Physical parameters
higgs_mass = float_me(125.0)
mw = float_me(80.379)
gw = float_me(2.085)
stw = float_me(0.22264585341299603)
muR2 = float_me(pow(80, 2))

# Cuts
mjj_cut = float_me(600**2)
pt2_cut = float_me(30**2)


# Collision parameters
tech_cut = float_me(1e-6)
s_in = float_me(pow(8 * 1000, 2))
shat_min = tf.square(higgs_mass) * (1 + tf.sqrt(tech_cut))
# Flux factor
fbGeV2 = float_me(389379365600)
flux = fbGeV2 / 2.0 / s_in
