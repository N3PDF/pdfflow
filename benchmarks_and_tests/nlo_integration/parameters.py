from pdfflow.configflow import float_me, DTYPE
import tensorflow as tf

# Settings
TFLOAT1 = tf.TensorSpec(shape=[None], dtype=DTYPE)
TFLOAT4 = tf.TensorSpec(shape=[4, None], dtype=DTYPE)
TECH_CUT = 1e-8 # change this for debugging or to avoid going to too divergent zones

# Physical parameters
higgs_mass = float_me(125.0)
mw = float_me(80.379)
gw = float_me(2.085)
stw = float_me(0.22264585341299603)
muR2 = float_me(pow(higgs_mass, 2))

# Cuts
pt2_cut = float_me(30 ** 2)
rdistance = float_me(0.3 ** 2)
deltaycut = float_me(4.5)
m2jj_cut = float_me(600 ** 2)

# Collision parameters
s_in = float_me(pow(13 * 1000, 2))
# Flux factor
fbGeV2 = float_me(389379365600)
flux = fbGeV2 / 2.0 / s_in

# Compute shat_min taking into account the higgs mass and the cuts
# only pt cuts, only two jets are required to have pt > pt_cut
shat_min = (
    tf.square(higgs_mass) + 2.0 * pt2_cut + 4.0 * higgs_mass * tf.sqrt(pt2_cut) + 4.0 * TECH_CUT
)

### Debug parameters
# Set the subtraction term on or off, useful to check that
# indeed the R ME does not converge
SUBTRACT = False

# Uncomment when testing R ME at tree level for a more faithful smin
# shat_min = tf.square(higgs_mass) + 6.0*pt2_cut + 6.0*higgs_mass*tf.sqrt(pt2_cut)
