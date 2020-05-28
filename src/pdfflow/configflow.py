"""
Define some constants, header style
"""
# Most of this can be moved to a yaml file without loss of generality
import tensorflow as tf

# Define the tensorflow number types
DTYPE = tf.float64
DTYPEINT = tf.int32

# Create wrappers in order to have numbers of the correct type
def int_me(i):
    return tf.constant(i, dtype=DTYPEINT)


def float_me(i):
    return tf.constant(i, dtype=DTYPE)

ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)
