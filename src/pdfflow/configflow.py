"""
Define some constants, header style
"""
# Most of this can be moved to a yaml file without loss of generality
import os

# Set TF to only log errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
# uncomment this line for debugging to avoid compiling any tf.function
# tf.config.experimental_run_functions_eagerly(True)

# Configure pdfflow logging
import logging

module_name = __name__.split(".")[0]
logger = logging.getLogger(module_name)
# Set level debug for development
logger.setLevel(logging.DEBUG)
# Create a handler and format it
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter("[%(levelname)s] (%(name)s) %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# logging.basicConfig(format='%(asctime)s %(message)s')

# Define the tensorflow number types
DTYPE = tf.float64
DTYPEINT = tf.int32


# The wrappers below transform tensors and array to the correct type
def int_me(i):
    return tf.cast(i, dtype=DTYPEINT)


def float_me(i):
    return tf.cast(i, dtype=DTYPE)


ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)
