"""
Define some constants, header style
"""
# Most of this can be moved to a yaml file without loss of generality
import os
import logging
import numpy as np

# Set TF to only log errors
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf

# uncomment this line for debugging to avoid compiling any tf.function
# tf.config.run_functions_eagerly(True)

def run_eager(flag=True):
    """ Wrapper around `run_functions_eagerly` """
    if tf.__version__ < '2.3.0':
        tf.config.experimental_run_functions_eagerly(flag)
    else:
        tf.config.run_functions_eagerly(flag)

# Configure pdfflow logging
module_name = __name__.split(".")[0]
logger = logging.getLogger(module_name)

# Read the log level from environment, 3 == debug, 2 (default) == info, 1 == warning, 0 == error
DEFAULT_LOG_LEVEL = "2"
log_level_idx = os.environ.get("PDFFLOW_LOG_LEVEL", DEFAULT_LOG_LEVEL)
log_dict = {"0": logging.ERROR, "1": logging.WARNING, "2": logging.INFO, "3": logging.DEBUG}
bad_log_warning = None
if log_level_idx not in log_dict:
    bad_log_warning = log_level_idx
    log_level_idx = DEFAULT_LOG_LEVEL
log_level = log_dict[log_level_idx]

# Set level debug for development
logger.setLevel(log_level)
# Create a handler and format it
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_format = logging.Formatter("[%(levelname)s] (%(name)s) %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# Now that the logging has been created, warn about the bad logging level
if bad_log_warning is not None:
    logger.warning(
        "Accepted log levels are: %s, received: %s", list(log_dict.keys()), bad_log_warning
    )
    logger.warning(f"Setting log level to its default value: {DEFAULT_LOG_LEVEL}")

# Define the tensorflow number types
_float_env = os.environ.get("PDFFLOW_FLOAT", "64")
_int_env = os.environ.get("PDFFLOW_INT", "32")

if _float_env == "64":
    DTYPE = tf.float64
elif _float_env == "32":
    DTYPE = tf.float32
else:
    logger.warning(f"PDFFLOW_FLOAT={_float_env} not understood, defaulting to 64 bits")

if _int_env == "64":
    DTYPEINT = tf.int64
elif _int_env == "32":
    DTYPEINT = tf.int32
else:
    logger.warning(f"PDFFLOW_INT={_int_env} not understood, defaulting to 64 bits")

FMAX = tf.constant(np.finfo(np.float64).max, dtype=DTYPE)

# The wrappers below transform tensors and array to the correct type
def int_me(i):
    """ Cast the input to the `DTYPEINT` type """
    return tf.cast(i, dtype=DTYPEINT)


def float_me(i):
    """ Cast the input to the `DTYPE` type """
    return tf.cast(i, dtype=DTYPE)


ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)
