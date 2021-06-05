"""
    Encapsulate the definition of constant, logging behaviour and environment variables in one single module
"""
import os
import sys
import pathlib
import logging
import subprocess as sp
import numpy as np

# Log levels
LOG_DICT = {"0": logging.ERROR, "1": logging.WARNING, "2": logging.INFO, "3": logging.DEBUG}

# Read the PDFFLOW environment variables
_log_level_idx = os.environ.get("PDFFLOW_LOG_LEVEL")
_data_path = os.environ.get("PDFFLOW_DATA_PATH")
_lhapdf_data_path = os.environ.get("LHAPDF_DATA_PATH")
_float_env = os.environ.get("PDFFLOW_FLOAT", "64")
_int_env = os.environ.get("PDFFLOW_INT", "32")


# Logging
_bad_log_warning = None
if _log_level_idx not in LOG_DICT:
    _bad_log_warning = _log_level_idx
    _log_level_idx = None

if _log_level_idx is None:
    # If no log level is provided, set some defaults
    _log_level = LOG_DICT["2"]
    _tf_log_level = LOG_DICT["0"]
else:
    _log_level = _tf_log_level = LOG_DICT[_log_level_idx]

# Configure pdfflow logging
logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(_log_level)

# Create and format the log handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(_log_level)
_console_format = logging.Formatter("[%(levelname)s] (%(name)s) %(message)s")
_console_handler.setFormatter(_console_format)
logger.addHandler(_console_handler)

# pdfflow options set, now import tensorfow to prepare convenience wrappers
# and set any options that we need
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf

tf.get_logger().setLevel(_tf_log_level)


def run_eager(flag=True):
    """Wrapper around `run_functions_eagerly`
    When used no function is compiled
    """
    if tf.__version__ < "2.3.0":
        tf.config.experimental_run_functions_eagerly(flag)
    else:
        tf.config.run_functions_eagerly(flag)


# set the precision type
if _float_env == "64":
    DTYPE = tf.float64
    FMAX = tf.constant(np.finfo(np.float64).max, dtype=DTYPE)
elif _float_env == "32":
    DTYPE = tf.float32
    FMAX = tf.constant(np.finfo(np.float32).max, dtype=DTYPE)
else:
    DTYPE = tf.float64
    FMAX = tf.constant(np.finfo(np.float64).max, dtype=DTYPE)
    logger.warning(f"PDFFLOW_FLOAT={_float_env} not understood, defaulting to 64 bits")

if _int_env == "64":
    DTYPEINT = tf.int64
elif _int_env == "32":
    DTYPEINT = tf.int32
else:
    DTYPEINT = tf.int64
    logger.warning(f"PDFFLOW_INT={_int_env} not understood, defaulting to 64 bits")

# The wrappers below transform tensors and array to the correct type
def int_me(i):
    """Cast the input to the `DTYPEINT` type"""
    return tf.cast(i, dtype=DTYPEINT)


def float_me(i):
    """Cast the input to the `DTYPE` type"""
    return tf.cast(i, dtype=DTYPE)


ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)

# PDF paths
def find_pdf_path(pdfname):
    """Look in all possible directories for the given `pdfname`
    if the pdfname is not found anywhere, fail with error
    """
    all_paths = []
    if _data_path:
        all_paths.append(_data_path)
    if _lhapdf_data_path:
        all_paths.append(_lhapdf_data_path)
    try:
        import lhapdf

        lhapdf_cmd = ["lhapdf-config", "--datadir"]
        # Check the python version in order to use the right subprocess call
        if sys.version_info.major == 3 and sys.version_info.minor < 7:
            dirname_raw = sp.run(lhapdf_cmd, check=True, stdout=sp.PIPE)
            dirname = dirname_raw.stdout.decode().strip()
        else:
            dirname_raw = sp.run(lhapdf_cmd, capture_output=True, text=True, check=True)
            dirname = dirname_raw.stdout.strip()
        all_paths.append(dirname)
    except ModuleNotFoundError:
        # If lhapdf is not installed, make a note and continue
        lhapdf = None

    # Return whatever path has the pdf inside
    for path in all_paths:
        if pathlib.Path(f"{path}/{pdfname}").exists():
            return path

    # If none of them do, fail but inform the user
    error_msg = f"The PDF set {pdfname} could not be found"
    if lhapdf is not None:
        error_msg += f"\nIt can be installed with ~$ lhapdf install {pdfname}"
    elif _data_path is not None:
        error_msg += f"\nPlease, download it and uncompress it in {_data_path}"
    elif _lhapdf_data_path is not None:
        error_msg += f"\nPlease, download it and uncompress it in {_lhapdf_data_path}"
    else:
        error_msg += f"""
Please, either download the set to an appropiate folder and make the environment variable
PDFFLOW_DATA_PATH point to it or install the lhapdf python wrapper"""
    raise ValueError(error_msg)
