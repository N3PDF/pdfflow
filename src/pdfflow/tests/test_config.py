"""
    Test that the configuration is consistent
"""

import os
import numpy as np
import importlib
import pdfflow.configflow
from pdfflow.configflow import DTYPE, DTYPEINT, int_me, float_me


def test_int_me():
    res = int_me(4)
    assert res.dtype == DTYPEINT


def test_float_me():
    res = float_me(4.0)
    assert res.dtype == DTYPE


def test_float_env():
    os.environ["PDFFLOW_FLOAT"] = "32"
    importlib.reload(pdfflow.configflow)
    from pdfflow.configflow import DTYPE

    assert DTYPE.as_numpy_dtype == np.float32
    os.environ["PDFFLOW_FLOAT"] = "64"
    importlib.reload(pdfflow.configflow)
    from pdfflow.configflow import DTYPE

    assert DTYPE.as_numpy_dtype == np.float64
    # Reset to default
    os.environ["PDFFLOW_FLOAT"] = "64"
    importlib.reload(pdfflow.configflow)


def test_int_env():
    os.environ["PDFFLOW_INT"] = "32"
    importlib.reload(pdfflow.configflow)
    from pdfflow.configflow import DTYPEINT

    assert DTYPEINT.as_numpy_dtype == np.int32
    os.environ["PDFFLOW_INT"] = "64"
    importlib.reload(pdfflow.configflow)
    from pdfflow.configflow import DTYPEINT

    assert DTYPEINT.as_numpy_dtype == np.int64
    # Reset to default
    os.environ["PDFFLOW_INT"] = "32"
    importlib.reload(pdfflow.configflow)
