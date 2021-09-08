"""
    Checks pdflow can run with no errors

    This file also checks that functions can indeed compile

    Note that this test file does not install LHAPDF and (should)
    use a PDF set that is not installed by any other test.
    This ensures that pdfflow can indeed run independently of LHAPDF
"""
from pdfflow.pflow import mkPDF, mkPDFs
from pdfflow.configflow import run_eager, int_me, float_me
import logging

logger = logging.getLogger("pdfflow.test")
import os
import subprocess as sp
import numpy as np

# Run tests in CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

PDFNAME = "NNPDF31_nnlo_as_0118"


def pdfflow_tester(pdf, members=None):
    """Test several pdfflow features:
    - Check the single/many/all-pid signatures
    - Checks the python and TF signatures
    - Checks the output of the python and TF signatures are the same
    - Checks the expected shape of the signatures is correct
    """
    grid_size = 7
    x = np.random.rand(grid_size)
    q2 = 1000.0 * np.random.rand(grid_size)
    xtf = float_me(x)
    q2tf = float_me(q2)
    # Check I can get just one pid
    for i in range(-1, 2):
        # as int
        res_1 = pdf.py_xfxQ2(i, x, q2)
        # as list
        res_2 = pdf.py_xfxQ2([i], x, q2)
        np.testing.assert_allclose(res_1, res_2)
        # as tf objects
        tfpid = int_me([i])
        res_3 = pdf.xfxQ2(tfpid, xtf, q2tf)
        np.testing.assert_allclose(res_2, res_3)

        # Check shape
        if members is None:
            assert res_1.numpy().shape == (grid_size,)
        else:
            assert res_1.numpy().shape == (
                members,
                grid_size,
            )

    # Check I can get more than one pid
    nfl_size = 6
    fl_scheme = pdf.flavor_scheme.numpy()
    nfl_total = fl_scheme.size
    many_pid = np.random.choice(fl_scheme, nfl_size)

    res_1 = pdf.py_xfxQ2(many_pid, x, q2)
    res_2 = pdf.xfxQ2(int_me(many_pid), xtf, q2tf)
    np.testing.assert_allclose(res_1, res_2)
    # Check shape
    if members is None:
        assert res_1.numpy().shape == (grid_size, nfl_size)
    else:
        assert res_1.numpy().shape == (members, grid_size, nfl_size)

    # Check I can actually get all PID
    res_1 = pdf.py_xfxQ2_allpid(x, q2)
    res_2 = pdf.xfxQ2_allpid(xtf, q2tf)

    np.testing.assert_allclose(res_1, res_2)
    # Check shape
    if members is None:
        assert res_1.numpy().shape == (grid_size, nfl_total)
    else:
        assert res_1.numpy().shape == (members, grid_size, nfl_total)


def test_onemember():
    """Test the one-central-member of pdfflow"""
    # Check the central member
    pdf = mkPDF(f"{PDFNAME}/0")
    pdfflow_tester(pdf)
    # Try a non-central member, but trace first
    # Ensure it is not running eagerly
    run_eager(False)
    pdf = mkPDF(f"{PDFNAME}/1")
    pdf.trace()
    pdfflow_tester(pdf)


def test_multimember():
    """Test the multi-member capabilities of pdfflow"""
    run_eager(False)
    members = 5
    pdf = mkPDFs(PDFNAME, range(members))
    assert pdf.nmembers == members
    pdf.trace()
    assert len(pdf.active_members) == members
    pdfflow_tester(pdf, members=members)


def test_one_multi():
    """Test that the multimember-I is indeed the same as just the Ith instance"""
    run_eager(True)
    pdf = mkPDF(f"{PDFNAME}/0")
    multi_pdf = mkPDFs(PDFNAME, [4, 0, 6])
    grid_size = 4
    x = np.random.rand(grid_size)
    q2 = np.random.rand(grid_size) * 1000.0
    res_1 = pdf.py_xfxQ2_allpid(x, q2)
    res_2 = multi_pdf.py_xfxQ2_allpid(x, q2)
    np.testing.assert_allclose(res_1, res_2[1])


if __name__ == "__main__":
    test_one_multi()
