"""
    Checks pdflow can run with no errors

    This file also checks that functions can indeed compile
"""
from pdfflow.pflow import mkPDF, mkPDFs
from pdfflow.configflow import run_eager
import logging

logger = logging.getLogger("pdfflow.test")
import os
import subprocess as sp
from pdfflow.tests.test_lhapdf import install_lhapdf

# Run tests in CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

PDFNAME = "NNPDF31_nlo_as_0118"
install_lhapdf(PDFNAME)


def pdfflow_tester(pdf):
    """ Test several pdfflow features:
        - Instantiationg
        - Calling it with different PID
    """
    q2 = [16.7]
    x = [0.5]
    # Check I can get just one pid
    for i in range(-1, 2):
        # as int
        pdf.py_xfxQ2(i, x, q2)
        # as list
        pdf.py_xfxQ2([i], x, q2)
        # check the tf signature
        tfpid = tf.constant([i])
        pdf.xfxQ2(tfpid, x, q2)
    # Check I can get more than one pid
    pdf.py_xfxQ2([1, 4, -2, -1, 4], x, q2)
    tfpid = tf.constant([1, 0, 21])
    pdf.xfxQ2(tfpid, x, q2)
    # Check I can ask for pid 21
    pdf.py_xfxQ2(21, x, q2)
    # or all pids
    pdf.py_xfxQ2_allpid(x, q2)
    pdf.xfxQ2_allpid(x, q2)

def test_onemember():
    """ Test the one-central-member of pdfflow """
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
    """ Test the multi-member capabilities of pdfflow """
    run_eager(False)
    pdf = mkPDFs(PDFNAME, [0, 2, 4, 7])
    # Check the tracing with the central-member grid
    pdfflow_tester(pdf)
    pdf = mkPDFs(PDFNAME, [3, 9])
    # Check the tracing with all-members
    pdfflow_tester(pdf, all_members=True)

if __name__ == "__main__":
    test_onemember()
