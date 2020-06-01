"""
    Checks pdflow can run with no errors
"""
from pdfflow.pflow import mkPDF
import logging

logger = logging.getLogger("pdfflow.test")
import os
import subprocess as sp
from pdfflow.tests.test_lhapdf import install_lhapdf
# Run tests in CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PDFNAME = "NNPDF31_nlo_as_0118"
install_lhapdf(PDFNAME)

def test_pflow():
    """ Test several pdfflow features form instanciation to calling it
    with different PID formats"""
    pdf = mkPDF(f"{PDFNAME}/0")
    q2 = [16.7]
    x = [0.5]
    # Check I can get just one pid 
    for i in range(-1, 2):
        # as int
        pdf.xfxQ2(i, x, q2)
        # as list
        pdf.xfxQ2([i], x, q2)
    # Check I can ask for pid 21
    pdf.xfxQ2(21, x, q2)
    # or all pids
    pdf.xfxQ2_allpid(x, q2)

if __name__ == "__main__":
    test_pflow()
