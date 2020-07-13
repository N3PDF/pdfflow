"""
    Ensures pdfflow produces results which are compatible with lhpdf
    this code is made to run eagerly as @tf.function is tested by test_pflow
    running eagerly means less overhead on the CI which is running on CPU
"""
import pdfflow.pflow as pdf
import logging

logger = logging.getLogger("pdfflow.test")

import os

# Run tests in CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import lhapdf
import numpy as np
import subprocess as sp

# Utility to install lhapdf sets
def install_lhapdf(pdfset):
    try:
        lhapdf.mkPDF(pdfset)
    except RuntimeError:
        sp.run(["lhapdf", "install", pdfset])


SIZE = 200

# Set up the PDF
LIST_PDF = ["NNPDF31_nnlo_as_0118", "cteq6"]
MEMBERS = 2
DIRNAME = sp.run(["lhapdf-config", "--datadir"], stdout=sp.PIPE).stdout.strip().decode()

# Install the pdfs if they don't exist
for pdfset in LIST_PDF:
    install_lhapdf(pdfset)

# Set up the Q2 arr
QS = [(1, 10), (100, 10000), (10, 100)]

# utilities
def gen_q2(qmin, qmax):
    """ generate an array of q2 between qmin and qmax """
    return np.random.rand(SIZE) * (qmax - qmin) + qmin


def get_alphavals(q2arr, pdfset, sq2 = False):
    """ Generate an array of alphas(q) values from LHAPDF """
    lhapdf_pdf = lhapdf.mkPDF(pdfset)
    if sq2:
        return np.array([lhapdf_pdf.alphasQ2(iq) for iq in q2arr])
    else:
        return np.array([lhapdf_pdf.alphasQ(iq) for iq in q2arr])


def test_accuracy_alphas(atol=1e-6):
    """ Check the accuracy for all PDF sets for all members given
    when computing alpha_s given Q is compatible within atol
    between pdfflow and LHAPDF.
    This test run eagerly
    """
    import tensorflow as tf

    tf.config.experimental_run_functions_eagerly(True)
    for setname in LIST_PDF:
        for member in range(MEMBERS):
            pdfset = f"{setname}/{member}"
            logger.info(" > Checking %s", pdfset)
            pdfflow = pdf.mkPDF(pdfset, f"{DIRNAME}/", alpha_computation=True)
            for qi, qf in QS:
                qi = max(qi, pdfflow.q2min)
                qf = min(qf, pdfflow.q2max)
                q2arr = gen_q2(qi, qf)
                logger.info(" Q2 from %f to %f", qi, qf)
                flow_values = pdfflow.py_alphasQ(q2arr)
                lhapdf_values = get_alphavals(q2arr, pdfset, sq2 = False)
                np.testing.assert_allclose(flow_values, lhapdf_values, atol=atol)
    tf.config.experimental_run_functions_eagerly(False)

def test_alphas_q2(atol=1e-6):
    """ Check the accuracy for all PDF sets for all members given
    when computing alpha_s given Q is compatible within atol
    between pdfflow and LHAPDF
    This test does not run eagerly
    """
    for setname in LIST_PDF:
        for member in range(MEMBERS):
            pdfset = f"{setname}/{member}"
            logger.info(" > Checking %s", pdfset)
            pdfflow = pdf.mkPDF(pdfset, f"{DIRNAME}/", alpha_computation=True)
            for qi, qf in QS:
                qi = max(qi, pdfflow.q2min)
                qf = min(qf, pdfflow.q2max)
                q2arr = gen_q2(qi, qf)
                logger.info(" Q2 from %f to %f", qi, qf)
                flow_values = pdfflow.py_alphasQ2(q2arr)
                lhapdf_values = get_alphavals(q2arr, pdfset, sq2 = True)
                np.testing.assert_allclose(flow_values, lhapdf_values, atol=atol)



if __name__ == "__main__":
    test_accuracy_alphas()
    test_alphas_q2()
