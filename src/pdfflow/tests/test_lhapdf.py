"""
    Ensures pdfflow produces results which are compatible with lhpdf
    this code is made to run eagerly as @tf.function is tested by test_pflow
    running eagerly means less overhead on the CI which is running on CPU
"""
import pdfflow.pflow as pdf
import logging
from lhapdf_management import pdf_install
from lhapdf_management.configuration import environment

logger = logging.getLogger("pdfflow.test")

import os

# Run tests in CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

try:
    import lhapdf
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Tests against lhapdf need an installation of LHAPDF")


SIZE = 200

# Set up the PDF
LIST_PDF = [
    "PDF4LHC15_nnlo_100",
    "NNPDF31_nlo_as_0118",  # some problem for the first bin
    "MSTW2008lo68cl_nf3",
    "NNPDF30_nnlo_as_0121_nf_6",
    "cteq61",
]
MEMBERS = 2
FLAVS = list(range(-3, 4))
FLAVS[FLAVS.index(0)] = 21
DIRNAME = environment.datapath

# Utility to install lhapdf sets
def install_lhapdf(pdfset):
    try:
        lhapdf.mkPDF(pdfset)
    except RuntimeError:
        pdf_install(pdfset)


# Install the pdfs if they don't exist
for pdfset in LIST_PDF:
    install_lhapdf(pdfset)

# Set up the xarr
XARR = np.random.rand(SIZE)
# ensure there is at least a point with a very low x
XARR[0] = 1e-10

# Set up the Q2 arr
QS = [(1, 10), (100, 10000), (10, 100)]

# utilities
def gen_q2(qmin, qmax):
    """generate an array of q2 between qmin and qmax"""
    return np.random.rand(SIZE) * (qmax - qmin) + qmin


def dict_update(old_dict, new_dict):
    if not old_dict:
        for key, item in new_dict.items():
            old_dict[key] = [item]
    else:
        for key, item in new_dict.items():
            old_dict[key].append(item)


def get_pdfvals(xarr, qarr, pdfset):
    """Get the pdf values from LHAPDF"""
    lhapdf_pdf = lhapdf.mkPDF(pdfset)
    res = {}
    for x, q in zip(xarr, qarr):
        dict_update(res, lhapdf_pdf.xfxQ2(x, q))
    return res


def test_accuracy(atol=1e-6):
    """Check the accuracy for all PDF sets for all members
    in the lists LIST_PDF and MEMBERS
    for all defined ranges of Q for all flavours
    is better than atol.

    This test doesnt care about Q extrapolation
    """
    import tensorflow as tf

    tf.config.experimental_run_functions_eagerly(True)
    for setname in LIST_PDF:
        for member in range(MEMBERS):
            pdfset = f"{setname}/{member}"
            logger.info(" > Checking %s", pdfset)
            pdfflow = pdf.mkPDF(pdfset, f"{DIRNAME}/")
            for qi, qf in QS:
                # Dont test extrapolation
                qi = max(qi, pdfflow.q2min)
                qf = min(qf, pdfflow.q2max)
                q2arr = gen_q2(qi, qf)
                logger.info(" Q2 from %f to %f", qi, qf)
                flow_values = pdfflow.py_xfxQ2(FLAVS, XARR, q2arr)
                lhapdf_values = get_pdfvals(XARR, q2arr, pdfset)
            for i, f in enumerate(FLAVS):
                np.testing.assert_allclose(flow_values[:, i], lhapdf_values[f], atol=atol)
    tf.config.experimental_run_functions_eagerly(False)


if __name__ == "__main__":
    test_accuracy()
