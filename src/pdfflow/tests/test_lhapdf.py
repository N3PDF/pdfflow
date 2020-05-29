"""
    Ensures vegasflow produces results which are compatible with lhpdf
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

SIZE = 10

# Set up the PDF
LIST_PDF = ["NNPDF31_nlo_as_0118", "PDF4LHC15_nnlo_100"]
MEMBERS = 2
FLAVS = list(range(-3,4))
FLAVS[FLAVS.index(0)] = 21
DIRNAME = sp.run(['lhapdf-config', '--datadir'], stdout=sp.PIPE).stdout.strip().decode()

# Install the pdfs
for pdfset in LIST_PDF:
    sp.run(['lhapdf', 'install', pdfset])

# Set up the xarr
XARR = np.random.rand(SIZE)

# Set up the Q2 arr
QS = [ (100, 10000), (3, 10), (10, 100)]

# utilities
def gen_q2(qmin, qmax):
    """ generate an array of q2 between qmin and qmax """
    return np.random.rand(SIZE)*(qmax-qmin) + qmin

def dict_update(old_dict, new_dict):
    if not old_dict:
        for key, item in new_dict.items():
            old_dict[key] = [item]
    else:
        for key, item in new_dict.items():
            old_dict[key].append(item)

def get_pdfvals(xarr, qarr, pdfset):
    """ Get the pdf values from LHAPDF """
    lhapdf_pdf = lhapdf.mkPDF(pdfset)
    res = {}
    for x, q in zip(xarr, qarr):
        dict_update(res, lhapdf_pdf.xfxQ2(x, q))
    return res

def test_accuracy(atol=1e-6):
    """ Check the accuracy for all PDF sets for all members
    in the lists LIST_PDF and MEMBERS
    for all defined ranges of Q for all flavours
    is better than atol
    """
    for setname in LIST_PDF:
        for member in range(MEMBERS):
            pdfset = f"{setname}/{member}"
            logger.info(" > Checking %s", pdfset)
            pdfflow = pdf.mkPDF(pdfset, f"{DIRNAME}/")
            for qi, qf in QS:
                q2arr = gen_q2(qi, qf)
                logger.info(" Q2 from %f to %f", qi, qf)
                flow_values = pdfflow.xfxQ2(FLAVS, XARR, q2arr)
                lhapdf_values = get_pdfvals(XARR, q2arr, pdfset)
            for i, f in enumerate(FLAVS):
                try:
                    np.testing.assert_allclose(flow_values[:,i], lhapdf_values[f], atol=atol)
                    logger.info(" checked flavour %d, passed", f)
                except:
                    print(q2arr)
                    print(flow_values[:,i] - lhapdf_values[f])
                    import ipdb
                    ipdb.set_trace()

if __name__ == "__main__":
    test_accuracy()
