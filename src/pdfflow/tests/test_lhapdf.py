"""
    Ensures vegasflow produces results which are compatible with lhpdf
"""
import lhapdf
import pdfflow.pflow as pdf
import numpy as np
import subprocess as sp

TESTPDF = "NNPDF31_nlo_as_0118/0"
FLAVS = [1,2,21]
DIRNAME = sp.run(['lhapdf-config', '--datadir'], stdout=sp.PIPE).stdout.strip().decode()

XARR = np.random.rand(10)
QMIN = 40 # For now avoid getting close to the danger zone
QARR = np.random.rand(10)*300 + QMIN

def dict_update(old_dict, new_dict):
    if not old_dict:
        for key, item in new_dict.items():
            old_dict[key] = [item]
    else:
        for key, item in new_dict.items():
            old_dict[key].append(item)

def get_pdfvals(xarr, qarr):
    lhapdf = lhapdf.mkPDF(TESTPDF)
    res = {}
    for x, q in zip(xarr, qarr):
        dict_update(res, lhapdf.xfxQ2(x, q))
    return res

def test_accuracy(atol=1e-6):
    pdfflow = pdf.mkPDF(TESTPDF, f"{DIRNAME}/")
    flow_values = pdfflow.xfxQ2(XARR, QARR)
    lhapdf_values = get_pdfvals(XARR, QARR)
    for f in FLAVS:
        np.testing.assert_allclose(flow_values[f], lhapdf_values[f], atol=atol)
