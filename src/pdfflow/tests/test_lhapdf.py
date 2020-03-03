"""
    Ensures vegasflow produces results which are compatible with lhpdf
"""
import pdfflow.pflow as pdf
import numpy as np
import subprocess as sp

TESTPDF = "NNPDF31_nlo_as_0118/0"
FLAVS = [1,2,21]
DIRNAME = sp.run(['lhapdf-config', '--datadir'], stdout=sp.PIPE).stdout.strip()

XARR = np.random.rand(10)
QARR = np.random.rand(10)*100

def get_pdfvals(xarr, qarr):
    import lhapdf
    lhapdf = lhapdf.mkPDF(TESTPDF)
    

def test_accuracy():
    pdfflow = pdf.mkPDF(TESTPDF), f"{DIRNAME}/")

    for f in FLAVS:
        flow_values = pdfflow.xfxQ2(XARR, QARR, f)
