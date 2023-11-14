# This file is part of
from cpdfflow import ffi
from pdfflow import pflow
import numpy as np

pdf = None

@ffi.def_extern()
def mkpdf(fname, dirname):
    """Generate a PDF givena PDF name and a directory."""
    pdfset = ffi.string(fname).decode('utf-8')
    path = ffi.string(dirname).decode('utf-8')
    global pdf
    pdf = pflow.mkPDF(pdfset, path)


@ffi.def_extern()
def xfxq2(pid, x, q2):
    """Returns the xfxQ2 value for a given PID at specific x and q2."""
    global pdf
    return pdf.xfxQ2([pid], [x], [q2])


@ffi.def_extern()
def alphasq2(q2, n):
    """Returns the alpha strong coupling at specific q2 value."""
    global pdf
    q2_numpy = np.frombuffer(ffi.buffer(q2, n*ffi.sizeof('double')), dtype='double')
    ret = pdf.alphasQ2(q2_numpy).numpy()
    return ffi.cast("double*", ffi.from_buffer(ret))
