# This file is part of
from cpdfflow import ffi
from pdfflow import pflow

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
def alphasq2(q2):
    """Returns the alpha strong coupling at specific q2 value."""
    global pdf
    return pdf.alphasQ2([q2])
