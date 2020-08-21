# This file is part of
from cpdfflow import ffi
from pdfflow import pflow


@ffi.def_extern()
def mkPDF(fname, dirname):
    """Generate a PDF givena PDF name and a directory."""
    pdfset = ffi.string(fname).decode('utf-8')
    path = ffi.string(dirname).decode('utf-8')
    pdf = pflow.mkPDF(pdfset, path)