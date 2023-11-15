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
def xfxq2(pid, n, x, m, q2, o):
    """Returns the xfxQ2 value for arrays of PID, x and q2 values."""
    global pdf
    pid_numpy = np.frombuffer(ffi.buffer(pid, n*ffi.sizeof('int')), dtype='int32')
    x_numpy = np.frombuffer(ffi.buffer(x, m*ffi.sizeof('double')), dtype='double')
    q2_numpy = np.frombuffer(ffi.buffer(q2, o*ffi.sizeof('double')), dtype='double')
    ret = pdf.xfxQ2(pid_numpy, x_numpy, q2_numpy).numpy()
    return ffi.cast("double*", ffi.from_buffer(ret))


@ffi.def_extern()
def alphasq2(q2, n):
    """Returns the alpha strong coupling at for an array of q2 values."""
    global pdf
    q2_numpy = np.frombuffer(ffi.buffer(q2, n*ffi.sizeof('double')), dtype='double')
    ret = pdf.alphasQ2(q2_numpy).numpy()
    return ffi.cast("double*", ffi.from_buffer(ret))
