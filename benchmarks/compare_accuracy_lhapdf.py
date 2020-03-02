"""
Benchmark script for LHAPDF comparison.
"""
import lhapdf
import argparse
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str, help='The PDF set name/replica number.')
parser.add_argument("--pid", default=21, type=int, help='The flavour PID.')
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE, universal_newlines=True).stdout.strip('\n') + '/'
EPS = np.finfo(float).eps


def main(pdfname, pid):
    """Testing PDFflow vs LHAPDF performance."""
    import pdfflow.pflow as pdf

    p = pdf.mkPDF(pdfname, DIRNAME)
    l_pdf = lhapdf.mkPDF(pdfname)

    plt.figure(figsize=(16.0, 12.0))
    plt.subplot(2, 2, 1)
    x = np.logspace(-9, 0, 1000, dtype=float)
    q2 = np.array([1.65, 3, 5, 10, 50, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3], dtype=float)**2
    for iq2 in q2:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for ix in x])
        _,_,vp = p.xfxQ2(x, [iq2]*len(x), pid)
        plt.plot(x, np.abs(vp-vl)/(np.abs(vl)+EPS), label='$Q=%.2e$' % iq2**0.5)
    plt.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], colors='orange', linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|/(|f_{lhapdf}|+eps$')
    plt.xlabel('x')
    plt.legend()

    plt.subplot(2, 2, 3)
    for iq2 in q2:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for ix in x])
        _,_,vp = p.xfxQ2(x, [iq2]*len(x), pid)
        plt.plot(x, np.abs(vp-vl), label='$Q=%.2e$' % iq2**0.5)
    plt.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], colors='orange', linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|$')
    plt.xlabel('x')
    plt.legend()

    x = np.array([1e-8, 1e-6, 1e-4, 0.01, 0.1, 0.2, 0.5, 0.8], dtype=float)
    q2 = np.logspace(0.5, 4.9, 1000, dtype=float)**2
    plt.subplot(2, 2, 2)
    for ix in x:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for iq2 in q2])
        _,_,vp = p.xfxQ2([ix]*len(q2), q2, pid)
        plt.plot(q2**0.5, np.abs(vp-vl)/(np.abs(vl)+EPS), label='$x=%.2e$' % ix)
    plt.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], colors='orange', linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|/(|f_{lhapdf}|+eps$')
    plt.xlabel('Q')
    plt.legend()

    plt.subplot(2, 2, 4)
    for ix in x:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for iq2 in q2])
        _,_,vp = p.xfxQ2([ix]*len(q2), q2, pid)
        plt.plot(q2**0.5, np.abs(vp-vl), label='$x=%.2e$' % ix)
    plt.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], colors='orange', linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|$')
    plt.xlabel('Q')
    plt.legend()

    plt.savefig('diff_%s_flav%d.png' % (pdfname.replace('/','-'), pid), bbox_inches='tight')


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)