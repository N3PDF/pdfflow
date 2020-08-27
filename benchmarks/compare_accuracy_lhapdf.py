"""
Benchmark script for LHAPDF comparison.
"""
from pdfflow.configflow import float_me
import lhapdf
import argparse
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import time
from math import floor, log10


parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str, help='The PDF set name/replica number.')
parser.add_argument("--pid", default=21, type=int, help='The flavour PID.')
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE, universal_newlines=True).stdout.strip('\n') + '/'
EPS = np.finfo(float).eps

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num == 0:
        return r'0'
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    if precision is None:
        precision = decimal_digits
    coeff = round(num / float(10**exponent), decimal_digits)
    return r"%.2f \times 10^{%d}"%(coeff, exponent)


def main(pdfname, pid):
    """Testing PDFflow vs LHAPDF performance."""
    mpl.rcParams['text.usetex'] = True
    import pdfflow.pflow as pdf

    p = pdf.mkPDF(pdfname, DIRNAME)
    l_pdf = lhapdf.mkPDF(pdfname)
    name = '\_'.join(pdfname.split('_'))

    s = time.time()
    p.trace()
    print("\nPDFflow\n\tBuilding graph time: %f\n"%(time.time()-s))

    plt.figure(tight_layout=True)
    ax = plt.subplot(1, 1, 1)
    x = np.logspace(-12,0,100000, dtype=float)
    q2 = np.array([0.1,1.65,1.7,4.92,1e2,1e3,1e4,1e5,1e6,2e6], dtype=float)**2
    for iq2 in q2:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for ix in x])
        vp = p.py_xfxQ2(pid, float_me(x), float_me([iq2]*len(x)))

        #print(r'%s'% sci_notation(iq2**0.5,3))

        ax.plot(x, np.abs(vp-vl)/(np.abs(vl)+EPS), label=r'$Q=%s$' % sci_notation(iq2**0.5,3))
    #exit()
    ax.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks = list(np.logspace(-12,0,13))
    labels = [r'$10^{-12}$']
    for i in range(-10,1,2):
        labels.extend(['',r'$10^{%d}$'%i])
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    locmin = mpl.ticker.LogLocator(base=10.0,subs=[i/10 for i in range(1,10)],numticks=13)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim([1e-12,1.])
    ax.set_ylim([1e-5, 10])
    ax.tick_params(axis='x', which='both', direction='in',top=True, labeltop=False)
    ax.tick_params(axis='y', which='both', direction='in',right=True, labelright=False)
    ax.title.set_text(r'%s, flav = %d' % (name, pid))
    ax.set_ylabel(r'$\displaystyle{\frac{|f_{p} - f_{l}|}{|f_{l}|+\epsilon}}$')
    ax.set_xlabel(r'$x$')
    ax.legend(frameon=False, ncol=2)
    plt.savefig('diff_%s_flav%d_fixedQ.png' % (pdfname.replace('/','-'), pid), bbox_inches='tight', dpi=200)
    plt.close()

    '''
    plt.subplot(2, 2, 3)
    for iq2 in q2:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for ix in x])
        vp = p.py_xfxQ2(pid, float_me(x), float_me([iq2]*len(x)))
        
        plt.plot(x, np.abs(vp-vl), label='$Q=%.2e$' % iq2**0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    #plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|$')
    plt.xlabel(r'\textbf{x}')
    plt.legend()
    '''

    x = np.array([1e-10,1e-9,1.1e-9,5e-7,1e-6,1e-4,1e-2,0.5,0.99], dtype=float)
    q2 = np.logspace(-3, 7, 100000, dtype=float)**2
    plt.figure(tight_layout=True)
    ax = plt.subplot(1, 1, 1)
    for ix in x:
        s_time = time.time()
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for iq2 in q2])
        l_time = time.time()
        vp = p.py_xfxQ2(pid, float_me([ix]*len(q2)), float_me(q2))
        p_time = time.time()
        

        ax.plot(q2**0.5, np.abs(vp-vl)/(np.abs(vl)+EPS), label=r'$x=%s$' % sci_notation(ix,3))
    ax.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-3,1e7])
    ax.set_ylim([1e-5, 10])
    ax.title.set_text(r'%s, flav = %d' % (name, pid))
    ax.set_ylabel(r'$\displaystyle{\frac{|f_{p} - f_{l}|}{|f_{l}|+\epsilon}}$')
    ax.set_xlabel(r'$Q$')

    ticks = list(np.logspace(-3,7,11))
    labels = [r'$10^{-3}$']
    for i in [i for i in range(-1,8,2)]:
        labels.extend(['',r'$10^{%d}$'%i])
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    locmin = mpl.ticker.LogLocator(base=10.0,subs=[i/10 for i in range(1,10)],numticks=11)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params(axis='x', which='both', direction='in',top=True, labeltop=False)
    ax.tick_params(axis='y', which='both', direction='in',right=True, labelright=False)


    ax.legend(frameon=False, ncol=2)

    '''
    plt.subplot(2, 2, 4)
    for ix in x:
        s_time = time.time()
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for iq2 in q2])
        l_time = time.time()
        vp = p.py_xfxQ2(pid, float_me([ix]*len(q2)), float_me(q2))
        p_time = time.time()
        
        plt.plot(q2**0.5, np.abs(vp-vl), label='$x=%.2e$' % ix)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    #plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|$')
    plt.xlabel(r'\textbf{Q}')
    plt.legend()
    '''

    plt.savefig('diff_%s_flav%d_fixedx.png' % (pdfname.replace('/','-'), pid), bbox_inches='tight', dpi=200)
    plt.close()

    print("\nDry run time comparison:")
    #print("\tlhapdf: %f"%(l_time - s_time))
    print("{:>10}:{:>15.8f}".format("lhapdf", l_time - s_time))
    print("{:>10}:{:>15.8f}".format("pdfflow", p_time - l_time))


if __name__ == "__main__":
    args = vars(parser.parse_args())
    if args['pid'] == 0:
        args['pid'] = 21
    start=time.time()
    main(**args)
    print("Total time: ", time.time()-start)
    
