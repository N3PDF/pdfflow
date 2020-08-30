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
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0",
                    type=str, help='The PDF set name/replica number.')
parser.add_argument("--pid", default=21, type=int, help='The flavour PID.')
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE,
                 universal_newlines=True).stdout.strip('\n') + '/'
EPS = np.finfo(float).eps

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX, with specified number of significant
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
    return r'%s'%format(coeff, f'.{decimal_digits}f') + r"\times 10^{%d}"%exponent

def set_ticks(ax, start, end, numticks, axis, nskip=2):
    """
    Set both major and minor axes ticks in the logarithmical scale
    Parameters:
        ax: matplotlib.axes.Axes object
        start: int, leftmost tick
        end: int, rightmost tick
        numticks
        axis: 1 y axis, 0 x axis
        nskip: int, major ticks to leave without label
    """

    ticks = list(np.logspace(start,end,end-start+1))
    labels = [r'$10^{%d}$'%start]
    for i in [i for i in range(start+2,end+1,nskip)]:
        labels.extend(['' for i in range(nskip-1)]+[r'$10^{%d}$'%i])
    locmin = mpl.ticker.LogLocator(base=10.0,
                                   subs=[i/10 for i in range(1,10)],
                                   numticks=numticks)
    if axis == 'x':
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if axis == 'y':
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    return ax    
    

    

def main(pdfname, pid):
    """Testing PDFflow vs LHAPDF performance."""
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [11,5.5]
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17
    mpl.rcParams['legend.fontsize'] = 14



    import pdfflow.pflow as pdf

    p = pdf.mkPDF(pdfname, DIRNAME)
    l_pdf = lhapdf.mkPDF(pdfname)
    name = '\_'.join(pdfname.split('_'))

    s = time.time()
    p.trace()
    print("\nPDFflow\n\tBuilding graph time: %f\n"%(time.time()-s))

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.05)
    ax = fig.add_subplot(gs[0])
    x = np.logspace(-12,0,10000, dtype=float)
    q2 = np.array([1.65,1.7,4.92,1e2,1e3,1e4,1e5,1e6,2e6], dtype=float)**2
    for iq2 in q2:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for ix in x])
        vp = p.py_xfxQ2(pid, float_me(x), float_me([iq2]*len(x)))


        ax.plot(x, np.abs(vp-vl)/(np.abs(vl)+EPS),
                label=r'$Q=%s$' % sci_notation(iq2**0.5,2))

    ax.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-12,1.])
    ax.set_ylim([EPS, .01])
    
    ax = set_ticks(ax, -12, 0, 13, 'x', 4)
    ax.tick_params(axis='x', which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False)

    ax = set_ticks(ax, -15, -3, 16, 'y')
    ax.tick_params(axis='y', which='both', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)

    ax.set_title(r'%s, flav = %d' % (name, pid))
    ax.set_ylabel(r'$\displaystyle{r_{i}(x,Q)}$',
                  fontsize=20)
    ax.set_xlabel(r'$x$', fontsize=17)
    ax.legend(frameon=False, ncol=2,
              loc='upper right', bbox_to_anchor=(1.02,0.9))
    #plt.savefig('diff_%s_flav%d_fixedQ.pdf' % (pdfname.replace('/','-'), pid),
    #            bbox_inches='tight', dpi=200)
    #plt.close()

    x = np.array([1e-10,1e-9,1.1e-9,5e-7,1e-6,1e-4,1e-2,0.5,0.99], dtype=float)
    q2 = np.logspace(1, 7, 10000, dtype=float)**2
    #fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(gs[1])
    for ix in x:
        s_time = time.time()
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for iq2 in q2])
        l_time = time.time()
        vp = p.py_xfxQ2(pid, float_me([ix]*len(q2)), float_me(q2))
        p_time = time.time()

        ax.plot(q2**0.5, np.abs(vp-vl)/(np.abs(vl)+EPS),
                label=r'$x=%s$' % sci_notation(ix,1))

    ax.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1,1e7])
    ax.set_ylim([EPS, .01])
    
    ax = set_ticks(ax, 1, 7, 9, 'x')
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)

    ax = set_ticks(ax, -15, -3, 16, 'y')
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=False)

    ax.set_title(r'%s, flav = %d' % (name, pid))
    #ax.set_ylabel(r'$\displaystyle{\frac{|f_{p} - f_{l}|}{|f_{l}|+\epsilon}}$',
    #              fontsize=21)
    ax.set_xlabel(r'$Q$', fontsize=17)
    ax.legend(frameon=False, ncol=2,
              loc='upper right', bbox_to_anchor=(1.02,0.9))
    plt.savefig('diff_%s_flav%d.pdf' % (pdfname.replace('/','-'), pid),
                bbox_inches='tight', dpi=250)
    plt.close()

    print("\nDry run time comparison:")
    print("{:>10}:{:>15.8f}".format("lhapdf", l_time - s_time))
    print("{:>10}:{:>15.8f}".format("pdfflow", p_time - l_time))


if __name__ == "__main__":
    args = vars(parser.parse_args())
    if args['pid'] == 0:
        args['pid'] = 21
    start=time.time()
    main(**args)
    print("Total time: ", time.time()-start)
    
