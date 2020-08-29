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
import pdfflow.pflow as pdf
from compare_accuracy_lhapdf import set_ticks

parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str, help='The PDF set name/replica number.')
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE, universal_newlines=True).stdout.strip('\n') + '/'
EPS = np.finfo(float).eps

def compare_alphas(pdfname, ax, p=None):
    """
    Computes the alphas difference pdfflow vs lhapdf and returns
    axes for plots
    Parameters:
        pdfname: string
        ax: matplotlib.axes.Axes object
        p: pdf object
    Return:
        matplotlib.axes.Axes object
    """
    if p is None:
        p = pdf.mkPDF(pdfname, DIRNAME)
    l_pdf = lhapdf.mkPDF(pdfname)
    name = '\_'.join(pdfname.split('_'))

    q = np.logspace(-3, 9, 10000, dtype=float)

    s_time = time.time()
    vl = np.array([l_pdf.alphasQ(iq) for iq in q])
    l_time = time.time()
    vp = p.py_alphasQ(q)
    p_time = time.time()
        
    ax.plot(q, np.abs(vp-vl)/(np.abs(vl)+EPS))
    ax.hlines(1e-3, plt.xlim()[0], plt.xlim()[1],
              linestyles='dotted', color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-3,1e9])
    ax.set_ylim([EPS, 0.01])

    ax = set_ticks(ax, -3, 9, 13, 'x', 4)
    ax.tick_params(axis='x', which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False)

    ax = set_ticks(ax, -15, -3, 16, 'y')

    ax.title.set_text(r'%s, $\alpha_s(Q)$' % name)
    ax.set_xlabel(r'$Q$', fontsize=17)

    print("\nDry run time comparison for pdf %s:"%pdfname)
    print("{:>10}:{:>15.8f}".format("lhapdf", l_time - s_time))
    print("{:>10}:{:>15.8f}".format("pdfflow", p_time - l_time))

    return ax


def main(pdfname):
    """Testing PDFflow vs LHAPDF performance."""
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [11,5.5]
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17

    p = pdf.mkPDF(pdfname, DIRNAME)
    name = '\_'.join(pdfname.split('_'))

    s = time.time()
    p.alphas_trace()
    print("\nPDFflow alphas\n\tBuilding graph time: %f\n"%(time.time()-s))

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.1)

    ax = fig.add_subplot(gs[0])
    ax = compare_alphas(pdfname, ax, p)
    ax.tick_params(axis='y', which='both', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)
    ax.set_ylabel(r'$\displaystyle{\frac{|\alpha_{s,p} - \alpha_{s,l}|}{|\alpha_{s,l}|+\epsilon}}$',
                  fontsize=22)

    ax = fig.add_subplot(gs[1])
    pdfname = 'MMHT2014nlo68cl/0'
    ax = compare_alphas(pdfname, ax)
    ax.tick_params(axis='y', which='both', direction='in',
                   left=True, labelleft=False,
                   right=True, labelright=False)
    
    plt.savefig('diff_alphas.pdf', bbox_inches='tight',dpi=250)
    plt.close()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start=time.time()
    main(**args)
    print("Total time: ", time.time()-start)
    
