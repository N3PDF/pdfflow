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

parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str, help='The PDF set name/replica number.')
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE, universal_newlines=True).stdout.strip('\n') + '/'
EPS = np.finfo(float).eps

def main(pdfname):
    """Testing PDFflow vs LHAPDF performance."""
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [5.5,5.5]
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17

    import pdfflow.pflow as pdf
    from compare_accuracy_lhapdf import set_ticks

    p = pdf.mkPDF(pdfname, DIRNAME)
    l_pdf = lhapdf.mkPDF(pdfname)
    name = '\_'.join(pdfname.split('_'))

    s = time.time()
    p.alphas_trace()
    print("\nPDFflow alphas\n\tBuilding graph time: %f\n"%(time.time()-s))

    plt.figure(tight_layout=True)
    
    q = np.logspace(-3, 9, 10000, dtype=float)
    ax = plt.subplot(111)

    s_time = time.time()
    vl = np.array([l_pdf.alphasQ(iq) for iq in q])
    l_time = time.time()
    vp = p.py_alphasQ(q)
    p_time = time.time()

        
    ax.plot(q, np.abs(vp-vl)/(np.abs(vl)+EPS))
    ax.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-3,1e9])
    ax.set_ylim([EPS, 0.01])

    ax = set_ticks(ax, -3, 9, 13, 'x', 4)
    ax.tick_params(axis='x', which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False)

    ax = set_ticks(ax, -15, -3, 16, 'y')
    ax.tick_params(axis='y', which='both', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)

    ax.title.set_text(r'%s, $\alpha_s(Q)$' % name)
    ax.set_ylabel(r'$\displaystyle{\frac{|\alpha_{s,p} - \alpha_{s,l}|}{|\alpha_{s,l}|+\epsilon}}$', fontsize=21)
    ax.set_xlabel(r'$Q$', fontsize=16)
    plt.savefig('diff_%s_alphas.pdf' % (pdfname.replace('/','-')), bbox_inches='tight',dpi=250)

    print("\nDry run time comparison:")
    print("{:>10}:{:>15.8f}".format("lhapdf", l_time - s_time))
    print("{:>10}:{:>15.8f}".format("pdfflow", p_time - l_time))


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start=time.time()
    main(**args)
    print("Total time: ", time.time()-start)
    
