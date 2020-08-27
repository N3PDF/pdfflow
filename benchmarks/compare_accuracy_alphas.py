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
    import pdfflow.pflow as pdf

    p = pdf.mkPDF(pdfname, DIRNAME)
    l_pdf = lhapdf.mkPDF(pdfname)
    name = '\_'.join(pdfname.split('_'))

    s = time.time()
    p.alphas_trace()
    print("\nPDFflow alphas\n\tBuilding graph time: %f\n"%(time.time()-s))

    plt.figure(tight_layout=True)
    
    q = np.logspace(-3, 9, 1000000, dtype=float)
    ax = plt.subplot(111)

    s_time = time.time()
    vl = np.array([l_pdf.alphasQ(iq) for iq in q])
    l_time = time.time()
    vp = p.py_alphasQ(q)
    p_time = time.time()

        
    ax.plot(q, np.abs(vp-vl)/(np.abs(vl)+EPS))
    ax.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-3,1e9])
    ax.set_ylim([1e-5, 10])
    ax.title.set_text(r'%s, $\alpha_s(Q)$' % name)
    ax.set_ylabel(r'$\displaystyle{\frac{|\alpha_{s,p} - \alpha_{s,l}|}{|\alpha_{s,l}|+\epsilon}}$')
    ax.set_xlabel(r'$Q$')

    ticks = list(np.logspace(-3,9,13))
    labels = [r'$10^{-3}$']
    for i in range(-1,10,2):
        labels.extend(['',r'$10^{%d}$'%i])
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    locmin = mpl.ticker.LogLocator(base=10.0,subs=[i/10 for i in range(1,10)],numticks=13)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params(axis='x', which='both', direction='in',top=True, labeltop=False)
    ax.tick_params(axis='y', which='both', direction='in',right=True, labelright=False)

    #plt.subplot(2, 1, 2)
    #plt.plot(q, np.abs(vp-vl))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim([1e-5, 10])
    ##plt.title(r'%s %s$_{s}(Q)$'%(pdfname.split('/')[0],alpha))
    #plt.ylabel(r'$|$%s$_{s,pdfflow} - $%s$_{s,lhapdf}|$'%(alpha,alpha))
    #plt.xlabel('Q')

    plt.savefig('diff_%s_alphas.png' % (pdfname.replace('/','-')), bbox_inches='tight',dpi=200)

    print("\nDry run time comparison:")
    #print("\tlhapdf: %f"%(l_time - s_time))
    print("{:>10}:{:>15.8f}".format("lhapdf", l_time - s_time))
    print("{:>10}:{:>15.8f}".format("pdfflow", p_time - l_time))


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start=time.time()
    main(**args)
    print("Total time: ", time.time()-start)
    
