"""
Benchmark script for LHAPDF comparison.
"""
from pdfflow.configflow import float_me
import lhapdf
import argparse
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str, help='The PDF set name/replica number.')
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE, universal_newlines=True).stdout.strip('\n') + '/'
EPS = np.finfo(float).eps

def main(pdfname):
    """Testing PDFflow vs LHAPDF performance."""
    import pdfflow.pflow as pdf
    alpha = '\u03B1'

    p = pdf.mkPDF(pdfname, DIRNAME, alpha_computation=True)
    l_pdf = lhapdf.mkPDF(pdfname)

    q = [100.]
    s = time.time()
    p.py_alphaSQ(q)
    print("\nPDFflow AlphaS\n\tBuilding graph time: %f\n"%(time.time()-s))

    plt.figure(figsize=(16.0, 12.0))
    
    q = np.logspace(-3, 10, 1000000, dtype=float)
    plt.subplot(2, 1, 1)

    s_time = time.time()
    vl = np.array([l_pdf.alphasQ(iq) for iq in q])
    l_time = time.time()
    vp = p.py_alphaSQ(q)
    p_time = time.time()

        
    plt.plot(q, np.abs(vp-vl)/(np.abs(vl)+EPS))
    plt.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title(r'%s %s$_{s}(Q)$'%(pdfname.split('/')[0],alpha))
    plt.ylabel(r'$|$%s$_{s,pdfflow} - $%s$_{s,lhapdf}|/(|$%s$_{s,lhapdf}|+eps)$'%(alpha,alpha,alpha))
    plt.xlabel('Q')

    plt.subplot(2, 1, 2)
    plt.plot(q, np.abs(vp-vl))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title(r'%s %s$_{s}(Q)$'%(pdfname.split('/')[0],alpha))
    plt.ylabel(r'$|$%s$_{s,pdfflow} - $%s$_{s,lhapdf}|$'%(alpha,alpha))
    plt.xlabel('Q')

    plt.savefig('diff_%s_alphaS.png' % (pdfname.replace('/','-')), bbox_inches='tight')

    print("\nDry run time comparison:")
    #print("\tlhapdf: %f"%(l_time - s_time))
    print("{:>10}:{:>15.8f}".format("lhapdf", l_time - s_time))
    print("{:>10}:{:>15.8f}".format("pdfflow", p_time - l_time))


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start=time.time()
    main(**args)
    print("Total time: ", time.time()-start)
    
