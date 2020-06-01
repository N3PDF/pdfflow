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

# def sort_arrays_Qx(a,b,f):
#     a = np.array(a)
#     b = np.array(b)
#     f = np.array(f)
#     
#     ind = np.argsort(a)
# 
#     aa = np.take_along_axis(a,ind,0)
#     bb = np.take_along_axis(b,ind,0)
#     ff = np.take_along_axis(f,ind,0)
# 
#     ind = np.argsort(bb)
#     return np.take_along_axis(aa,ind,0), np.take_along_axis(bb,ind,0), np.take_along_axis(ff,ind,0)
# 
# 
# def sort_arrays_xQ(a,b,f):
#     a = np.array(a)
#     b = np.array(b)
#     f = np.array(f)
#     ind = np.argsort(b)
# 
#     aa = np.take_along_axis(a,ind,0)
#     bb = np.take_along_axis(b,ind,0)
#     ff = np.take_along_axis(f,ind,0)
# 
#     ind = np.argsort(aa)
#     return np.take_along_axis(aa,ind,0), np.take_along_axis(bb,ind,0), np.take_along_axis(ff,ind,0)

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

    x = [0.5]
    q2 = [100.]
    s = time.time()
    p.xfxQ2(21, x,q2)
    print("\nPDFflow\n\tBuilding graph time: %f\n"%(time.time()-s))

    plt.figure(figsize=(16.0, 12.0))
    plt.subplot(2, 2, 1)
    x = np.logspace(-11,0,100000, dtype=float)
    q2 = np.array([0.1,1.65,1.7,4.92,1e2,1e3,1e4,1e5,1e6,2e6], dtype=float)**2
    for iq2 in q2:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for ix in x])
        vp = p.xfxQ2(pid, float_me(x), float_me([iq2]*len(x)))
        #print('%e'%np.sqrt(iq2), vl,vp,'\n')

        plt.plot(x, np.abs(vp-vl)/(np.abs(vl)+EPS), label='$Q=%.2e$' % iq2**0.5)
    plt.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|/(|f_{lhapdf}|+eps$)')
    plt.xlabel('x')
    plt.legend()

    plt.subplot(2, 2, 3)
    for iq2 in q2:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for ix in x])
        vp = p.xfxQ2(pid, float_me(x), float_me([iq2]*len(x)))
        
        plt.plot(x, np.abs(vp-vl), label='$Q=%.2e$' % iq2**0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|$')
    plt.xlabel('x')
    plt.legend()

    x = np.array([1e-10,1e-9,1.1e-9,5e-7,1e-6,1e-4,1e-2,0.5,0.99], dtype=float)
    q2 = np.logspace(-3, 7, 100000, dtype=float)**2
    plt.subplot(2, 2, 2)
    for ix in x:
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for iq2 in q2])
        vp = p.xfxQ2(pid, float_me([ix]*len(q2)), float_me(q2))
        
        plt.plot(q2**0.5, np.abs(vp-vl)/(np.abs(vl)+EPS), label='$x=%.2e$' % ix)
    plt.hlines(1e-3, plt.xlim()[0], plt.xlim()[1], linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|/(|f_{lhapdf}|+eps$)')
    plt.xlabel('Q')
    plt.legend()

    plt.subplot(2, 2, 4)
    for ix in x:
        s_time = time.time()
        vl = np.array([l_pdf.xfxQ2(pid, ix, iq2) for iq2 in q2])
        l_time = time.time()
        vp = p.xfxQ2(pid, float_me([ix]*len(q2)), float_me(q2))
        p_time = time.time()
        
        plt.plot(q2**0.5, np.abs(vp-vl), label='$x=%.2e$' % ix)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-5, 10])
    plt.title('%s, flav = %d' % (pdfname, pid))
    plt.ylabel(r'$|f_{pdfflow} - f_{lhapdf}|$')
    plt.xlabel('Q')
    plt.legend()

    plt.savefig('diff_%s_flav%d.png' % (pdfname.replace('/','-'), pid), bbox_inches='tight')

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
    
