"""
Benchmark script for LHAPDF comparison.
"""
import lhapdf
import pdfflow.pflow as pdf
import argparse
import subprocess as sp
import numpy as np
import tensorflow as tf
from time import time
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str,
                    help="The PDF set name/replica number.")
parser.add_argument("--n-draws", default=100000, type=int,
                    help="Number of trials.")
parser.add_argument("--pid", default=21, type=int,
                    help="The flavour PID.")
parser.add_argument("--no_lhapdf", action="store_false",
                    help="Don't run lhapdf, only pdfflow")
parser.add_argument("-t", "--tensorboard", action="store_true",
                    help="Enable tensorboard profile logging")
parser.add_argument("--dev0", default=None, type=str,
                    help="First pdfflow running device")
parser.add_argument("--dev1", default=None, type=str,
                    help="Second pdfflow running device")
parser.add_argument("--label0", default=None, type=str,
                    help=" ".join(["Legend label of first pdfflow running device,",
                                    "defaults to tf device auto selection"]))
parser.add_argument("--label1", default=None, type=str,
                    help=" ".join(["Legend label of second pdfflow running device,",
                                    "defaults to tf device auto selection"]))
parser.add_argument("--no_tex", action="store_false",
                    help="Don't render pyplot with tex")
DIRNAME = (sp.run(["lhapdf-config", "--datadir"], stdout=sp.PIPE,
           universal_newlines=True).stdout.strip("\n") + "/")


def test_pdfflow(p, a_x, a_q2):
    start = time()
    p.py_xfxQ2_allpid(a_x, a_q2)
    return time() - start


def test_lhapdf(l_pdf, a_x, a_q2):    
    start = time()
    f_lha = []
    for i in range(a_x.shape[0]):
        l_pdf.xfxQ2(a_x[i], a_q2[i])
    return time() - start


def accumulate_times(pdfname, dev0, dev1, no_lhapdf):
    """
    Computes performance times
    Parameters:
        dev0: str, device name over which run pdfflow
        dev1: str, device name over which run pdfflow
    """
    with tf.device(dev0):
        p0 = pdf.mkPDF(pdfname, DIRNAME)
        p0.trace()

    if dev1 is not None:
        with tf.device(dev1):
            p1 = pdf.mkPDF(pdfname, DIRNAME)
            p1.trace()

    if no_lhapdf:
        l_pdf = lhapdf.mkPDF(pdfname)
    else:
        l_pdf = None

    xmin = np.exp(p0.grids[0][0].log_xmin)
    xmax = np.exp(p0.grids[0][0].log_xmax)
    q2min = np.sqrt(np.exp(p0.grids[0][0].log_q2min))
    q2max = np.sqrt(np.exp(p0.grids[0][-1].log_q2max))

    t_pdf0 = []
    t_pdf1 = []
    t_lha = []
    
    n = np.linspace(1e5,1e6,2)
    for j in range(2):
        t0 = []
        t1 = []
        t2 = []
        for i in tqdm.tqdm(n):
            a_x = np.random.uniform(xmin, xmax,[int(i),])
            a_q2 = np.exp(np.random.uniform(np.log(q2min),
                                            np.log(q2max),[int(i),]))
            with tf.device(dev0):
                t_ =  test_pdfflow(p0, a_x, a_q2)
            t0 += [t_]

            if dev1 is not None:
                with tf.device(dev1):
                    t_ =  test_pdfflow(p1, a_x, a_q2)
                t1 += [t_]
            else:
                t1 += [[]]

            t_ = test_lhapdf(l_pdf, a_x, a_q2) if no_lhapdf else []
            t2 += [t_]
            
        t_pdf0 += [t0]
        t_pdf1 += [t1]
        t_lha += [t2]

    t_pdf0 = np.stack(t_pdf0)
    t_pdf1 = np.stack(t_pdf1)
    t_lha = np.stack(t_lha)

    return n, t_pdf0, t_pdf1, t_lha


def main(pdfname=None, n_draws=10, pid=21, no_lhapdf=False,
         tensorboard=False, dev0=None, dev1=None,
         label0=None, label1=None, no_tex=True):
    """Testing PDFflow vs LHAPDF performance."""
    if tensorboard:
        tf.profiler.experimental.start('logdir')
    
    #check legend labels
    if label0 is None:
        label0 = dev0
    
    if label1 is None:
        label1 = dev1

    n, t_pdf0, t_pdf1, t_lha = accumulate_times(pdfname, dev0, dev1, no_lhapdf)

    if tensorboard:
        tf.profiler.experimental.stop('logdir')

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = no_tex
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [7,8]
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17
    mpl.rcParams['legend.fontsize'] = 18

    avg_l = t_lha.mean(0)
    avg_p0 = t_pdf0.mean(0)
    avg_p1 = t_pdf1.mean(0) if dev1 is not None else None
    std_l = t_lha.std(0)
    std_p0 = t_pdf0.std(0)
    std_p1 = t_pdf1.std(0) if dev1 is not None else None

    std_ratio0 = np.sqrt((std_l/avg_p0)**2 + (avg_l*std_p0/(avg_p0)**2)**2)
    std_ratio1 = np.sqrt((std_l/avg_p1)**2 + (avg_l*std_p1/(avg_p1)**2)**2)\
                 if dev1 is not None else None

    k = len(t_pdf0)**0.5

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.1)

    ax = fig.add_subplot(gs[:-1,:])
    PDFFLOW = r'\texttt{PDFFlow}' if no_tex else r'PDFFlow'
    ax.errorbar(n,avg_p0,yerr=std_p0,label=r'%s: %s'%(PDFFLOW, label0),
                linestyle='--', color='b', marker='^')
    ax.errorbar(n,avg_p1,yerr=std_p1,label=r'%s: %s'%(PDFFLOW, label1),
                linestyle='--', color='#ff7f0e', marker='s')
    ax.errorbar(n,avg_l,yerr=std_l/k,label=r'LHAPDF (CPU)',
                linestyle='--', color='g', marker='o')
    ax.title.set_text('%s - LHAPDF performances'%PDFFLOW)
    ax.set_ylabel(r'$t [s]$', fontsize=20)
    ticks = list(np.linspace(1e5,1e6,10))
    labels = [r'%d'%i for i in range(1,11)]
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    ax.tick_params(axis='x', direction='in',
                   bottom=True, labelbottom=False,
                   top=True, labeltop=False)
    ax.tick_params(axis='y', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[-1,:])
    ax.errorbar(n, (avg_l/avg_p0),yerr=std_ratio0/k, label=r'%s'%label0,
                linestyle='--', color='b', marker='^')
    ax.errorbar(n, (avg_l/avg_p1),yerr=std_ratio1/k, label=r'%s'%label1,
                linestyle='--', color='#ff7f0e', marker='s')
    xlabel = r'$[\times 10^{5}]$' if no_tex else '$x10^{5}$'
    ax.set_xlabel(''.join([r'Number of $(x,Q)$ points drawn', xlabel]),
                  fontsize=18)
    ax.set_ylabel(r'Ratio to LHAPDF',
                  fontsize=18)
    ax.set_yscale('log')
    ticks = list(np.linspace(1e5,1e6,10))
    labels = [r'%d'%i for i in range(1,11)]
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    ax.tick_params(axis='x', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False)
    ax.tick_params(axis='y', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)

    plt.savefig('time.pdf', bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(time() - start)
