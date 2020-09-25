"""
Benchmark script for LHAPDF comparison.
"""
import os

import lhapdf
import pdfflow.pflow as pdf
import argparse
import subprocess as sp
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context

from time import time
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str,
                    help="The PDF set name/replica number.")
parser.add_argument("--n-draws", default=100000, type=int,
                    help="Number of trials.")
parser.add_argument("--pid", default=21, type=int,
                    help="The flavour PID.")
parser.add_argument("-t", "--tensorboard", action="store_true",
                    help="Enable tensorboard profile logging")
parser.add_argument("--plot", action="store_true",
                    help="Just collect and plot results")
parser.add_argument("--no_mkl", action="store_true",
                    help="do not use MKL-DNN")
parser.add_argument("--lhapdf", action="store_true",
                    help="do not compute lhapdf")
parser.add_argument("--intra_op", default=1, type=int,
                    help="TensorFlow intra-op parameter")
parser.add_argument("--inter_op", default=96, type=int,
                    help="TensorFlow inter-op parameter")
parser.add_argument("--verbose", action="store_true",
                    help="show mkl-dnn verbose")
DIRNAME = (sp.run(["lhapdf-config", "--datadir"], stdout=sp.PIPE,
           universal_newlines=True).stdout.strip("\n") + "/")

def set_variables(args, verbose=False):
    """
    Sets the environment variables and tune MKL-DNN parameters
    Parameters:
        args: dict
        verbose: bool
    """
    if args is not None:
        if verbose:
            os.environ["MKLDNN_VERBOSE"] = "1"
        os.environ["KMP_AFFINITY"] = args['KMP_AFFINITY']
        os.environ["KMP_BLOCKTIME"] = args["KMP_BLOCKTIME"]#'1'
        os.environ["KMP_SETTINGS"] = args["KMP_SETTINGS"]
        os.environ["OMP_NUM_THREADS"] = args['OMP_NUM_THREADS']#'96'

        context._context = None
        context._create_context()
        if args["inter_op"] is not None:
            tf.config.threading.set_inter_op_parallelism_threads(args["inter_op"])
            tf.config.threading.set_intra_op_parallelism_threads(args["intra_op"])
    else:
        os.environ["TF_DISABLE_MKL"] = "1"


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


def accumulate_times(pdfname, no_lhapdf=True, args=None, verbose=False):
    """
    Computes performance times
    Parameters:
        pdfname: str
        no_lhapdf: str, device name over which run pdfflow
    """
    if not no_lhapdf:
        p = lhapdf.mkPDF(pdfname)
        xmin = 1e-9
        xmax = 1
        q2min = 1.65**2
        q2max = 1e10
    else:
        set_variables(args, verbose)
        p = pdf.mkPDF(pdfname, DIRNAME)
        p.trace()
        xmin = np.exp(p.grids[0][0].log_xmin)
        xmax = np.exp(p.grids[0][0].log_xmax)
        q2min = np.sqrt(np.exp(p.grids[0][0].log_q2min))
        q2max = np.sqrt(np.exp(p.grids[0][-1].log_q2max))

    t_tot = []
    
    n = np.linspace(1e5,1e6,20)
    for j in range(10):
        t = []
        for i in tqdm.tqdm(n):
            a_x = np.random.uniform(xmin, xmax,[int(i),])
            a_q2 = np.exp(np.random.uniform(np.log(q2min),
                                            np.log(q2max),[int(i),]))
            if not no_lhapdf:
                t_ = test_lhapdf(p, a_x, a_q2)
            else:
                t_ =  test_pdfflow(p, a_x, a_q2)
            t += [t_]
            
        t_tot += [t]

    t_tot = np.stack(t_tot)
    return n, t_tot


def compute(pdfname=None, n_draws=10, pid=21, tensorboard=False,
            no_mkl=False, lhapdf=False, intra_op="1", inter_op="96", verbose=False):
    """
    Different kinds of computations:
    |lhapdf---or---pdfflow
    |             /       \
    |       no_mkl         mkl
    |                    /  | \
    |                   /   |  \
    |                  /    |   \
    |                 0   1,96  2,96
                     (inter_op,intra_op)
    Derfault: mkl with inter_op 1 and intra_op 96
    """
    if tensorboard:
        tf.profiler.experimental.start('logdir')

    if lhapdf:
        n, t = accumulate_times(pdfname, False)
    
        fname_n = "../benchmarks/n_lha"
        fname_t = "../benchmarks/t_lha"
    else:        
        if no_mkl:
            n, t = accumulate_times(pdfname)
            fname_n = "../benchmarks/n_no_mkl"
            fname_t = "../benchmarks/t_no_mkl"
        else:
            args = {}
            args["KMP_AFFINITY"] = "granularity=tile,compact"
            args["KMP_BLOCKTIME"] = "0"
            args["KMP_SETTINGS"] = "1"
            args["OMP_NUM_THREADS"] = "96"
            args["inter_op"] = inter_op
            args["intra_op"] = intra_op

            n, t = accumulate_times(pdfname, args=args, verbose=verbose)
            fname_n = f"../benchmarks/n_inter{inter_op}_intra{intra_op}"
            fname_t = f"../benchmarks/t_inter{inter_op}_intra{intra_op}"

    if tensorboard:
        tf.profiler.experimental.stop('logdir')

    np.save(fname_n, n)
    np.save(fname_t, t)

def make_plot(inter_op="1", intra_op="96"):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [7,8]
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17
    mpl.rcParams['legend.fontsize'] = 18

    dir_name = "../benchmarks/"
    n_lha = np.load(dir_name + "n_lha.npy")
    t_lha = np.load(dir_name + "t_lha.npy")
    
    n_def = np.load(dir_name + "n_no_mkl.npy")
    t_def = np.load(dir_name + "t_no_mkl.npy")

    #n_0 = np.load(dir_name + f"n_{}.npy")
    #t_0 = np.load(dir_name + f"t_{}.npy")

    n_1 = np.load(dir_name + f"n_inter{inter_op}_intra{intra_op}.npy")
    t_1 = np.load(dir_name + f"t_inter{inter_op}_intra{intra_op}.npy")

    n_2 = np.load(dir_name + f"n_inter{inter_op}_intra{intra_op}.npy")
    t_2 = np.load(dir_name + f"t_inter{inter_op}_intra{intra_op}.npy")

    n_3 = np.load(dir_name + f"n_inter{inter_op}_intra{intra_op}.npy")
    t_3 = np.load(dir_name + f"t_inter{inter_op}_intra{intra_op}.npy")


    def unc(t_l, t_p):
        avg_l = t_l.mean(0)
        std_l = t_l.std(0)
        avg_p = t_p.mean(0)
        std_p = t_p.std(0)
        return np.sqrt((std_l/avg_p)**2 + (avg_l*std_p/(avg_p)**2)**2)

    k = len(t_lha)**0.5

        
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.1)

    ax = fig.add_subplot(gs[:-1,:])
    ax.errorbar(n_lha,t_lha.mean(0),yerr=t_lha.std(0)/k,
                label=r'LHAPDF',
                linestyle='--', color='blue', marker='^')
    ax.errorbar(n_def,t_def.mean(0),yerr=t_def.std(0)/k,
                label=r'\texttt{PDFFlow}: no MKL-DNN',
                linestyle='--', color='lime', marker='s')
    #ax.errorbar(n_0,t_0.mean(0),yerr=t_0.std(0)/k,
    #            label=r'\texttt{PDFFlow}: None, None ',
    #            linestyle='--', color='salmon', marker='o')
    ax.errorbar(n_1,t_1.mean(0),yerr=t_1.std(0)/k,
                label=r'\texttt{PDFFlow}: 0, 0',
                linestyle='--', color='sandybrown', marker='o')
    ax.errorbar(n_2,t_2.mean(0),yerr=t_2.std(0)/k,
                label=r'\texttt{PDFFlow}: 1, 96',
                linestyle='--', color='red', marker='o')
    ax.errorbar(n_3,t_3.mean(0),yerr=t_3.std(0)/k,
                label=r'\texttt{PDFFlow}: 2, 96',
                linestyle='--', color='darkmagenta', marker='o')
    ax.title.set_text(r'\texttt{PDFflow} - LHAPDF perfomances')
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
    ax.errorbar(n_def,t_lha.mean(0)/t_def.mean(0),yerr=unc(t_lha,t_def)/k,
                label=r'\texttt{PDFFlow}: no MKL-DNN',
                linestyle='--', color='lime', marker='s')
    #ax.errorbar(n_0,t_lha.mean(0)/t_0.mean(0),yerr=unc(t_lha,t_0)/k,
    #            label=r'\texttt{PDFFlow}: None, None ',
    #            linestyle='--', color='salmon', marker='o')
    ax.errorbar(n_1,t_lha.mean(0)/t_1.mean(0),yerr=unc(t_lha,t_1)/k,
                label=r'\texttt{PDFFlow}: 0, 0',
                linestyle='--', color='sandybrown', marker='o')
    ax.errorbar(n_2,t_lha.mean(0)/t_2.mean(0),yerr=unc(t_lha,t_2)/k,
                label=r'\texttt{PDFFlow}: 1, 96',
                linestyle='--', color='red', marker='o')
    ax.errorbar(n_3,t_lha.mean(0)/t_3.mean(0),yerr=unc(t_lha,t_3)/k,
                label=r'\texttt{PDFFlow}: 2, 96',
                linestyle='--', color='darkmagenta', marker='o')
    ax.set_xlabel(r'Number of $(x,Q)$ points drawn $[\times 10^{5}]$',
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


def main(args):
    """
    Testing PDFflow vs LHAPDF performance on Intel hardware.
    Exploits MKL-DNN library
    """

    if args["plot"]:
        make_plot(intra_op=args["intra_op"], inter_op=args["inter_op"])
    else:
        del args["plot"]
        compute(**args)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(args)
    print(time() - start)