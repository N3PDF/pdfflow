"""
Benchmark script for LHAPDF comparison being hardware agnostic.
--mode flag has three possible values:
- generate: generate the input points
- run: run the experiment
- plot: collect results and plot
"""
import os
import glob
import lhapdf
import argparse
import subprocess as sp
import numpy as np
import tensorflow as tf
from time import time
import tqdm
import pdfflow.pflow as pdf
from pdfflow.configflow import float_me

parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str,
                    help="The PDF set name/replica number.")
parser.add_argument("--mode", default="generate", type=str,
                    help="generate/run/plot")
parser.add_argument("--n_exp", default=10, type=int,
                    help="Number of experiments.")
parser.add_argument("--n_points", default=20, type=int,
                    help="Number of different query array lengths.")
parser.add_argument("--no_lhapdf", action="store_false",
                    help="Don't run lhapdf, only pdfflow")
parser.add_argument("-t", "--tensorboard", action="store_true",
                    help="Enable tensorboard profile logging")
parser.add_argument("--dev", default="GPU:*", type=str,
                    help="pdfflow running device: CPU:0/GPU:<n,*>/TPU")
parser.add_argument("--no_tex", action="store_false",
                    help="Don't render pyplot with tex")
DIRNAME = (sp.run(["lhapdf-config", "--datadir"], stdout=sp.PIPE,
           universal_newlines=True).stdout.strip("\n") + "/")

def set_env_vars(dev):
    """
    This function fixes the proper environment variables
    dev: str, could be one of CPU:<> / GPU:<> / TPU
    """
    if "GPU" in dev:
        print("Running PDFFlow on GPU")
        gpu, gpu_n = dev.split(":")
        gpus = [i for i in range(len(tf.config.list_physical_devices('GPU')))]
        gpus.append("*") # add the possibility to take all the GPUs
        if not gpu_n in gpus:
            raise AssertionError("Selected GPU not present on machine")
        else:
            if gpu_n !=  "*":
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_n

    if "CPU" in dev:
        print("Running PDFFlow on CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_inputs(pdfname, n_points, n_exp):
    n = np.linspace(1e5, 1e6, n_points).astype(int).cumsum()
    
    pdfname = "-".join(pdfname.split("/"))
    dir_name = "../benchmarks/tmp/"

    fname = "".join([dir_name,  f"input_{pdfname}_{n_points}_{n_exp}_x.npy"])
    x = np.load(fname)
    x = np.split(x.reshape([n_exp,-1]), n, axis=1)[:-1]

    fname = "".join([dir_name,  f"input_{pdfname}_{n_points}_{n_exp}_q2.npy"])
    q2 = np.load(fname)
    q2 = np.split(q2.reshape([n_exp,-1]), n, axis=1)[:-1]

    return x, q2


def test_pdfflow(p, a_x, a_q2, strategy):
    """
    Test pdfflow
    Parameters:
        p: PDF object
        a_x: numpy array of inputs
        a_q2: numpy array of inputs
        strategy: TPUStrategy, allows TPU computation
    """
    if strategy == None:
        start = time()
        p.py_xfxQ2_allpid(a_x, a_q2)
    else:
        start = time()
        a_x = float_me(a_x)
        a_q2 = float_me(a_q2)
        strategy.run(p.xfxQ2_allpid, args=(a_x, a_q2))
    return time() - start


def test_lhapdf(l_pdf, a_x, a_q2):
    start = time()
    f_lha = []
    for i in range(a_x.shape[0]):
        l_pdf.xfxQ2(a_x[i], a_q2[i])
    return time() - start


def accumulate_times(pdfname, points_exp_x, points_exp_q2, no_lhapdf, dev):
    """
    Computes performance times
    Parameters:
        p: PDF object
        x: list, x arrays to be passed as inputs
        q2: list, q2 arrays to be passed as inputs
        no_lhapdf: bool, if not to do also lhapdf times
    """
    if dev == "TPU":
        print("Running PDFFlow on TPU")
        tpu_address = os.environ["TPU_NAME"]
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = None
    p = pdf.mkPDF(pdfname, DIRNAME)
    p.trace()

    l_pdf = None if no_lhapdf else lhapdf.mkPDF(pdfname)

    xmin = np.exp(p.grids[0][0].log_xmin)
    xmax = np.exp(p.grids[0][0].log_xmax)
    q2min = np.sqrt(np.exp(p.grids[0][0].log_q2min))
    q2max = np.sqrt(np.exp(p.grids[0][-1].log_q2max))

    t_pdf = []
    t_lha = None if no_lhapdf else []
    
    for exp_x, exp_q2 in tqdm.tqdm(zip(points_exp_x, points_exp_q2)):
        #iterate over n_points query lengths
        tp = []
        tl = None if no_lhapdf else []
        for x, q2 in tqdm.tqdm(zip(exp_x, exp_q2)):
            # iterate over the experiments
            tp.append(test_pdfflow(p, x, q2, strategy))

            if not no_lhapdf:
                tl.append(test_lhapdf(l_pdf, x, q2))
        t_pdf.append(tp)
        if not no_lhapdf:
            t_lha.append(tl)
    return np.array(t_pdf), np.array(t_lha)


def generate(pdfname, n_points, n_exp):
    print("Generate inputs")
    dir_name = "../benchmarks/tmp/"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    p = pdf.mkPDF(pdfname, DIRNAME)

    xmin = np.exp(p.grids[0][0].log_xmin)
    xmax = np.exp(p.grids[0][0].log_xmax)
    q2min = np.sqrt(np.exp(p.grids[0][0].log_q2min))
    q2max = np.sqrt(np.exp(p.grids[0][-1].log_q2max))

    n = np.linspace(1e5, 1e6, n_points).astype(int)
    x = []
    q2 = []
    for j in range(n_exp):
        for i in n:
            x.append(np.random.uniform(xmin, xmax,[i,]))
            q2.append(np.exp(np.random.uniform(np.log(q2min),
                                        np.log(q2max),[i,])))
    x = np.concatenate(x)
    q2 = np.concatenate(q2)

    pdfname = "-".join(pdfname.split("/"))
    fname = "".join([dir_name,  f"input_{pdfname}_{n_points}_{n_exp}_x"])
    np.save(fname, x)
    
    fname = "".join([dir_name,  f"input_{pdfname}_{n_points}_{n_exp}_q2"])
    np.save(fname, q2)


def run(pdfname, n_points, n_exp, no_lhapdf, dev):
    """
    Run the experiment
    It's user's responsibility to load the appropriate input .npy files,
    set the correct flags.
    """
    print("Running experiments")
    print("Loading inputs:")

    set_env_vars(dev)

    x, q2 = load_inputs(pdfname, n_points, n_exp)

    res_pdf, res_lha = accumulate_times(pdfname, x, q2, no_lhapdf, dev)

    dir_name = "../benchmarks/tmp/"
    pdfname = "-".join(pdfname.split("/"))
    fname = "".join([dir_name,  f"results_{pdfname}_{n_points}_{n_exp}_{dev}"])
    np.save(res_pdf, fname)

    if not no_lhapdf:
        fname = "".join([dir_name,  f"results_{pdfname}_{n_points}_{n_exp}_lhapdf"])
        np.save(res_lha, fname)


def plot():
    print("Collect results and plotting")


def main_2(pdfname=None, n_draws=10, pid=21, no_lhapdf=False,
         tensorboard=False, dev0=None, dev1=None,
         label0=None, label1=None, no_tex=True):
    """Testing PDFflow vs LHAPDF performance."""
    if tensorboard:
        tf.profiler.experimental.start('logdir')
    
    #check legend labels
    if label0 == None:
        label0 = dev0
    
    if label1 == None:
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


def main(args):
    if args["mode"] == "generate":
        generate(args["pdfname"], args["n_points"], args["n_exp"])
    if args["mode"] == "run":
        run(args["pdfname"], args["n_points"], args["n_exp"], args["no_lhapdf"], args["dev"])
    if args["mode"] == "plot":
        plot()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(args)
    print(time() - start)
