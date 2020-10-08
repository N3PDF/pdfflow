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
import matplotlib.pyplot as plt
import matplotlib as mpl
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
                    help="Number of experiments to average on.")
parser.add_argument("--n_points", default=20, type=int,
                    help="Number of different query array lengths.")
parser.add_argument("--no_lhapdf", action="store_true",
                    help="Don't run lhapdf, only pdfflow")
parser.add_argument("-t", "--tensorboard", action="store_true",
                    help="Enable tensorboard profile logging")
parser.add_argument("--dev", default="GPU:*", type=str,
                    help="pdfflow running device: CPU:0/GPU:<n,*>/TPU")
parser.add_argument("--no_tex", action="store_false",
                    help="Don't render pyplot with tex")
parser.add_argument("--fname", default="time.pdf", type=str,
                    help="Output plot file name")
DIRNAME = (sp.run(["lhapdf-config", "--datadir"], stdout=sp.PIPE,
           universal_newlines=True).stdout.strip("\n") + "/")
DIRTMP = "../benchmarks/tmp/"

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


def load_run_inputs(pdfname, n_points, n_exp):
    n = np.linspace(1e5, 1e6, n_points).astype(int).cumsum()
    
    pdfname = "-".join(pdfname.split("/"))

    fname = "".join([DIRTMP,  f"input_{pdfname}_{n_points}_{n_exp}_x.npy"])
    x = np.load(fname)
    x = np.split(x.reshape([n_exp,-1]), n, axis=1)[:-1]

    fname = "".join([DIRTMP,  f"input_{pdfname}_{n_points}_{n_exp}_q2.npy"])
    q2 = np.load(fname) #shape [n_exp, all draws]
    q2 = np.split(q2.reshape([n_exp,-1]), n, axis=1)[:-1]

    return x, q2


def load_plot_inputs(fname, dev):
    fname = "".join([DIRTMP,
                     f"{fname}_{dev}.npy"])
    res = np.load(fname)

    n = res[:,-1]
    mean = res[:,:-1].mean(1)
    std = res[:,:-1].std(1)/np.sqrt(len(n))

    return {"n": n,
            "mean": mean,
            "std": std}


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
    n = []
    
    for exp_x, exp_q2 in tqdm.tqdm(zip(points_exp_x, points_exp_q2)):
        #iterate over n_points query lengths
        tp = []
        tl = None if no_lhapdf else []
        n.append(exp_x.shape[-1])
        for x, q2 in tqdm.tqdm(zip(exp_x, exp_q2)):
            # iterate over the experiments
            tp.append(test_pdfflow(p, x, q2, strategy))

            if not no_lhapdf:
                tl.append(test_lhapdf(l_pdf, x, q2))
        t_pdf.append(tp)
        if not no_lhapdf:
            t_lha.append(tl)
    # t_pdf is a list with shape [n_points, n_exp]
    # n is a list with shape [n_points]
    return np.array(n)[:,None], np.array(t_pdf), np.array(t_lha)


def make_plots(fname, no_tex, **kwargs):
    """
    Function for making plots
    fname: str, plots file name
    kwargs: format key = value
            value is a dict with the following keys:
            n, results, label, color, marker keys
            it must contain a kwarg called lhapdf!

    Note: the function is general except for the fine tuning on the tick axes
    """
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.1)

    ax = fig.add_subplot(gs[:-1,:])
    for key, v in kwargs.items():
        n = v["n"]
        avg = v["mean"]
        err = v["std"]
        ax.errorbar(n,avg,yerr=err,label=v["label"],
                    linestyle='--', color=v["color"],
                    marker=v["marker"])
    PDFFLOW= "PDFFlow" if no_tex else r"\texttt{PDFFlow}"
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
    
    def unc(avg_l, std_l, avg_p, std_p):
        return np.sqrt((std_l/avg_p)**2 + (avg_l*std_p/(avg_p)**2)**2)
    
    for key, v in kwargs.items():
        if key == "lha":
            continue
        n = v["n"]
        avg = v["mean"]
        std = v["std"]
        err = unc(kwargs["lha"]["mean"], kwargs["lha"]["std"], avg, std)
        ax.errorbar(n,avg/kwargs["lha"]["mean"],yerr=err,label=v["label"],
                    linestyle='--', color=v["color"],
                    marker=v["marker"])
    xlabel = r'$[\times 10^{5}]$' if no_tex else '$x10^{5}$'
    ax.set_xlabel(''.join([r'Number of $(x,Q)$ points drawn', xlabel]),
                  fontsize=18)
    ax.set_ylabel(r'Ratio to LHAPDF',
                  fontsize=18)
    ax.set_yscale('log')
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


def generate(pdfname, n_points, n_exp):
    print("Generate inputs")
    if not os.path.isdir(DIRTMP):
        os.makedirs(DIRTMP)

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
    fname = "".join([DIRTMP,  f"input_{pdfname}_{n_points}_{n_exp}_x"])
    np.save(fname, x)
    
    fname = "".join([DIRTMP,  f"input_{pdfname}_{n_points}_{n_exp}_q2"])
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

    x, q2 = load_run_inputs(pdfname, n_points, n_exp)

    n, res_pdf, res_lha = accumulate_times(pdfname, x, q2, no_lhapdf, dev)

    pdfname = "-".join(pdfname.split("/"))
    fname = "".join([DIRTMP,  f"results_{pdfname}_{n_points}_{n_exp}_{dev}"])
    print(n.shape)
    print(res_pdf.shape)
    np.save(fname, np.concatenate([res_pdf, n], 1))

    if not no_lhapdf:
        fname = "".join([DIRTMP,  f"results_{pdfname}_{n_points}_{n_exp}_lhapdf"])
        np.save(fname, np.concatenate([res_lha, n], 1))


def plot(in_fname, n_points, dev, out_fname, no_tex):
    print("Collect results and plotting")
    mpl.rcParams['text.usetex'] = no_tex
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [7,8]
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17
    mpl.rcParams['legend.fontsize'] = 18

    texpdf = "PDFFlow" if no_tex else r"\texttt{PDFFlow}"
    lha = {
        **load_plot_inputs(in_fname, "lhapdf"),
        "label": "LHAPDF",
        "color": "blue",
        "marker": "o"
    }
    pdf = {
        **load_plot_inputs(in_fname, "CPU:0"),
        "label": "%s: cpu"%texpdf,
        "color": "lime",
        "marker": "^"
    }
    make_plots(out_fname, no_tex, lha=lha, pdf=pdf)


def main(pdfname, mode, n_exp, n_points, no_lhapdf, tensorboard, dev, no_tex,
         fname):
    if mode == "generate":
        generate(pdfname, n_points, n_exp)
    if mode == "run":
        if tensorboard:
            tf.profile.experimental.start('logdir')
        run(pdfname, n_points, n_exp, no_lhapdf, dev)
        if tensorboard:
            tf.profile.experimental.stop('logdir')
    if mode == "plot":
        pdfname = "-".join(pdfname.split("/"))
        in_name = f"results_{pdfname}_{n_points}_{n_exp}"
        plot(in_name, n_points, dev, fname, no_tex)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(time() - start)
