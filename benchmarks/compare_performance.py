"""
Benchmark script for LHAPDF comparison being hardware agnostic.
--mode flag has three possible values:
- generate: generate the input points
- run: run the experiment
- plot: collect results and plot

The user has to set the dictionary inside the code for making plots
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
parser.add_argument("--dirname", "-d", default="../benchmarks/tmp", type=str,
                    help="Folder where to place results and plots.")
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str,
                    help="The PDF set name/replica number.")
parser.add_argument("--mode", default="generate", type=str,
                    help="generate/run/plot")
parser.add_argument("--n_exp", default=10, type=int,
                    help="Number of experiments to average on.")
parser.add_argument("--start_point", "-s", default=1e5, type=float,
                    help="Start range of query array lengths.")
parser.add_argument("--end_point", "-e", default=1e6, type=float,
                    help="End range of query array lengths.")
parser.add_argument("--n_points", default=20, type=int,
                    help="Number of different query array lengths.")
parser.add_argument("--no_lhapdf", action="store_true",
                    help="Don't run lhapdf, only pdfflow")
parser.add_argument("-t", "--tensorboard", action="store_true",
                    help="Enable tensorboard profile logging")
parser.add_argument("--dev", default="GPU:*", type=str,
                    help="pdfflow running device: CPU/GPU:<n,*>/TPU")
parser.add_argument("--no_tex", action="store_true",
                    help="Don't render pyplot with tex")
parser.add_argument("--fname", default="time.pdf", type=str,
                    help="Output plot file name")
DIRNAME = (sp.run(["lhapdf-config", "--datadir"], stdout=sp.PIPE,
                  universal_newlines=True).stdout.strip("\n") + "/")


def pprint(x):
    """Print the given number in a compact formato for filenames"""
    exponent = int(np.log10(x))
    root = x/10**exponent

    return "{:.1f}e{:d}".format(root, exponent)


def set_env_vars(dev):
    """
    This function fixes the proper environment variables
    dev: str, could be one of CPU:<> / GPU:<> / TPU
    """
    if "GPU" in dev:
        print("Running PDFFlow on GPU")
        gpus = [i for i in range(len(tf.config.list_physical_devices('GPU')))]
        if len(gpus) == 0:
            raise AssertionError("No GPU found, please run with --dev CPU")
        gpu, gpu_n = dev.split(":")
        if gpu_n != "*":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_n

    if "CPU" in dev:
        print("Running PDFFlow on CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_run_inputs(dirname, pdfname, points, n_exp):
    pdfname = "-".join(pdfname.split("/"))

    start = pprint(points[0])
    end = pprint(points[-1])

    fname = os.path.join(dirname,
                         "_".join(["input", f"{pdfname}", f"{start}",
                                   f"{end}", f"{len(points)}", f"{n_exp}",
                                   "x.npy"]))
    x = np.load(fname)

    fname = fname[:fname.rfind("_")] + "_q2.npy"
    q2 = np.load(fname)  # shape [n_exp, all draws]

    return x.reshape([n_exp, -1]), q2.reshape([n_exp, -1])


def load_plot_inputs(dirname, fname, dev):
    fname = os.path.join(dirname, f"{fname}_{dev}.npy")
    res = np.load(fname)

    n = res[-1]
    mean = res[:-1].mean(0)
    std = res[:-1].std(0)/np.sqrt(len(n))

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
    start = time()
    p.py_xfxQ2_allpid(a_x, a_q2, strategy)
    return time() - start


def test_lhapdf(l_pdf, a_x, a_q2):
    start = time()
    f_lha = []
    for i in range(a_x.shape[0]):
        l_pdf.xfxQ2(a_x[i], a_q2[i])
    return time() - start


def accumulate_times(pdfname, points_exp_x, points_exp_q2, n_query, no_lhapdf, dev):
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
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = None
    p = pdf.mkPDF(pdfname, DIRNAME)
    p.trace(strategy)

    l_pdf = None if no_lhapdf else lhapdf.mkPDF(pdfname)

    xmin = np.exp(p.grids[0][0].log_xmin)
    xmax = np.exp(p.grids[0][0].log_xmax)
    q2min = np.sqrt(np.exp(p.grids[0][0].log_q2min))
    q2max = np.sqrt(np.exp(p.grids[0][-1].log_q2max))

    t_pdf = []
    t_lha = None if no_lhapdf else []
    n = []

    for exp_x, exp_q2 in tqdm.tqdm(zip(points_exp_x, points_exp_q2)):
        # iterate over the experiments
        tp = []
        tl = None if no_lhapdf else []

        xs = np.split(exp_x, n_query)[:-1]
        q2s = np.split(exp_q2, n_query)[:-1]

        for x, q2 in tqdm.tqdm(zip(xs, q2s)):
            # iterate over arrays of length query points
            tp.append(test_pdfflow(p, x, q2, strategy))

            if not no_lhapdf:
                tl.append(test_lhapdf(l_pdf, x, q2))
        t_pdf.append(tp)
        if not no_lhapdf:
            t_lha.append(tl)
    # t_pdf is a list with shape [n_exp, n_query]
    return np.array(t_pdf), np.array(t_lha)


def make_plots(fname, no_tex, args):
    """
    Function for making plots
    fname: str, plots output file name
    args: list of dicts containing at least the following keys
          n, mean, std, label, color, marker. lhapdf results must always be
          the first element of the list

    Note: the function is general except for the fine tuning on the tick axes
    """
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.1)

    ax = fig.add_subplot(gs[:-1, :])
    for v in args:
        n = v["n"]
        avg = v["mean"]
        err = v["std"]
        ax.errorbar(n, avg, yerr=err, label=v["label"],
                    linestyle='--', color=v["color"],
                    marker=v["marker"])
    PDFFLOW = "PDFFlow" if no_tex else r"\texttt{PDFFlow}"
    ax.title.set_text('%s - LHAPDF performances' % PDFFLOW)
    ax.set_ylabel(r'$t [s]$', fontsize=20)
    ticks = list(np.linspace(1e5, 1e6, 10))
    labels = [r'%d' % i for i in range(1, 11)]
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    ax.tick_params(axis='x', direction='in',
                   bottom=True, labelbottom=False,
                   top=True, labeltop=False)
    ax.tick_params(axis='y', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[-1, :])

    def unc(avg_l, std_l, avg_p, std_p):
        return np.sqrt((std_l/avg_p)**2 + (avg_l*std_p/(avg_p)**2)**2)

    for v in args:
        if v["label"] == "LHAPDF":
            continue
        n = v["n"]
        avg = v["mean"]
        std = v["std"]
        err = unc(args[0]["mean"], args[0]["std"], avg, std)
        ax.errorbar(n, args[0]["mean"]/avg, yerr=err, label=v["label"],
                    linestyle='--', color=v["color"],
                    marker=v["marker"])
    xlabel = '$x10^{5}$' if no_tex else r'$[\times 10^{5}]$'
    ax.set_xlabel(''.join([r'Number of $(x,Q)$ points drawn ', xlabel]),
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

    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close()


def generate(dirname, pdfname, points, n_exp):
    print("Generating inputs...")
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    p = pdf.mkPDF(pdfname, DIRNAME)

    xmin = np.exp(p.grids[0][0].log_xmin)
    xmax = np.exp(p.grids[0][0].log_xmax)
    q2min = np.sqrt(np.exp(p.grids[0][0].log_q2min))
    q2max = np.sqrt(np.exp(p.grids[0][-1].log_q2max))

    x = []
    q2 = []
    for experiment in range(n_exp):
        for point in points:
            x.append(np.random.uniform(xmin, xmax, [point, ]))
            q2.append(np.exp(np.random.uniform(np.log(q2min),
                                               np.log(q2max), [point, ])))
    x = np.concatenate(x)
    q2 = np.concatenate(q2)

    pdfname = "-".join(pdfname.split("/"))

    start = pprint(points[0])
    end = pprint(points[-1])
    fname = os.path.join(dirname,
                         "_".join(["input", f"{pdfname}", f"{start}",
                                   f"{end}", f"{len(points)}", f"{n_exp}",
                                   "x.npy"]))
    np.save(fname, x)

    fname = fname[:fname.rfind("_")] + "_q2.npy"
    np.save(fname, q2)


def run(dirname, pdfname, points, n_exp, no_lhapdf, dev, ext=None):
    """
    Run the experiment
    It's user's responsibility to load the appropriate input .npy files,
    set the correct flags.
    ext: str, if provided overrides the output filename extension
    """
    print("Running experiments...")

    set_env_vars(dev)

    x, q2 = load_run_inputs(dirname, pdfname, points, n_exp)
    n_query = points.cumsum()

    res_pdf, res_lha = accumulate_times(pdfname, x, q2, n_query, no_lhapdf, dev)

    pdfname = "-".join(pdfname.split("/"))
    if ext == None:
        fname = os.path.join(
            dirname,  "_".join([f"results_{pdfname}_{pprint(points[0])}",
                                f"{pprint(points[-1])}_{len(points)}",
                                f"{n_exp}_{dev}"]))
    else:
        fname = fname[:fname.rfind("_")] + f"_{ext}"
    np.save(fname, np.concatenate([res_pdf, n_query[None]], 0))

    if not no_lhapdf:
        fname = fname[:fname.rfind("_")] + "_lhapdf"
        np.save(fname, np.concatenate([res_lha, n_query[None]], 0))


def plot(dirname, in_fname, out_fname, no_tex, args):
    print("Collect results and plotting...")
    mpl.rcParams['text.usetex'] = not no_tex
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [7, 8]
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17
    mpl.rcParams['legend.fontsize'] = 18

    for arg in args:
        arg.update(load_plot_inputs(dirname, in_fname, arg["ext"]))
    make_plots(out_fname, no_tex, args)


def main(dirname, pdfname, mode, n_exp, start_point, end_point, n_points, no_lhapdf,
         tensorboard, dev, no_tex, fname):
    points = np.linspace(start_point, end_point, n_points).astype(int)
    start = pprint(points[0])
    end = pprint(points[-1])

    if mode == "generate":
        generate(dirname, pdfname, points, n_exp)
    if mode == "run":
        if tensorboard:
            tf.profile.experimental.start('logdir')
        run(dirname, pdfname, points, n_exp, no_lhapdf, dev)
        if tensorboard:
            tf.profile.experimental.stop('logdir')
    if mode == "plot":
        pdfname = "-".join(pdfname.split("/"))
        in_name = f"results_{pdfname}_{start}_{end}_{n_points}_{n_exp}"
        texpdf = "PDFFlow" if no_tex else r"\texttt{PDFFlow}"
        args = [
            {
                "ext": "lhapdf",  # extension to input file name
                "label": "LHAPDF",
                "color": "blue",
                "marker": "o"
            },
            {
                "ext": "CPU",
                "label": "%s: cpu" % texpdf,
                "color": "lime",
                "marker": "^"
            }
        ]
        plot(dirname, in_name, fname, no_tex, args)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(time() - start)
