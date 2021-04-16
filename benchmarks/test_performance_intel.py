"""
Benchmark script for LHAPDF comparison.
"""
import os
import sys
import argparse
from time import time
sys.path.append(os.path.dirname(__file__))


parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str,
                    help="The PDF set name/replica number.")
parser.add_argument("--mode", "-m", default="generate", type=str,
                    help="generate/run/plot")
parser.add_argument("--no_lhapdf", action="store_true",
                    help="Don't compute lhapdf performances.")
parser.add_argument("--no_mkl", action="store_true",
                    help="Don't use Intel MKL-DNN library")
parser.add_argument("--inter_op", default=None, type=int,
                    help="Number of threads used by independent non-blocking operations.")
parser.add_argument("--intra_op", default=None, type=int,
                    help="Number of parallel threads for ops speed ups.")
parser.add_argument("--no_tex", action="store_true",
                    help="Don't render pyplot with tex")
parser.add_argument("--fname", default="time_intel.pdf", type=str,
                    help="Output plot file name")


def set_variables(no_mkl, inter_op, intra_op):
    """
    Sets the environment variables and tune MKL-DNN parameters
    Parameters:
        args: dict
    """
    if no_mkl:
        os.environ["TF_DISABLE_MKL"] = "1"
    else:
        # from https://software.intel.com/content/www/us/en/develop/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus.html
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        os.environ["OMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"

    import tensorflow as tf
    #from tensorflow.python.eager import context
    if not no_mkl:
        # context._context = None
        # context._create_context()
        # tf.config.threading.set_inter_op_parallelism_threads(inter_op)
        # tf.config.threading.set_intra_op_parallelism_threads(intra_op)
        # best should be intra 48, inter 2
        tf.compat.v1.ConfigProto(inter_op_parallelism_threads=inter_op,
                                 intra_op_parallelism_threads=intra_op,
                                 allow_soft_placement=True,
                                 device_count={'CPU': intra_op})


def tensorboard_graph(pdfname):
    """
    This function creates a folder in benchmarks/ called tf_logdir.
    If folder is already present, overwrites its contents.
    Collects tensorflow logs and allows to visualize pdfflow tf.Graph.
    After this function call, run: tensorboard --logdir ../benchmarks/tf_logdir
    """
    from pdfflow.pflow import mkPDF
    import tensorflow as tf
    import shutil

    pdf = mkPDF(pdfname)
    logdir = "../benchmarks/tf_logdir"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    x, q2 = pdf.trace(tensorboard=True)
    pid = tf.constant([21], dtype=tf.int32)
    tf.summary.trace_on(graph=True, profiler=True)
    pdf.xfxQ2(pid, x, q2)
    with writer.as_default():
        tf.summary.trace_export(
            name="pdflow_graph",
            step=0,
            profiler_outdir=logdir)


def run_exp(pdfname, no_lhapdf, no_mkl, inter_op, intra_op):
    """
    Different kinds of computations:
    |              pdfflow
    |             /       \
    |       no_mkl         mkl
    |                    /  | \
    |                   /   |  \
    |                  /    |   \
    |               0,0   1,96  2,48
                   (inter_op,intra_op)
    Derfault: mkl with inter_op 1 and intra_op 96
    """
    dev = "CPU:0"
    ext = "nomkl" if no_mkl else f"{inter_op}_{intra_op}"
    set_variables(no_mkl, inter_op, intra_op)
    from compare_performance import run
    run(pdfname, 20, 10, no_lhapdf, dev, ext=ext)


def main(pdfname, mode, no_lhapdf, no_mkl, inter_op, intra_op, no_tex, fname):
    if mode == "generate":
        from compare_performance import generate
        generate(pdfname, 20, 10)
    elif mode == "run":
        run_exp(pdfname, no_lhapdf, no_mkl, inter_op, intra_op)
    elif mode == "plot":
        pdfname = "-".join(pdfname.split("/"))
        in_name = f"results_{pdfname}_{20}_{10}"
        texpdf = "PDFFlow" if no_tex else r"\texttt{PDFFlow}"
        args = [
            {
                "ext": "lhapdf",  # extension to input file name
                "label": "LHAPDF",
                "color": "blue",
                "marker": "o"
            },
            {
                "ext": "nomkl",
                "label": "%s: no mkl" % texpdf,
                "color": "lime",
                "marker": "^"
            },
            {
                "ext": "0_0",
                "label": "%s: (0,0)" % texpdf,
                "color": "darkslategrey",
                "marker": "^"
            },
            {
                "ext": "1_96",
                "label": "%s: (1,96)" % texpdf,
                "color": "coral",
                "marker": "^"
            },
            {
                "ext": "2_48",
                "label": "%s: (2,48)" % texpdf,
                "color": "darkred",
                "marker": "^"
            }
        ]
        from compare_performance import plot
        plot(in_name, fname, no_tex, args)
    elif mode == "graph":
        tensorboard_graph(pdfname)
    else:
        print("Unrecognized mode, please retry.")
        print("Available modes: generate / run / plot / graph.")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(time() - start)
