"""
Benchmark script for LHAPDF comparison.
"""
import os
import sys
import lhapdf
import pdfflow.pflow as pdf
import argparse
import subprocess as sp
import numpy as np
from tensorflow.python.eager import context
from time import time
import tqdm
sys.path.append(os.path.dirname(__file__))
from compare_performance import generate, run, plot


parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str,
                    help="The PDF set name/replica number.")
parser.add_argument("--mode", default="generate", type=str,
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
    import tensorflow as tf
    if not no_mkl:
        tf.config.threading.set_inter_op_parallelism_threads(inter_op)
        tf.config.threading.set_intra_op_parallelism_threads(intra_op)
    
    #os.environ["KMP_AFFINITY"] = args['KMP_AFFINITY']
    #os.environ["KMP_BLOCKTIME"] = args["KMP_BLOCKTIME"]#'1'
    #os.environ["KMP_SETTINGS"] = args["KMP_SETTINGS"]
    #os.environ["OMP_NUM_THREADS"] = args['OMP_NUM_THREADS']#'96'

    # context._context = None
    # context._create_context()
    #import tensorflow as tf
    #if args["inter_op"] is not None:
    #    tf.config.threading.set_inter_op_parallelism_threads(args["inter_op"])
    #    tf.config.threading.set_intra_op_parallelism_threads(args["intra_op"])
    #else:
    #    os.environ["TF_DISABLE_MKL"] = "1"



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
    set_variables(args, inter_op, intra_op)
    run(pdfname, 20, 10, no_lhapdf, dev, ext=ext)


def main(pdfname, mode, no_lhapdf, no_mkl, inter_op, intra_op, no_tex, fname):
    if mode == "generate":
        generate(pdfname, 20, 10)
    if mode == "run":
        run_exp(pdfname, no_lhapdf, no_mkl, inter_op, intra_op)
    if mode == "plot":
        pdfname = "-".join(pdfname.split("/"))
        in_name = f"results_{pdfname}_{20}_{10}"
        texpdf = "PDFFlow" if no_tex else r"\texttt{PDFFlow}"
        args = [
            {
            "ext": "lhapdf", # extension to input file name
            "label": "LHAPDF",
            "color": "blue",
            "marker": "o"
            },
            {
            "ext": "nomkl",
            "label": "%s: no mkl"%texpdf,
            "color": "lime",
            "marker": "^"
            },
            {
            "ext": "0_0",
            "label": "%s: (0,0)"%texpdf,
            "color": "darkslategrey",
            "marker": "^"
            },
            {
            "ext": "1_96",
            "label": "%s: (1,96)"%texpdf,
            "color": "coral",
            "marker": "^"
            },
            {
            "ext": "2_48",
            "label": "%s: (2,48)"%texpdf,
            "color": "darkred",
            "marker": "^"
            }
        ]
        plot(in_name, fname, no_tex, args)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(time() - start)
