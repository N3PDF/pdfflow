"""
Benchmark script for LHAPDF comparison.
"""
import lhapdf
import argparse
import subprocess as sp
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pdfname",
    "-p",
    default="NNPDF31_nlo_as_0118/0",
    type=str,
    help="The PDF set name/replica number.",
)
parser.add_argument("--n-draws", default=100000, type=int, help="Number of trials.")
parser.add_argument("--pid", default=21, type=int, help="The flavour PID.")
parser.add_argument(
    "--no_lhapdf", action="store_true", help="Don't run lhapdf, only pdfflow"
)
DIRNAME = (
    sp.run(
        ["lhapdf-config", "--datadir"], stdout=sp.PIPE, universal_newlines=True
    ).stdout.strip("\n")
    + "/"
)


def main(pdfname=None, n_draws=10, pid=21, no_lhapdf=False):
    """Testing PDFflow vs LHAPDF performance."""
    import pdfflow.pflow as pdf
    from plot_utils import plots, test_time
    import tensorflow as tf

    p = pdf.mkPDF(pdfname, DIRNAME)

    # for pp in p.subgrids:
    #    pp.print_summary()

    xmin = np.exp(p.subgrids[0].log_xmin)
    xmax = np.exp(p.subgrids[0].log_xmax)
    Q2min = np.sqrt(np.exp(p.subgrids[0].log_q2min))
    Q2max = np.sqrt(np.exp(p.subgrids[-1].log_q2max))
    # a_x = np.logspace(xmin,xmax,n_draws)
    # a_Q2 = np.logspace(Q2min,Q2max,n_draws)

    a_x = np.exp(np.random.uniform(np.log(xmin), np.log(xmax), [n_draws,]))
    print(a_x.min())
    print(a_x.max())
    a_Q2 = np.exp(np.random.uniform(np.log(Q2min), np.log(Q2max), [n_draws,]))

    if no_lhapdf:
        l_pdf = lhapdf.mkPDF(pdfname)
    else:
        l_pdf = None

    print("Printing plots")

    with tf.profiler.experimental.Profile('/media/storageSSD/Academic_Workspace/N3PDF/pdfflow/benchmarks/logdir'):
        plots(pid, a_x, a_Q2, p, l_pdf, xmin, xmax, Q2min, Q2max)
        print("Printing times")
        test_time(p, l_pdf, xmin, xmax, Q2min, Q2max)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time.time()
    main(**args)
    print(time.time() - start)
