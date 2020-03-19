"""
Benchmark script for LHAPDF comparison.
"""
import lhapdf
import argparse
import subprocess as sp
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--pdfname", "-p", default="NNPDF31_nlo_as_0118/0", type=str, help='The PDF set name/replica number.')
parser.add_argument("--n-draws", default=100000, help="Number of trials.")
parser.add_argument("--pid", default=21, type=int, help='The flavour PID.')
DIRNAME = sp.run(['lhapdf-config','--datadir'], stdout=sp.PIPE, universal_newlines=True).stdout.strip('\n') + '/'


def main(pdfname, n_draws, pid):
    """Testing PDFflow vs LHAPDF performance."""
    import pdfflow.pflow as pdf
    from plot_utils import plots, test_time

    p = pdf.mkPDF(pdfname, DIRNAME)

    for pp in p.subgrids:
        pp.print_summary()

    l_pdf = lhapdf.mkPDF(pdfname)

    xmin = p.subgrids[0].xmin
    xmax = p.subgrids[0].xmax
    Q2min = p.subgrids[0].Q2min
    Q2max = p.subgrids[-1].Q2max

    a_x = np.exp(np.random.uniform(np.log(xmin), np.log(xmax),[n_draws,]))
    a_Q2 = np.exp(np.random.uniform(np.log(Q2min), np.log(Q2max),[n_draws,]))

    print('Printing plots')
    plots(pid, a_x, a_Q2, p, l_pdf, xmin, xmax, Q2min, Q2max)
    print('Printing times')
    test_time(p, l_pdf, xmin, xmax, Q2min, Q2max)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start=time.time()
    main(**args)
    print(time.time()-start)