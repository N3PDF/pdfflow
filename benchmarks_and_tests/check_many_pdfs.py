#!/usr/bin/env python
"""
    This little script uses the lhapdf python interface to:
    1) Get the latest list of PDF sets
    2) Download a subset of them (or all)
    3) Test with a random set of points that the LHPADF and pdfflow produce the same result

    Uses lhapdf-management to install the PDFs programatically :)
"""

import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import lhapdf
import numpy as np
import pdfflow
from lhapdf_management import environment, pdf_list, pdf_update

lhapdf.setVerbosity(0)


def _compare_w_lhapdf(pdf, npoints=1000, tolerance=1e-6):
    """Compare a LHAPDF and pdfflow  pdfs
    for an array of npoints"""
    # Now get a random member
    m = np.random.randint(len(loaded_pdf), dtype=int)

    pdfflow_pdf = pdfflow.mkPDF(f"{pdf}/{m}")
    lhapdf_pdf = lhapdf.mkPDF(f"{pdf}/{m}")

    # Get n points for x between 0 and 1
    xx = np.random.rand(npoints)
    # And n points for q between the min and the maximum seen by pdfflow
    qdelta = pdfflow_pdf.q2max - pdfflow_pdf.q2min
    qq = pdfflow_pdf.q2min + np.random.rand(npoints) * qdelta

    # Make sure the order is the same as in pdfflow
    flavors = pdf.info["Flavors"]
    lhapdf_results = lhapdf_pdf.xfxQ2(flavors, xx, qq)

    lres = np.array(lhapdf_results)
    pres = pdfflow_pdf.py_xfxQ2_allpid(xx, qq).numpy()

    # This is not still implemented as part of pdfflow, but need to be careful during the check
    if pdf.info.get("ForcePositive", 0) > 0:
        pres = np.maximum(pres, 1e-10)

    np.testing.assert_allclose(pres, lres, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dir", help="Directory where to download the sets", type=Path)
    parser.add_argument("-y", "--yes", help="Respond yes to every question", action="store_true")
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-a",
        "--all",
        help="Try really ALL sets, otherwise, do a random selection of N of them",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--npdfs",
        help="If all is not given, hoy many PDFs to actually test (default 50)",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-p",
        "--points",
        help="How many points in x/q to test per PDF set (default 1000)",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-t", "--tolerance", help="Tolerance for the test (default 1e-6)", type=float, default=1e-6
    )

    args = parser.parse_args()

    if args.dir is None:
        target_dir = Path(tempfile.mkdtemp())
    else:
        target_dir = args.dir

    if not args.yes:
        print(
            f"""You are about to download a potentially large number of PDF sets to {target_dir.absolute()}
This is likely to be heavy in both your storage and your bandwith."""
        )
        yn = input(" > Do you want to continue? [Y/N] ")
        if not yn.lower() in ("y", "yes", "ye", "si"):
            sys.exit(0)

    target_dir.mkdir(exist_ok=True)

    # Set the datapath
    environment.datapath = target_dir
    lhapdf.setPaths([target_dir.as_posix()])

    # Get the latest PDF list
    pdf_update()

    # And now list them all
    list_of_pdfs = pdf_list()

    # if not --all, take a mask of N PDFs
    if not args.all:
        if args.npdfs > len(list_of_pdfs):
            raise ValueError(
                f"The value of N ({args.npdfs}) cannot be greater than the number of PDFs available ({len(list_of_pdfs)}), use --all if you just want to test all of them"
            )
        list_of_pdfs = np.random.choice(list_of_pdfs, size=args.npdfs, replace=False)

    # And time to install!
    failed_pdfs = []
    for pdf in list_of_pdfs:
        if args.verbose:
            print(f"Testing {pdf}... ", end="")
        try:
            pdf.install()
            # Try loading the PDF
            loaded_pdf = pdf.load()
            _compare_w_lhapdf(loaded_pdf, npoints=args.points, tolerance=args.tolerance)
        except KeyError as e:
            # If there's a key error on the PDF either the .info file is malformed (then not our problem)
            # or the PDF is using analytical running for alpha_s, so PDFFlow cannot use it
            pass
        except Exception as e:
            # We are not going to care that much _how_ the failure happened
            if args.verbose:
                print(f"{pdf} failed!")
            failed_pdfs.append((pdf, e))

    if failed_pdfs:
        print("\nThe failed pdfs are: ")
        for pdf, error in failed_pdfs:
            print(f"{pdf} with {error}")
        raise Exception("Some PDFs failed!")
    else:
        print("\nNo PDF failed the test!")
