// This file is part of PDFFlow
#include <stdio.h>
#include "pdfflow/pdfflow.h"

// Some notes about the Fortran interface (to be moved to the docs)
// 
// There are two caveats to take into account when interfacing with Fortran:
// 1 - In fortran all arguments are passed by reference, therefore this file
// does little more than dereference the arguments to pass it through the cffi interface
// 2 - Fortran assumes all functions to have a `_` at the end. This is not true in C
// it is possible to change this behaviour with the  -fno-underscoring gcc flag, but since 
// it is not standard we prefer to provide an interface between the "cnaming" and the "fnaming"

void mkpdf_(const char *fname, const char *dirname) {
   mkpdf(fname, dirname);
}

void alphasq2_(const double *q2, double *alphas) {
   *alphas = alphasq2(*q2);
}

void xfxq2_(const int *pid, const double *x, const double *q2, double *f) {
   *f = xfxq2(*pid, *x, *q2);
}
