// This file is part of PDFFlow
#include <stdio.h>
#include "pdfflow/pdfflow.h"

int main() {
    // load pdf
    mkpdf("NNPDF40_nnlo_as_01180/0", "/usr/share/lhapdf/LHAPDF/");

    // test xfxq2 and alphasq2
    int pid[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    double xs[] = {0.1};
    double q2s[] = {1.65};
    double *xf_vectorized = xfxq2(pid, 11, xs, 1, q2s, 1);
    for (int fl = 0; fl < 11; fl++)
        printf("flv=%d - xfx = %f\n", fl-5, xf_vectorized[fl]);

    double q2_vectorized[] = {1.65, 10.65};
    double* as_vectorized = alphasq2(q2_vectorized, 2);

    printf("alphas(q2=%f) = %f\n", q2_vectorized[0], as_vectorized[0]);
    printf("alphas(q2=%f) = %f\n", q2_vectorized[1], as_vectorized[1]);

    return 0;
}
