// This file is part of PDFFlow
#include <stdio.h>
#include "pdfflow/pdfflow.h"

int main() {
    // load pdf
    mkpdf("NNPDF31_nlo_as_0118/0", "/usr/share/lhapdf/LHAPDF/");

    // test xfxq2 and alphasq2
    const double x = 0.1, q2 = 1.65;
    for (int fl=-5; fl <=5; fl++)
        printf("flv=%d - xfx = %f\n", fl, xfxq2(fl, x, q2));
    printf("alphas(q2=%f) = %f\n", q2, alphasq2(q2));

    return 0;
}
