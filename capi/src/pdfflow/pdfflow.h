/**
 * PDFFlow C API
 */

extern void mkpdf(const char *fname, const char * dirname);

extern double *xfxq2(int *pid, int n, double *x, int m, double *q2, int o);

extern double *alphasq2(double *q2, int n);
