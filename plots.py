import pdfflow
from plot_utils import *
import lhapdf
import pdfflow
import lhapdf
import matplotlib.pyplot as plt
from args import Args











args = Args()
p = pdfflow.mkPDF(args.fname, args.dirname)

for pp in p.subgrids:
    pp.print_summary()


l_pdf = lhapdf.mkPDF(args.fname)


#fixed = 'x'
#v = 2E-9
#plot_slice_pdf(fixed, v, p, l_pdf)


args.xmin = p.subgrids[0].xmin
args.xmax = p.subgrids[0].xmax
args.Q2min = p.subgrids[0].Q2min
args.Q2max = p.subgrids[-1].Q2max

a_x = np.exp(np.random.uniform(np.log(args.xmin), np.log(args.xmax),[args.n_draws,]))
a_Q2 = np.exp(np.random.uniform(np.log(args.Q2min), np.log(args.Q2max),[args.n_draws,]))

plots(args, a_x, a_Q2, p, l_pdf)
test_time(args, p, l_pdf)


#print('Average pdf value: ', float(tf.math.reduce_mean(f)))




