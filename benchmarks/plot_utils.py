import tensorflow as tf
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
from time import time
import tqdm
import os

float64 = tf.float64
EPS = np.finfo(float).eps
'''
def slice_test(fixed,v,p,l_pdf):
'''
    #Prints xf versus x or Q2 depending on the fixed flag
    #fixed: string, either x or Q2

'''
    if 'Q2' in fixed:
        a_x = np.exp(np.random.uniform(np.log(p.pdf[0].xmin), np.log(p.pdf[-1].xmax),[10000,]))
        a_Q2 = np.array([float(v) for i in range(10000)])
    if 'x' in fixed:
        a_Q2 = np.exp(np.random.uniform(np.log(p.pdf[0].Q2min), np.log(p.pdf[-1].Q2max),[10000,]))
        a_x = np.array([float(v) for i in range(10000)])

    o_x,o_Q2, f = p.xfpxQ2(-4, tf.constant(a_x, dtype=float64), tf.constant(a_Q2,dtype=float64))

    l_f = []
    for i in range(a_x.shape[0]):
        l_f += [l_pdf.xfxQ2(-4, a_x[i], a_Q2[i])]
    l_f = np.array(l_f)

    if 'Q2' in fixed:
        return o_x, f, a_x, l_f
    if 'x' in fixed:
        return o_Q2, f, a_Q2, l_f




def plot_slice_pdf(fixed, v, p, l_pdf):

    a,b,c,d = slice_test(fixed,v,p,l_pdf)
    fig = plt.figure(figsize=(15,10.5))
    fig.suptitle('xf(x, Q) plot with fixed at %s = %.2e'%(fixed, v), fontsize=16)
    ax = fig.add_subplot(211)
    ax.plot(a,b,'.')
    if 'Q2' in fixed:
        ax.set_xlim((1e-10,1.2))
        ax.set_xlabel('x')
    if 'x' in fixed:
        ax.set_xlabel('Q^2 [GeV^2]')
    ax.set_ylabel('xf (x, Q^2)')
    ax.set_xscale('log')
    ax.title.set_text('pdfflow')
    ax = fig.add_subplot(212)
    ax.plot(c,d, 'g.', label='lhapdf', lw=3)
    ax.plot(a,b, 'r.', label='pdfflow', alpha=0.2, lw=1.5)
    ax.legend()
    ax.title.set_text('lhapdf')
    ax.set_ylabel('xf (x, Q^2)')
    ax.set_xscale('log')
    if 'Q2' in fixed:
        ax.set_xlim((1e-10,1.2))
        ax.set_xlabel('x')
        plt.savefig('xf_x.png')
    if 'x' in fixed:
        ax.set_xlabel('Q^2 [GeV^2]')
        plt.savefig('xf_Q2.png')
    plt.close()




def sort_arrays(a,b,f):
    a = np.array(a)
    b = np.array(b)
    f = np.array(f)
    ind = np.argsort(b)

    aa = np.take_along_axis(a,ind,0)
    bb = np.take_along_axis(b,ind,0)
    ff = np.take_along_axis(f,ind,0)

    ind = np.argsort(aa)

    return np.take_along_axis(aa,ind,0), np.take_along_axis(bb,ind,0), np.take_along_axis(ff,ind,0)
'''



def plots(PID, a_x, a_Q2, p, l_pdf, xmin, xmax, Q2min, Q2max):
    #writer = tf.summary.create_file_writer(args.logdir)
    #tf.summary.trace_on(graph=True, profiler=True)
    f = p.xfxQ2(a_x, a_Q2, PID)
    #with writer.as_default():
    #    tf.summary.trace_export(
    #        name='xfxQ2_trace',
    #        step=0,
    #        profiler_outdir=args.logdir)

    #x,q,f = sort_arrays(x,q,f)

    f_lha = []
    for i in range(a_x.shape[0]):
        f_lha += [l_pdf.xfxQ2(PID,float(a_x[i]), float(a_Q2[i]))]
    f_lha = np.array(f_lha)
    #_, _, f_sort = sort_arrays(a_x, a_Q2, f_lha)

    fig = plt.figure(figsize=(20,16))

    ax = fig.add_subplot(311)
    z = ax.scatter(a_x, a_Q2, c=f)
    ax.title.set_text('pdfflow algorithm')
    ax.set_xlabel('x')
    ax.set_ylabel('Q^2 [Gev^2]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((Q2min, Q2max))
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(312)
    z = ax.scatter(a_x, a_Q2, c=f_lha)
    ax.title.set_text('lhapdf algorithm')
    ax.set_xlabel('x')
    ax.set_ylabel('Q^2 [Gev^2]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((Q2min, Q2max))
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(313)
    ax.title.set_text('Percentage Absolute ratio')
    ax.set_xlabel('x')
    ax.set_ylabel('Q^2 [Gev^2]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((Q2min, Q2max))
    z = ax.scatter(a_x,a_Q2, c=np.abs(f-f_lha)/(np.abs(f_lha)+EPS))
    fig.colorbar(z, ax=ax)

    plt.savefig('interpolation.png')

    plt.close()



def test(n_draws, p, l_pdf, xmin, xmax, Q2min, Q2max):
    a_x = np.random.uniform(xmin, xmax,[n_draws,])
    a_Q2 = np.exp(np.random.uniform(np.log(Q2min), np.log(Q2max),[n_draws,]))

    start = time()
    p.xfxQ2(tf.constant(a_x, dtype=float64), tf.constant(a_Q2,dtype=float64))
    t = time()- start

    start = time()
    f_lha = []
    for i in range(a_x.shape[0]):
        l_pdf.xfxQ2(float(a_x[i]), float(a_Q2[i]))
    tt = time()- start

    return t, tt

def test_time(p, l_pdf, xmin, xmax, Q2min, Q2max):

    t_pdf = []
    t_lha = []
    n = np.logspace(5,5.8,10)
    for j in tqdm.tqdm(range(10)):
        t = []
        tt = []
        for i in tqdm.tqdm(n):
            t_, tt_ =  test(int(i), p, l_pdf, xmin, xmax, Q2min, Q2max)
            t += [t_]
            tt += [tt_]
        t_pdf += [t]
        t_lha += [tt]

    t_pdf = np.stack(t_pdf)
    t_lha = np.stack(t_lha)

    avg_l = t_lha.mean(0)
    avg_p = t_pdf.mean(0)
    std_l = t_lha.std(0)
    std_p = t_pdf.std(0)
    std_ratio = np.sqrt((std_p/avg_l)**2 + (avg_p*std_l/(avg_l)**2)**2)


    fig = plt.figure(figsize=(15,10.5))
    ax = fig.add_subplot(121)

    #ax.plot(n, avg_p,label='pdfflow')
    ax.errorbar(n,avg_p,yerr=std_p,label='pdfflow')

    #ax.plot(n,avg_l,label='lhapdf')
    ax.errorbar(n,avg_l,yerr=std_l,label='lhapdf')

    ax.title.set_text('Algorithms working times')
    ax.set_xlabel('# points drawn')
    ax.set_ylabel('t [s]')
    ax.legend()

    ax = fig.add_subplot(222)

    #ax.plot(n,np.abs(avg_l-avg_p))
    ax.errorbar(n,np.abs(avg_l-avg_p),yerr=np.sqrt(std_l**2+std_p**2))
    ax.title.set_text('Absolute improvements of pdfflow')
    ax.set_xlabel('# points drawn')
    ax.set_ylabel('t [s]')
    ax.set_xscale('log')

    ax = fig.add_subplot(224)

    #ax.plot(n,np.abs((1-avg_p)/avg_l)*100)
    ax.errorbar(n,np.abs((1-avg_p)/avg_l)*100,yerr=std_ratio*100)
    ax.title.set_text('Improvements of pdfflow in percentage')
    ax.set_xlabel('# points drawn')
    ax.set_ylabel('%')
    ax.set_xscale('log')

    plt.savefig('time.png')
    plt.close()