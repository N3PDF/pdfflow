import tensorflow as tf
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
from time import time
import tqdm
import os

float64 = tf.float64
EPS = np.finfo(float).eps


def plots(PID, a_x, a_Q2, p, l_pdf, xmin, xmax, Q2min, Q2max):
    f = p.py_xfxQ2(PID, a_x, a_Q2)
    
    f_lha = f
    if l_pdf is not None:
        f_lha = []
        for i in range(a_x.shape[0]):
            f_lha += [l_pdf.xfxQ2(PID,float(a_x[i]), float(a_Q2[i]))]
        f_lha = np.array(f_lha)

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
    p.py_xfxQ2_allpid(a_x, a_Q2)
    t = time()- start

    start = time()
    if l_pdf is not None:
        f_lha = []
        for i in range(a_x.shape[0]):
            l_pdf.xfxQ2(a_x[i], a_Q2[i])
    tt = time()- start

    return t, tt


def test_time(p, l_pdf, xmin, xmax, Q2min, Q2max):
    #building graph for py_xfxQ2_allpid
    s = time()
    p.py_xfxQ2_allpid([0.5],[4.])
    print('Graph building time: ', (time()-s))

    t_pdf = []
    t_lha = []
    n = np.logspace(5,6,10)
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

    print('n', n)
    print('avg_l', avg_l)
    print('avg_p', avg_p)
    print('rel diff', 1-avg_p/avg_l)


    fig = plt.figure(figsize=(15,10.5))
    ax = fig.add_subplot(121)

    ax.errorbar(n,avg_p,yerr=std_p,label='pdfflow')

    ax.errorbar(n,avg_l,yerr=std_l,label='lhapdf')

    ax.title.set_text('Algorithms working times')
    ax.set_xlabel('# points drawn')
    ax.set_ylabel('t [s]')
    ax.legend()

    ax = fig.add_subplot(222)

    ax.errorbar(n,avg_l-avg_p,yerr=np.sqrt(std_l**2+std_p**2))
    ax.title.set_text('Absolute improvements of pdfflow')
    ax.set_xlabel('# points drawn')
    ax.set_ylabel(r'$t_{lhapdf}-t_{pdfflow}$ [s]')
    ax.set_xscale('log')

    ax = fig.add_subplot(224)

    ax.errorbar(n, (1-avg_p/avg_l)*100,yerr=std_ratio*100)
    ax.title.set_text('Improvements of pdfflow in percentage')
    ax.set_xlabel('# points drawn')
    ax.set_ylabel(r'$(t_{lhapdf}-t_{pdfflow})/t_{lhapdf}$ %')
    ax.set_xscale('log')

    plt.savefig('time.png', bbox_inches='tight')
    plt.close()
