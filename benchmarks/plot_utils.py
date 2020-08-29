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
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.figsize'] = [7,8]
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['ytick.labelsize'] = 17
    mpl.rcParams['xtick.labelsize'] = 17
    mpl.rcParams['legend.fontsize'] = 18

    t_pdf = []
    t_lha = []
    n = np.linspace(1e5,1e6,2)
    for j in tqdm.tqdm(range(2)):
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

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.1)
    ax = fig.add_subplot(gs[:-1,:])
    ax.errorbar(n,avg_p,yerr=std_p,label=r'\texttt{PDFFlow}')

    ax.errorbar(n,avg_l,yerr=std_l,label=r'LHAPDF')

    ax.title.set_text(r'\texttt{PDFflow} - LHAPDF perfomances')
    #ax.set_xlabel(r'\# points drawn $[\times 10^{5}]$')
    ax.set_ylabel(r'$t [s]$', fontsize=20)

    #ax.set_xlim([1e5,1e6])
    ticks = list(np.linspace(1e5,1e6,10))
    labels = [r'%d'%i for i in range(1,11)]
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))

    ax.legend(frameon=False)

    ax.tick_params(axis='x', direction='in',
                   bottom=True, labelbottom=False,
                   top=True, labeltop=False)
    ax.tick_params(axis='y', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)


    ax = fig.add_subplot(gs[-1,:])
    ax.errorbar(n, (1-avg_p/avg_l)*100,yerr=std_ratio*100)
    #ax.title.set_text(r'Improvements of pdfflow in percentage')
    ax.set_xlabel(r'\# points drawn  $[\times 10^{5}]$',
                  fontsize=20)
    ax.set_ylabel(r'$\displaystyle{\frac{t_{l}-t_{p}}{t_{l}}} \, \%$',
                  fontsize=20)

    #ax.set_xlim([1e5,1e6])
    ticks = list(np.linspace(1e5,1e6,10))
    labels = [r'%d'%i for i in range(1,11)]
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))

    ax.tick_params(axis='x', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False)
    ax.tick_params(axis='y', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)

    plt.savefig('time.pdf', bbox_inches='tight', dpi=200)
    plt.close()
