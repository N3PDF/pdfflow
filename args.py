import subprocess as sp

class Args():
    def __init__(self):
        self.fname = 'NNPDF31_nlo_as_0118/0'

        out = sp.run(['lhapdf-config','--prefix'], stdout=sp.PIPE, universal_newlines=True).stdout
        self.dirname = out.strip('\n') + '/share/LHAPDF/'

        self.xmin = 1E-9
        self.xmax = 1
        self.Qmin = 1.65
        self.Qmax = 1E5

        self.Q2min = self.Qmin**2
        self.Q2max = self.Qmax**2

        self.n_draws = 100000

        self.PID = 21

        self.logdir = './logs/'