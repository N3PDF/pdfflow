class Args():
    def __init__(self):
        self.fname = 'NNPDF31_nlo_as_0118/0'
        self.dirname = '/home/marco/PhD/unimi/PDFlow/local/share/LHAPDF/'#'./local/share/LHAPDF/'

        self.xmin = 1E-9
        self.xmax = 1
        self.Qmin = 1.65
        self.Qmax = 1E5

        self.Q2min = self.Qmin**2
        self.Q2max = self.Qmax**2

        self.n_draws = 100000

        self.PID = 21