import tensorflow as tf
import re
import numpy as np
from pdfflow.subgrid import Subgrid
from pdfflow.functions import inner_subgrid
from pdfflow.functions import first_subgrid
from pdfflow.functions import last_subgrid
from pdfflow.interpolations import float64
from pdfflow.interpolations import int64

def load_Data(fname):
    # Reads pdf from file and retrieves a list of grids
    # Each grid is a tuple containing numpy arrays (x,Q2, flavours, pdf)
    f = open(fname)

    n = []
    count = 0
    for line in f:
        if re.match("---", line):
            n += [count]

        count += 1
    f.close()

    grids = []
    for i in range(len(n) - 1):
        x = np.loadtxt(fname, skiprows=(n[i] + 1), max_rows=1)
        q2 = np.loadtxt(fname, skiprows=(n[i] + 2), max_rows=1)
        flav = np.loadtxt(fname, skiprows=(n[i] + 3), max_rows=1)
        grid = np.loadtxt(fname, skiprows=(n[i] + 4), max_rows=(n[i + 1] - n[i] - 4))

        grids += [(x, q2, flav, grid)]

    return grids

class mkPDF:
    def __init__(self, fname, dirname="./local/share/LHAPDF/"):
        """
        fname: string, must be in the format: '<set_name>/<set member number>'
        """
        self.dirname = dirname
        f = fname.split("/")

        self.fname = self.dirname + "%s/%s_%s.dat" % (f[0], f[0], f[1].zfill(4))

        print("pdfflow loading " + self.fname)
        grids = load_Data(self.fname)
        # [(x,Q2,flav,knots), ...]
        flav = list(map(lambda g: g[2], grids))
        for i in range(len(flav) - 1):
            if not np.all(flav[i] == flav[i + 1]):
                print("Flavor schemes do not match across\
                      all the subgrids ---> algorithm will break !")

        self.subgrids = list(map(Subgrid, grids))
        self.flavor_scheme = tf.cast(self.subgrids[0].flav, dtype=tf.int64)
        self.subgrids[-1].flag = tf.constant(-1, dtype=int64)
        self.subgrids[0].flag = tf.constant(0, dtype=int64)

    def _xfxQ2(self, u, aa_x, aa_q2):

        a_x = tf.cast(tf.math.log(aa_x, name="logx"), float64)
        a_q2 = tf.cast(tf.math.log(aa_q2, name="logq2"), float64)

        size = tf.shape(a_x)
        shape = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)

        res = tf.zeros(shape, dtype=float64)
        
        res += first_subgrid(u, a_x, a_q2,
                             self.subgrids[0].log_xmin,
                             self.subgrids[0].log_xmax,
                             self.subgrids[0].padded_x,
                             self.subgrids[0].s_x,
                             self.subgrids[0].log_q2min,
                             self.subgrids[0].log_q2max,
                             self.subgrids[0].padded_q2,
                             self.subgrids[0].s_q2,
                             self.subgrids[0].padded_grid,
                             shape)
        
        for s in self.subgrids[1:-1]:
            res += inner_subgrid(u, a_x, a_q2,
                                 s.log_xmin, s.log_xmax, s.padded_x,
                                 s.log_q2min, s.log_q2max, s.padded_q2,
                                 s.padded_grid,
                                 shape)

        res += last_subgrid(u, a_x, a_q2,
                            self.subgrids[-1].log_xmin,
                            self.subgrids[-1].log_xmax,
                            self.subgrids[-1].padded_x,
                            self.subgrids[-1].s_x,
                            self.subgrids[-1].log_q2min,
                            self.subgrids[-1].log_q2max,
                            self.subgrids[-1].padded_q2,
                            self.subgrids[-1].s_q2,
                            self.subgrids[-1].padded_grid,
                            shape)
        
        return res

    def xfxQ2(self, PID, a_x, a_q2):

        # must feed a mask for flavors to _xfxQ2
        # if PID is None, the mask is set to true everywhere
        # PID must be a list of PIDs
        if type(PID) == int:
            PID = [PID]

        PID = tf.expand_dims(tf.constant(PID, dtype=int64), -1)
        PID = tf.where(PID==0, 21, PID)
        idx = tf.where(tf.equal(self.flavor_scheme, PID))[:, 1]
        u, i = tf.unique(idx)

        f_f = self._xfxQ2(u, a_x, a_q2)
        f_f = tf.gather(f_f, i, axis=1)

        return tf.squeeze(f_f)

    def xfxQ2_allpid(self, a_x, a_q2):
        # return all the flavors
        PID = self.flavor_scheme
        return self.xfxQ2(PID, a_x, a_q2)
