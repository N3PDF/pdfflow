import tensorflow as tf
import re
import numpy as np
from pdfflow.configflow import DTYPE, DTYPEINT, int_me, ione, izero
from pdfflow.subgrid import Subgrid
from pdfflow.functions import inner_subgrid
from pdfflow.functions import first_subgrid
from pdfflow.functions import last_subgrid

PID_G = int_me(21)

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
        self.flavor_scheme = tf.cast(self.subgrids[0].flav, dtype=DTYPEINT)
        self.subgrids[-1].flag = -ione
        self.subgrids[0].flag = izero

    def _xfxQ2(self, u, aa_x, aa_q2):
        """
        Function to interpolate
        Called by xfxQ2
        It divides the computation on the q2 axis in subgrids and sums up
        all the results
        """

        a_x = tf.cast(tf.math.log(aa_x, name="logx"), DTYPE)
        a_q2 = tf.cast(tf.math.log(aa_q2, name="logq2"), DTYPE)

        size_a = tf.size(a_x, out_type=DTYPEINT)
        size_u = tf.size(u, out_type=DTYPEINT)
        shape = tf.stack([size_a, size_u])

        res = tf.zeros(shape, dtype=DTYPE)

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

    @tf.function
    def xfxQ2(self, pid, a_x, a_q2):
        """
        User interface for pdfflow
        It asks pid, x, q2 points
        """

        # must feed a mask for flavors to _xfxQ2
        # if pid is None, the mask is set to true everywhere
        # pid must be a list of pids
        if type(pid) == int:
            pid = [pid]

        # Since the user might be asking for a list, let's ensure it is a tensor of ints
        tensor_pid = int_me(pid)

        # And ensure it is unique
        # TODO maybe error if the user ask for the same pid twice or for a non-registered pid?
        upid, user_idx = tf.unique(tensor_pid, out_idx=DTYPEINT)

        # Change 0 to the LHAPDF gluon pid: 21
        upid = tf.where(upid==izero, PID_G, upid)
        # And return the positions in the flavor_scheme array
        # TODO maybe it is better to digest the flavor_scheme on initialization and avoid this
        upid = tf.expand_dims(upid,-1)
        pid_idx = tf.cast(tf.where(tf.equal(self.flavor_scheme, upid))[:, 1], dtype=DTYPEINT)

        # Perform the actual computation
        f_f = self._xfxQ2(pid_idx, a_x, a_q2)

        # Return the values in the order the user asked
        f_f = tf.gather(f_f, user_idx, axis=1)

        return tf.squeeze(f_f)

    def xfxQ2_allpid(self, a_x, a_q2):
        """
        User iterface for pdfflow
        Ask x, q2 points
        Return all flavors
        """
        pid = self.flavor_scheme
        return self.xfxQ2(pid, a_x, a_q2)
