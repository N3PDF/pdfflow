"""
    Main pdfflow module
"""
import logging
import collections

import subprocess as sp
import numpy as np

try:
    import lhapdf
except ModuleNotFoundError:
    lhapdf = None

# import configflow before tf to set some tf options
from pdfflow.configflow import DTYPE, DTYPEINT, int_me, izero, fzero, float_me
import tensorflow as tf
from pdfflow.subgrid import Subgrid

# lhapdf gluon code
PID_G = int_me(21)
# expected input shapes to be found in this module
GRID_F = tf.TensorSpec(shape=[None], dtype=DTYPE)
GRID_I = tf.TensorSpec(shape=[None], dtype=DTYPEINT)
# instantiate logger
logger = logging.getLogger(__name__)
# create the Grid namedtuple
GridTuple = collections.namedtuple('Grid', ['x', 'q2', 'flav', 'grid'])


def _load_data(pdf_file):
    """
    Reads pdf from file and retrieves a list of grids
    Each grid is a tuple containing numpy arrays (x,Q2, flavours, pdf)

    Note:
        the input q array in LHAPDF is just q, this functions
        squares the result and q^2 is used everwhere in the code

    Parameters
    ----------
        pdf_file: str
            PDF .dat file

    Returns
    -------
        grids: list(tuple(np.array))
            list of tuples of arrays (x, Q2, flavours, pdf values)
    """
    with open(pdf_file, "r") as pfile:
        n = []
        count = 0
        for line in pfile:
            if "---" in line:
                n += [count]
            count += 1

    grids = []
    for i in range(len(n) - 1):
        x = np.loadtxt(pdf_file, skiprows=(n[i] + 1), max_rows=1)
        q2 = pow(np.loadtxt(pdf_file, skiprows=(n[i] + 2), max_rows=1), 2)
        flav = np.loadtxt(pdf_file, skiprows=(n[i] + 3), max_rows=1)
        grid = np.loadtxt(pdf_file, skiprows=(n[i] + 4), max_rows=(n[i + 1] - n[i] - 4))
        grids += [GridTuple(x, q2, flav, grid)]


    return grids


def mkPDF(fname, dirname=None):
    """ Wrapper to generate a PDF given a PDF name and a directory
    where to find the grid files.

    Parameters
    ----------
        fname: str
            PDF name and member in the format '<set_name>/<set member number>'
        dirname: str
            LHAPDF datadir, if None will try to guess from LHAPDF

    Returns
    -------
        PDF: pdfflow.PDF
            instantiated member of the PDF class
    """
    if dirname is None:
        if lhapdf is None:
            raise ValueError("mkPDF needs a PDF name if lhapdf-python is not installed")
        dirname_raw = sp.run(
            ["lhapdf-config", "--datadir"], capture_output=True, text=True, check=True
        )
        dirname = dirname_raw.stdout.strip()
    return PDF(fname, dirname)


class PDF:
    """
    PDF class exposing the high level pdfflow interfaces:

    Contains
    --------
         xfxQ2: tf.tensor
            Returns a grid for the value of the pdf for each value of (pid, x, q2)
         xfxQ2_allpid: tf.tensor
            Wrapper to return a grid for the value of the pdf for all pids (pid, x, q2)

    Parameters
    ----------
        fname: str
            PDF name and member, must be in the format: '<set_name>/<set member number>'
        dirname: str
            LHAPDF datadir
    """

    def __init__(self, fname, dirname):
        self.dirname = dirname
        fname, member = fname.split("/")
        member = member.zfill(4)

        self.fname = f"{self.dirname}/{fname}/{fname}_{member}.dat"

        logger.info("loading %s", self.fname)
        grids = _load_data(self.fname)
        # [(x,Q2,flav,knots), ...]
        flav = list(map(lambda g: g[2], grids))
        for i in range(len(flav) - 1):
            if not np.all(flav[i] == flav[i + 1]):
                # TODO: should this be an error?
                logger.warning(
                    "Flavor schemes do not match across all the subgrids --> algorithm will break!"
                )

        self.subgrids = [Subgrid(grid, i, len(grids)) for i, grid in enumerate(grids)]

        # By default all subgrids are called with the inner subgrid

        # Look at the flavor_scheme and ensure that it is sorted
        # save the whole thing in case it is not sorted
        flavor_scheme = grids[0].flav
        self.flavor_scheme = int_me(flavor_scheme)

        flavor_scheme[flavor_scheme == PID_G.numpy()] = 0
        if all(np.diff(flavor_scheme) == 1):
            self.flavors_sorted = True
            self.flavor_shift = -flavor_scheme[0]
        else:
            # TODO can't we rely on the PDF flavours to be sorted?
            self.flavors_sorted = False
            self.flavor_shift = 0

    @property
    def q2max(self):
        q2max = self.subgrids[-1].log_q2max
        return np.exp(q2max)

    @property
    def q2min(self):
        q2min = self.subgrids[0].log_q2min
        return np.exp(q2min)

    @tf.function(input_signature=[GRID_I, GRID_F, GRID_F])
    def _xfxQ2(self, u, aa_x, aa_q2):
        """
        Function to interpolate
        Called by xfxQ2
        It divides the computation on the q2 axis in subgrids and sums up
        all the results

        Parameters
        ----------
            u: tf.tensor(int)
                list of PID to compute
            aa_x: tf.tensor(float)
                x-grid for the evaluation of the pdf
            aa_q2: tf.tensor(float)
                q2-grid for the evaluiation of the pdf
        """

        a_x = tf.math.log(aa_x, name="logx")
        a_q2 = tf.math.log(aa_q2, name="logq2")

        size_a = tf.size(a_x, out_type=DTYPEINT)
        size_u = tf.size(u, out_type=DTYPEINT)
        shape = tf.stack([size_a, size_u])

        res = fzero
        for subgrid in self.subgrids:
            res += subgrid(
                u,
                shape,
                a_x,
                a_q2,
            )

        return res

    @tf.function(experimental_relax_shapes=True)
    def xfxQ2(self, pid, arr_x, arr_q2):
        """
        User interface for pdfflow
        It asks pid, x, q2 points

        Parameters
        ----------
            pid: list(int)
                list of PID to be computed
            arr_x: array
                grid on x where to compute the PDF
            arr_q2: array
                grid on q^2 where to compute the PDF
        Returns
        -------
            pdf: tensor
                PDF evaluated in each f(x,q2) for each flavour
        """

        # must feed a mask for flavors to _xfxQ2
        # if pid is None, the mask is set to true everywhere
        # pid must be a list of pids
        if type(pid) == int:
            pid = [pid]

        # Since the user might be asking for a list, let's ensure it is a tensor of ints
        tensor_pid = int_me(pid)

        # same for the a_x and a_q2 arrays
        a_x = float_me(arr_x)
        a_q2 = float_me(arr_q2)

        # And ensure it is unique
        # TODO maybe error if the user ask for the same pid twice or for a non-registered pid?
        upid, user_idx = tf.unique(tensor_pid, out_idx=DTYPEINT)


        # And return the positions in the flavor_scheme array
        # if the flavours are sorted, do it the easy way
        if self.flavors_sorted:
            # Change the LHAPDF gluon number to 0
            upid = tf.where(upid == PID_G, izero, upid)
            pid_idx = self.flavor_shift + upid
        else:
            # Change 0 to the LHAPDF gluon pid: 21
            upid = tf.where(upid == izero, PID_G, upid)
            # TODO maybe it is better to digest the flavor_scheme on initialization and avoid this
            upid = tf.expand_dims(upid, -1)
            pid_idx = tf.cast(
                tf.where(tf.equal(self.flavor_scheme, upid))[:, 1], dtype=DTYPEINT
            )

        # Perform the actual computation
        f_f = self._xfxQ2(pid_idx, a_x, a_q2)

        # Return the values in the order the user asked
        f_f = tf.gather(f_f, user_idx, axis=1)

        result = tf.squeeze(f_f)
        return result

    @tf.function
    def xfxQ2_allpid(self, a_x, a_q2):
        """
        User iterface for pdfflow
        Ask x, q2 points
        Return all flavors
        """
        pid = self.flavor_scheme
        return self.xfxQ2(pid, a_x, a_q2)
