"""
    Main pdfflow module
"""
import logging
import collections
import yaml

import subprocess as sp
import numpy as np

try:
    import lhapdf
except ModuleNotFoundError:
    lhapdf = None

# import configflow before tf to set some tf options
from pdfflow.configflow import DTYPE, DTYPEINT, int_me, izero, float_me
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
GridTuple = collections.namedtuple("Grid", ["x", "q2", "flav", "grid"])
AlphaTuple = collections.namedtuple("Alpha", ["q2", "grid"])


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


def _load_alphas(info_file):
    """
    Reads metadata from info file and retrieves a list of alphas subgrids
    Each subgrid is a tuple containing numpy arrays (Q2, alphas)

    Note:
        the input q array in LHAPDF is just q, this functions
        squares the result and q^2 is used everwhere in the code

    Parameters
    ----------
        pdf_file: str
            Metadata .info file

    Returns
    -------
        grids: list(tuple(np.array))
            list of tuples of arrays (Q2, alphas values)
    """
    with open(info_file, "r") as ifile:
        idict = yaml.load(ifile, Loader=yaml.FullLoader)

    alpha_qs = np.array(idict["AlphaS_Qs"])
    alpha_vals = np.array(idict["AlphaS_Vals"])

    grids = []

    EPS = np.finfo(alpha_qs.dtype).eps
    diff = alpha_qs[1:] - alpha_qs[:-1]
    t = np.where(diff < EPS)[0] + 1

    splits_qs = np.split(alpha_qs ** 2, t)
    splits_vals = np.split(alpha_vals, t)

    for q, v in zip(splits_qs, splits_vals):
        grids.append(AlphaTuple(q, v))

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

    def __init__(self, fname, dirname, compilable=True):
        if not compilable:
            logger.warning("Running pdfflow in eager mode")
            logger.warning("Setting eager mode will affect all of TF")
            tf.config.experimental_run_functions_eagerly(True)

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

        # now load metadata from info file
        logger.info("Enabling computation of alpha")
        self.fname = f"{self.dirname}/{fname}/{fname}.info"

        logger.info("loading %s", self.fname)
        alpha_grids = _load_alphas(self.fname)
        self.alphas_subgrids = [
            Subgrid(grid, i, len(grids), alpha_s=True) for i, grid in enumerate(alpha_grids)
        ]

    @property
    def q2max(self):
        """ Upper boundary in q2 of the grid """
        q2max = self.subgrids[-1].log_q2max
        return np.exp(q2max)

    @property
    def q2min(self):
        """ Lower boundary in q2 of the grid """
        q2min = self.subgrids[0].log_q2min
        return np.exp(q2min)

    @tf.function(input_signature=[GRID_I, GRID_F, GRID_F])
    def _xfxQ2(self, u, arr_x, arr_q2):
        """
        Function to interpolate
        Called by xfxQ2
        It divides the computation on the q2 axis in subgrids and sums up
        all the results

        Parameters
        ----------
            u: tf.tensor(int)
                list of PID to compute
            arr_x: tf.tensor(float)
                x-grid for the evaluation of the pdf
            arr_q2: tf.tensor(float)
                q2-grid for the evaluation of the pdf
        """
        print('retracing _xfxQ2')
        a_x = tf.math.log(arr_x, name="logx")
        a_q2 = tf.math.log(arr_q2, name="logq2")

        size_a = tf.size(a_x, out_type=DTYPEINT)
        size_u = tf.size(u, out_type=DTYPEINT)
        shape = tf.stack([size_a, size_u])

        res = tf.zeros(shape, dtype=DTYPE)
        for subgrid in self.subgrids:
            res += subgrid(shape, a_q2, pids=u, arr_x=a_x)

        return res

    @tf.function(input_signature=[GRID_I, GRID_F, GRID_F])
    def xfxQ2(self, pid, a_x, a_q2):
        """
        User interface for pdfflow when called with
        tensorflow tensors
        It asks pid, x, q2 points

        Parameters
        ----------
            pid: tf.tensor, dtype=int
                list of PID to be computed
            a_x: tf.tensor, dtype=float
                grid on x where to compute the PDF
            a_q2: tf.tensor, dtype=float
                grid on q^2 where to compute the PDF
        Returns
        -------
            pdf: tensor
                PDF evaluated in each f(x,q2) for each flavour
        """
        # Parse the input
        # this function assumes the user is asking for a tensor of pids
        # TODO if the user is to do non-tf stuff print a warning and direct
        # them to use the python version of the functions
        # must feed a mask for flavors to _xfxQ2
        # cast down if necessary the type of the pid
        # pid = int_me(pid)

        # same for the a_x and a_q2 arrays
        # a_x = float_me(arr_x)
        # a_q2 = float_me(arr_q2)

        # And ensure it is unique
        # TODO maybe error if the user ask for the same pid twice or for a non-registered pid?
        print('retracing xfxQ2')
        upid, user_idx = tf.unique(pid, out_idx=DTYPEINT)

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
            pid_idx = tf.cast(tf.where(tf.equal(self.flavor_scheme, upid))[:, 1], dtype=DTYPEINT)

        # Perform the actual computation
        f_f = self._xfxQ2(pid_idx, a_x, a_q2)

        # Return the values in the order the user asked
        f_f = tf.gather(f_f, user_idx, axis=1)

        result = tf.squeeze(f_f)
        return result

    @tf.function(input_signature=[GRID_F, GRID_F])
    def xfxQ2_allpid(self, a_x, a_q2):
        """
        User interface for pdfflow when called with
        tensorflow tensors
        It asks x, q2 points
        returns all flavours

        Parameters
        ----------
            a_x: tf.tensor, dtype=float
                grid on x where to compute the PDF
            a_q2: tf.tensor, dtype=float
                grid on q^2 where to compute the PDF
        Returns
        -------
            pdf: tensor
                PDF evaluated in each f(x,q2) for each flavour
        """
        print('retracing xfxQ2 all pid')
        pid = self.flavor_scheme
        return self.xfxQ2(pid, a_x, a_q2)

    # Python version of the above functions with the correct casting to tf
    def py_xfxQ2_allpid(self, arr_x, arr_q2):
        """
        Python interface for pdfflow
        The input gets converted to the right
        tf type before calling the corresponding functions
        Returns all flavours

        Parameters
        ----------
            arr_x: np.array
                grid on x where to compute the PDF
            arr_q2: np.array
                grid on q^2 where to compute the PDF
        Returns
        -------
            pdf: tensor
                PDF evaluated in each f(x,q2) for each flavour
        """
        a_x = float_me(arr_x)
        a_q2 = float_me(arr_q2)
        return self.xfxQ2_allpid(a_x, a_q2)

    def py_xfxQ2(self, pid, arr_x, arr_q2):
        """
        Python interface for pdfflow
        The input gets converted to the right
        tf type before calling the corresponding functions

        Parameters
        ----------
            pid: list(int)
                list of PID to be computed
            arr_x: np.array
                grid on x where to compute the PDF
            arr_q2: np.array
                grid on q^2 where to compute the PDF
        Returns
        -------
            pdf: tensor
                PDF evaluated in each f(x,q2) for each flavour
        """
        # if pid is None, the mask is set to true everywhere
        if pid is None:
            return self.py_xfxQ2_allpid(arr_x, arr_q2)

        tensor_pid = tf.reshape(int_me(pid), (-1,))
        a_x = float_me(arr_x)
        a_q2 = float_me(arr_q2)
        return self.xfxQ2(tensor_pid, a_x, a_q2)

    @tf.function(input_signature=[GRID_F])
    def _alphasQ2(self, arr_q2):
        """
        Function to interpolate
        Called by alphasQ2
        It divides the computation on the q2 axis in subgrids and sums up
        all the results

        Parameters
        ----------
            arr_q2: tf.tensor(float)
                q2-grid for the evaluation of alphas
        """
        a_q2 = tf.math.log(arr_q2, name="logq2")

        shape = tf.size(a_q2, out_type=DTYPEINT)

        res = tf.zeros(shape, dtype=DTYPE)
        for subgrid in self.alphas_subgrids:
            res += subgrid(shape, a_q2)

        return res

    @tf.function(input_signature=[GRID_F])
    def alphasQ2(self, a_q2):
        """
        User interface for pdfflow alphas interpolation when called with
        tensorflow tensors
        It asks q2 points

        Parameters
        ----------
            a_q2: tf.tensor, dtype=float
                grid on q^2 where to compute alphas
        Returns
        -------
            alphas: tensor
                alphas evaluated in each q^2 query point
        """
        # Parse the input
        # a_q2 = float_me(arr_q2)

        # Perform the actual computation
        return self._alphasQ2(a_q2)

    @tf.function(input_signature=[GRID_F])
    def alphasQ(self, a_q):
        """
        User interface for pdfflow alphas interpolation when called with
        tensorflow tensors
        It asks q points

        Parameters
        ----------
            a_q: tf.tensor, dtype=float
                grid on q where to compute alphas
        Returns
        -------
            alphas: tensor
                alphas evaluated in each q query point
        """
        # Parse the input
        # print('trace')
        # a_q = float_me(arr_q)
        a_q2 = a_q ** 2

        # Perform the actual computation
        return self._alphasQ2(a_q2)

    def py_alphasQ2(self, arr_q2):
        """
        User interface for pdfflow alphas interpolation when called with
        tensorflow tensors
        It asks q^2 points

        Parameters
        ----------
            arr_q: tf.tensor, dtype=float
                grid on q^2 where to compute alphas
        Returns
        -------
            alphas: tensor
                alphas evaluated in each q^2 query point
        """
        # Parse the input
        a_q2 = float_me(arr_q2)

        # Perform the actual computation
        return self._alphasQ2(a_q2)

    def py_alphasQ(self, arr_q):
        """
        User interface for pdfflow alphas interpolation when called with
        tensorflow tensors
        It asks q points

        Parameters
        ----------
            arr_q: tf.tensor, dtype=float
                grid on q where to compute alphas
        Returns
        -------
            alphas: tensor
                alphas evaluated in each q query point
        """
        # Parse the input
        a_q = float_me(arr_q)

        # Perform the actual computation
        return self.alphasQ(a_q)
