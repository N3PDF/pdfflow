"""
    Main pdfflow module

    Example
    -------
    >>> import pdfflow
    >>> pdf = pdfflow.mkPDFs("NNPDF31_nnlo_as_0118", members=[0,3])
    >>> pdf.trace()
    >>> pdf.py_xfxQ2([3,2], [0.3, 0.5], [1.5, 120.4])
"""
import logging
import collections
import yaml
from pathlib import Path

import subprocess as sp
import numpy as np

import os, sys

from lhapdf_management.pdfsets import PDF as LHA_PDF

# import configflow before tf to set some tf options
from pdfflow.configflow import DTYPE, DTYPEINT, int_me, izero, float_me, find_pdf_path
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


def _load_alphas(info_file):
    """
    Reads metadata from info file and retrieves a list of alphas subgrids
    Each subgrid is a tuple containing numpy arrays (Q2, alphas)

    Note:
        the input q array in LHAPDF is just q, this functions
        squares the result and q^2 is used everwhere in the code

    Parameters
    ----------
        info_file: dict
            Dictionary containg the .info file information

    Returns
    -------
        grids: list(tuple(np.array))
            list of tuples of arrays (Q2, alphas values)
    """
    alpha_qs = np.array(info_file["AlphaS_Qs"])
    alpha_vals = np.array(info_file["AlphaS_Vals"])

    grids = []

    EPS = np.finfo(alpha_qs.dtype).eps
    diff = alpha_qs[1:] - alpha_qs[:-1]
    t = np.where(diff < EPS)[0] + 1

    splits_qs = np.split(alpha_qs ** 2, t)
    splits_vals = np.split(alpha_vals, t)

    for q, v in zip(splits_qs, splits_vals):
        grids.append(AlphaTuple(q, v))

    return grids


def mkPDFs(fname, members=None, dirname=None):
    """Wrapper to generate a multimember PDF
    Needs a name and a directory where to find the grid files.

    Parameters
    ----------
        fname: str
            PDF name and member in the format '<set_name>'
        members: list(int)
            List of members to load
        dirname: str
            LHAPDF datadir, if None will try to guess from LHAPDF

    Returns
    -------
        PDF: pdfflow.PDF
            instantiated member of the PDF class
    """
    if dirname is None:
        try:
            dirname = find_pdf_path(fname)
        except ValueError as e:
            raise ValueError(f"mkPDFs need a directory") from e
    return PDF(dirname, fname, members)


def mkPDF(fname, dirname=None):
    """Wrapper to generate a PDF given a PDF name and a directory
    where to find the grid files.

    Parameters
    ----------
        fname: str
            PDF name and member in the format '<set_name>/<set member number>'
            If the set number is not given, assume member 0
        dirname: str
            LHAPDF datadir, if None will try to guess from LHAPDF

    Returns
    -------
        PDF: pdfflow.PDF
            instantiated member of the PDF class
    """
    try:
        fname_sp, member = fname.split("/")
        member = int(member)
    except ValueError:
        fname_sp = fname
        member = 0
    return mkPDFs(fname_sp, [member], dirname=dirname)


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
        dirname: str
            LHAPDF datadir
        fname: str
            PDF name must be in the format: '<set_name>'
        members: list(int)
            list of integer with the members to be level
    """

    def __init__(self, dirname, fname, members, compilable=True):
        if not compilable:
            logger.warning("Running pdfflow in eager mode")
            logger.warning("Setting eager mode will affect all of TF")
            tf.config.experimental_run_functions_eagerly(True)

        self.dirname = dirname
        self.fname = fname
        self.grids = []
        lhapdf_pdf = LHA_PDF(Path(self.dirname) / fname)
        self.info = lhapdf_pdf.info

        if members is None:
            total_members = self.info.get("NumMembers", 1)
            members = range(total_members)

        if len(members) == 1:
            logger.info("Loading member %d from %s", members[0], self.fname)
        else:
            logger.info("Loading %d members from %s", len(members), self.fname)

        for member_int in members:
            grids = lhapdf_pdf.get_member_grids(member_int)
            subgrids = [Subgrid(grid, i, len(grids)) for i, grid in enumerate(grids)]
            self.grids.append(subgrids)
        self.members = members

        # Get the flavour scheme from the info file
        flavor_scheme = np.array(self.info.get("Flavors", None))
        if flavor_scheme is None:
            # fallback to getting the scheme from the first grid, as all grids should have the same number of flavours
            # if there's a failure here it should be the grid fault so no need to check from our side?
            flavor_scheme = grids[0].flav

        # Look at the flavor_scheme and ensure that it is sorted
        # save the whole thing in case it is not sorted
        self.flavor_scheme = int_me(flavor_scheme)

        flavor_scheme[flavor_scheme == PID_G.numpy()] = 0
        if all(np.diff(flavor_scheme) == 1):
            self.flavors_sorted = True
            self.flavor_shift = -flavor_scheme[0]
        else:
            # TODO can't we rely on the PDF flavours to be sorted?
            self.flavors_sorted = False
            self.flavor_shift = 0

        # Finalize by loading the alpha information form the .info file
        alpha_grids = _load_alphas(self.info)
        self.alphas_subgrids = [
            Subgrid(grid, i, len(grids), alpha_s=True) for i, grid in enumerate(alpha_grids)
        ]

    @property
    def q2max(self):
        """Upper boundary in q2 of the first grid"""
        q2max = self.grids[0][-1].log_q2max
        return np.exp(q2max)

    @property
    def q2min(self):
        """Lower boundary in q2 of the first grid"""
        q2min = self.grids[0][0].log_q2min
        return np.exp(q2min)

    @property
    def nmembers(self):
        """Number of members for this PDF"""
        return len(self.members)

    @property
    def active_members(self):
        """List of all member files"""
        member_list = []
        for member_int in self.members:
            member = str(member_int).zfill(4)
            member_list.append(f"{self.fname}_{member}.dat")
        return member_list

    @tf.function(input_signature=[GRID_I, GRID_F, GRID_F])
    def _xfxQ2(self, u, arr_x, arr_q2):
        """
        Function to interpolate
        Called by xfxQ2
        It divides the computation on the q2 axis in subgrids and sums up
        all the results.

        If the PDF is instantiated with more than one member, the output
        will contain an extra first dimension to accomodate the members

        Parameters
        ----------
            u: tf.tensor(int)
                list of PID to compute
            arr_x: tf.tensor(float)
                x-grid for the evaluation of the pdf
            arr_q2: tf.tensor(float)
                q2-grid for the evaluation of the pdf

        Returns
        -------
            res: tf.tensor(float)
                grid of results ([members], number of points, flavour)
        """

        a_x = tf.math.log(arr_x, name="logx")
        a_q2 = tf.math.log(arr_q2, name="logq2")

        size_a = tf.size(a_x, out_type=DTYPEINT)
        size_u = tf.size(u, out_type=DTYPEINT)
        shape = tf.stack([size_a, size_u])

        all_res = []
        for subgrids in self.grids:
            res = tf.zeros(shape, dtype=DTYPE)
            for subgrid in subgrids:
                res += subgrid(shape, a_q2, pids=u, arr_x=a_x)
            all_res.append(res)

        # This conditional is only seen once
        # if there is only one member, it will always return the result directly
        if len(self.grids) == 1:
            return res
        else:
            return tf.stack(all_res)

    @tf.function(input_signature=[GRID_I, GRID_F, GRID_F])
    def xfxQ2(self, pid, a_x, a_q2):
        """
        User interface for pdfflow when called with
        tensorflow tensors
        returns PDF evaluated in each f(x,q2) for each pid

        The output of the function is of shape
        (members, number_of_points, flavours)
        but note that dimensions of size 1 will be squeezed out.
        for instance, if called with one single value of the x, q2 pair
        for one single member for one single flavour the result will be a scalar.

        Example
        -------
        >>> from pdfflow.pflow import mkPDF
        >>> from pdfflow.configflow import run_eager, float_me, int_me
        >>> run_eager(True)
        >>> pdf = mkPDF("NNPDF31_nlo_as_0118/0")
        >>> pdf.xfxQ2(int_me([21]), float_me([0.4]), float_me([15625.0]))
        <tf.Tensor: shape=(), dtype=float64, numpy=0.06684181536088425>
        >>> pdf.xfxQ2(int_me([1,2]), float_me([0.4]), float_me([15625.0]))
        <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.08914275, 0.28612276])>
        >>> pdf.xfxQ2(int_me([1]), float_me([0.1, 0.3, 0.6]), float_me([15625.0, 91.0, 2.0]))
        <tf.Tensor: shape=(3,), dtype=float64, numpy=array([0.39789932, 0.18068107, 0.02278675])>
        >>> pdf.xfxQ2(int_me([21, 1]), float_me([0.1, 0.3, 0.6]), float_me([15625.0, 91.0, 2.0]))
        <tf.Tensor: shape=(3, 2), dtype=float64, numpy=
        array([[1.1642452 , 0.39789932],
            [0.16300335, 0.18068107],
            [0.05240091, 0.02278675]])>

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
            pdf: tf.tensor
                grid of results ([members], [number of points], [flavour])
        """
        # Check whether the array is a tensor
        if not tf.is_tensor(pid) or not tf.is_tensor(a_x) or not tf.is_tensor(a_q2):
            logger.error(
                "Method xfxQ2 can only be called with tensorflow variables "
                "use `py_xfxQ2` to obtain results with regular python variables "
                "or use the functions `int_me` and `float_me` from pdfflow.configflow "
                "to cast the input to tensorflow variables"
            )
            raise TypeError("xfxQ2 called with wrong types")
            # TODO: this error only shows up in EagerMode because otherwise it fails
            # before arriving due to the signature

        # And ensure it is unique
        # TODO maybe error if the user ask for the same pid twice or for a non-registered pid?
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
        # the flavour axis here is the last one
        f_f = tf.gather(f_f, user_idx, axis=-1)

        result = tf.squeeze(f_f)
        return result

    @tf.function(input_signature=[GRID_F, GRID_F])
    def xfxQ2_allpid(self, a_x, a_q2):
        """
        User interface for pdfflow when called with
        tensorflow tensors.
        returns PDF evaluated in each f(x,q2) for all flavours

        The output of the function is of shape
        (members, number_of_points, all_flavours)
        but note that dimensions of size 1 will be squeezed out.

        Example
        -------
        >>> from pdfflow.pflow import mkPDF
        >>> from pdfflow.configflow import run_eager, float_me, int_me
        >>> run_eager(True)
        >>> pdf = mkPDF("NNPDF31_nlo_as_0118/0")
        >>> pdf.xfxQ2_allpid(float_me([0.4]), float_me([15625.0]))
        <tf.Tensor: shape=(11,), dtype=float64, numpy=
        array([2.03487396e-04, 4.64913697e-03, 2.36497526e-04, 4.54391659e-03,
            3.81215383e-03, 6.68418154e-02, 8.91427455e-02, 2.86122760e-01,
            5.96581806e-03, 4.64913709e-03, 2.03487286e-04])>

        Parameters
        ----------
            a_x: tf.tensor, dtype=float
                grid on x where to compute the PDF
            a_q2: tf.tensor, dtype=float
                grid on q^2 where to compute the PDF
        Returns
        -------
            pdf: tensor
                grid of results ([members], [number of points], flavour)
        """
        pid = self.flavor_scheme
        return self.xfxQ2(pid, a_x, a_q2)

    # Python version of the above functions with the correct casting to tf
    def py_xfxQ2_allpid(self, arr_x, arr_q2):
        """
        Python interface for pdfflow
        The input gets converted to the right tensorflow type
        before calling the corresponding function: `xfxQ2_allpid`
        Returns all flavours

        The output of the function is of shape
        (members, number_of_points, all_flavours)
        but note that dimensions of size 1 will be squeezed out.

        Example
        -------
        >>> from pdfflow.pflow import mkPDF
        >>> from pdfflow.configflow import run_eager
        >>> run_eager(True)
        >>> pdf = mkPDF("NNPDF31_nlo_as_0118/0")
        >>> pdf.py_xfxQ2_allpid([0.4], [15625.0])
        <tf.Tensor: shape=(11,), dtype=float64, numpy=
        array([2.03487396e-04, 4.64913697e-03, 2.36497526e-04, 4.54391659e-03,
            3.81215383e-03, 6.68418154e-02, 8.91427455e-02, 2.86122760e-01,
            5.96581806e-03, 4.64913709e-03, 2.03487286e-04])>

        Parameters
        ----------
            arr_x: np.array
                grid on x where to compute the PDF
            arr_q2: np.array
                grid on q^2 where to compute the PDF
        Returns
        -------
            pdf: tensor
                grid of results ([members], [number of points], flavour)
        """
        a_x = float_me(arr_x)
        a_q2 = float_me(arr_q2)
        return self.xfxQ2_allpid(a_x, a_q2)

    def py_xfxQ2(self, pid, arr_x, arr_q2):
        """
        Python interface for pdfflow
        The input gets converted to the right tensorflow type
        before calling the corresponding function: `xfxQ2`
        returns PDF evaluated in each f(x,q2) for each pid

        The output of the function is of shape
        (members, number_of_points, flavours)
        but note that dimensions of size 1 will be squeezed out.

        Example
        -------
        >>> from pdfflow.pflow import mkPDFs
        >>> from pdfflow.configflow import run_eager, float_me, int_me
        >>> run_eager(True)
        >>> pdf = mkPDFs("NNPDF31_nlo_as_0118", [0, 1, 4])
        >>> pdf.py_xfxQ2(21, [0.4, 0.5], [15625.0, 15625.0])
        <tf.Tensor: shape=(3, 2), dtype=float64, numpy=
        array([[0.02977381, 0.00854525],
            [0.03653673, 0.00929325],
            [0.031387  , 0.00896622]])>
        >>> pdf.py_xfxQ2([1,2], [0.4], [15625.0])
        <tf.Tensor: shape=(3, 2), dtype=float64, numpy=
        array([[0.05569674, 0.19323399],
            [0.05352555, 0.18965438],
            [0.04515956, 0.18704451]])>


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
                grid of results ([members], [number of points], [flavour])
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
        # Check whether the array is a tensor
        if not tf.is_tensor(a_q2):
            logger.error(
                "The `alphasQ2` method can only be called with tensorflow variables "
                "use `py_alphasQ2` to obtain results with regular python variables "
                "or use the functions `int_me` and `float_me` from pdfflow.configflow "
                "to cast the input to tensorflow variables"
            )
            raise TypeError("alphasQ2 called with wrong types")
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
        if not tf.is_tensor(a_q):
            logger.error(
                "The `alphasQ` method can only be called with tensorflow variables "
                "use `py_alphasQ` to obtain results with regular python variables "
                "or use the functions `int_me` and `float_me` from pdfflow.configflow "
                "to cast the input to tensorflow variables"
            )
            raise TypeError("alphasQ called with wrong types")
        return self.alphasQ2(tf.square(a_q))

    def py_alphasQ2(self, arr_q2):
        """
        User interface for pdfflow alphas interpolation when called
        with python variables.
        Returns alpha_s for each value of Q2

        Example
        -------
        >>> from pdfflow.pflow import mkPDF
        >>> from pdfflow.configflow import run_eager
        >>> run_eager(True)
        >>> pdf = mkPDF("NNPDF31_nlo_as_0118/0")
        >>> pdf.py_alphasQ2([15625.0, 94])
        <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.11264939, 0.17897409])>

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
        User interface for pdfflow alphas interpolation when called
        with python variables.
        Returns alpha_s for each value of Q
        User interface for pdfflow alphas interpolation when called with

        Example
        -------
        >>> from pdfflow.pflow import mkPDF
        >>> from pdfflow.configflow import run_eager
        >>> run_eager(True)
        >>> pdf = mkPDF("NNPDF31_nlo_as_0118/0")
        >>> pdf.py_alphasQ([125.0, 94])
        <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.11264939, 0.11746421])>

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

    def trace(self):
        """
        Builds all the needed graph in advance of interpolations
        """
        logger.info("Building tf.Graph, this can take a while...")
        x = []
        q2 = []

        # We need to build all members and they could have different grids
        xgrids = []
        q2grids = []

        for subgrids in self.grids:

            # Get all x and q
            for s in subgrids:
                q2min = tf.math.exp(s.log_q2min).numpy()
                q2max = tf.math.exp(s.log_q2max).numpy()

                xmin = tf.math.exp(s.log_xmin).numpy()
                xmax = tf.math.exp(s.log_xmax).numpy()

                xgrids.append((xmin, xmax))
                q2grids.append((q2min, q2max))

        # Make the lists into sets to remove duplicates
        xgrids = list(set(xgrids))
        q2grids = list(set(q2grids))

        # Put points in the extrapolation region
        xmax = np.max(xgrids)
        xmin = np.min(xgrids)

        q2max = np.max(q2grids)
        q2min = np.min(q2grids)

        xgrids.append((xmin * 0.99, xmin))
        xgrids.append((xmax, xmax * 1.01))
        q2grids.append((q2min * 0.99, q2min))
        q2grids.append((q2max, q2max * 1.01))

        # Now create a set of points such that all x-grids are visited
        # for all grids in q2
        for xmin, xmax in xgrids:
            xpoint = (xmax - xmin) / 2.0
            for q2min, q2max in q2grids:
                x.append(xpoint)
                q2.append((q2max - q2min) / 2.0)

        # Make into an array that can be called
        x = np.array(x)
        q2 = np.array(q2)

        # trigger retracings
        self.py_xfxQ2_allpid(x, q2)
        self.py_xfxQ2(21, x, q2)

    def alphas_trace(self):
        """
        Builds all the needed graph in advance
        of alpha_s interpolations
        """
        logger.info("Building tf.Graph ...")
        q2 = []

        q2min = np.exp(self.alphas_subgrids[0].log_q2min)

        # Q2 < Q2min
        q2 += [q2min * 0.99]

        for s in self.alphas_subgrids:

            q2min = np.exp(s.log_q2min)
            q2max = np.exp(s.log_q2max)

            # points inside the grid
            q2 += [(q2min + q2max) * 0.5]

        # Get the upper boundary
        q2max = np.exp(self.alphas_subgrids[-1].log_q2max)

        # Q2 > Q2max
        q2 += [q2max * 1.01]

        q2 = np.array(q2)

        # trigger retracings
        self.py_alphasQ2(q2)
        self.py_alphasQ(q2 ** 0.5)
