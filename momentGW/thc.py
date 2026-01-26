"""
Tensor hyper-contraction.
"""

import h5py
import numpy as np

from momentGW import ints, logging, util
from momentGW.logging import init_logging

import time

from pyscf import lib
from pyscf.dft import gen_grid
from scipy.linalg.lapack import dpstrf
from hilbertcurve.hilbertcurve import HilbertCurve

class Integrals(ints.Integrals):
    """
    Container for the tensor-hypercontracted integrals required for GW
    methods.

    Parameters
    ----------
    with_df : pyscf.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    mo_occ : numpy.ndarray
        Molecular orbital occupations.
    file_path : str, optional
        Path to the HDF5 file containing the integrals. Default value is
        `None`.
    """

    def __init__(
        self,
        with_df,
        mo_coeff,
        mo_occ,
        mol=None,
        file_path=None,
        cholesky_tol=1e-6,
    ):
        # Parameters
        self._with_df = with_df
        self._mo_coeff = mo_coeff
        self._mo_occ = mo_occ

        # Options
        self.file_path = file_path
        self.compression = None
        self.mol = mol
        self.cholesky_tol = cholesky_tol

        # Logging
        init_logging()

        # Attributes
        self._blocks = {}
        self._blocks["coll"] = None
        self._blocks["cou"] = None
        self._mo_coeff_g = None
        self._mo_coeff_w = None
        self._mo_occ_w = None
        self._rot = None

    def get_compression_metric(self):
        """Return the compression metric - not currently used in THC."""
        return None

    def import_thc_components(self):
        """
        Import a HDF5 file containing a dictionary. The keys
        `"collocation_matrix"` and a `"coulomb_matrix"` must exist, with
        shapes ``(MO, aux)`` and ``(aux, aux)``, respectively.
        """

        if self.file_path is None:
            if self.mol is None:
                raise ValueError("Include mol as a thc_opts")
            cderi = lib.unpack_tril(self._with_df._cderi, axis=-1)
            t0 = time.time()
            thc_ints = Gen_integrals(self.mol,cderi,cholesky_tol=self.cholesky_tol)
            t1 = time.time()
            print("Build THC (%.4f s)" % (t1 - t0))
            del cderi
            coll = thc_ints.coll
            cou = thc_ints.cou

            # raise ValueError("file path cannot be None for THC implementation")
        else:
            thc_eri = h5py.File(self.file_path, "r")
            coll = np.array(thc_eri["collocation_matrix"])[..., 0].T
            cou = np.array(thc_eri["coulomb_matrix"])[0, ..., 0]
        self._blocks["coll"] = coll
        self._blocks["cou"] = cou

        self._naux = self.cou.shape[0]

    @logging.with_status("Transforming integrals")
    def transform(self, do_Lpq=True, do_Lpx=True, do_Lia=True):
        """
        Transform the integrals in-place.

        Parameters
        ----------
        do_Lpq : bool, optional
            Whether the ``(aux, MO, MO)`` array is required. In THC,
            this requires the `Lp` array. Default value is `True`.
        do_Lpx : bool, optional
            Whether the ``(aux, MO, MO)`` array is required. In THC,
            this requires the `Lx` array. Default value is `True`.
        do_Lia : bool, optional
            Whether the ``(aux, occ, vir)`` array is required. In THC,
            this requires the `Li` and `La` arrays. Default value is
            `True`.
        """

        # Check if any arrays are required
        if not any([do_Lpq, do_Lpx, do_Lia]):
            return

        # Import THC components
        if self.coll is None and self.cou is None:
            self.import_thc_components()

        # Transform the (L|pq) array
        if do_Lpq:
            Lp = util.einsum("Lp,pq->Lq", self.coll, self.mo_coeff)
            self._blocks["Lp"] = Lp

        # Transform the (L|px) array
        if do_Lpx:
            Lx = util.einsum("Lp,pq->Lq", self.coll, self.mo_coeff_g)
            self._blocks["Lx"] = Lx

        # Transform the (L|ia) and (L|ai) arrays
        if do_Lia:
            ci = self.mo_coeff_w[:, self.mo_occ_w > 0]
            ca = self.mo_coeff_w[:, self.mo_occ_w == 0]

            Li = util.einsum("Lp,pi->Li", self.coll, ci)
            La = util.einsum("Lp,pa->La", self.coll, ca)

            self._blocks["Li"] = Li
            self._blocks["La"] = La

    @logging.with_timer("J matrix")
    @logging.with_status("Building J matrix")
    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : numpy.ndarray
            J matrix.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        # Build the J matrix
        tmp = util.einsum("pq,Kp,Kq->K", dm, Lp, Lp)
        tmp = util.einsum("K,KL->L", tmp, cou)
        vj = util.einsum("L,Lr,Ls->rs", tmp, Lp, Lp)

        return vj

    @logging.with_timer("K matrix")
    @logging.with_status("Building K matrix")
    def get_k(self, dm, basis="mo"):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        # Build the K matrix
        tmp = util.einsum("pq,Kp->Kq", dm, Lp)
        tmp = util.einsum("Kq,Lq->KL", tmp, Lp)
        tmp = util.einsum("KL,KL->KL", tmp, cou)
        tmp = util.einsum("KL,Ks->Ls", tmp, Lp)
        vk = util.einsum("Ls,Lr->rs", tmp, Lp)

        return vk

    @property
    def coll(self):
        """Get the ``(aux, MO)`` collocation array."""
        return self._blocks["coll"]

    @property
    def cou(self):
        """Get the ``(aux, aux)`` Coulomb array."""
        return self._blocks["cou"]

    @property
    def Lp(self):
        """Get the ``(aux, MO)`` array."""
        return self._blocks["Lp"]

    @property
    def Lx(self):
        """Get the ``(aux, MO)`` array."""
        return self._blocks["Lx"]

    @property
    def Li(self):
        """Get the ``(aux, W occ)`` array."""
        return self._blocks["Li"]

    @property
    def La(self):
        """Get the ``(aux, W vir)`` array."""
        return self._blocks["La"]


class Gen_integrals():
    def __init__(self,
    mol,
    cderi,
    grids=None,
    coeffs=None,
    cholesky_tol = (1/3)*1e-5,#1e-5,#1e-8,
    cluster_block_size = 1024,
    hilbert_sort_bits = 16,
    max_iterations = 100,
    ):
        self.mol = mol
        self.grids = grids
        self.cderi = cderi

        self.coeffs = coeffs
        self._cholesky_tol = cholesky_tol
        self._cluster_block_size = cluster_block_size
        self._hilbert_sort_bits = hilbert_sort_bits
        self._max_iterations = max_iterations

        if grids is not gen_grid.Grids:
            print("INTERNAL GRIDS")
            grids = gen_grid.Grids(self.mol)
            grids.prune = None
            grids.level = 0
            grids.kernel()
            self.grids = grids

        self.coll, self.cou = self.factorise_thc()
        self.cderi=None

    def factorise_thc(self):
        r"""Perform tensor hypercontraction on a 3-center integral tensor in N^4 time.

        .. math::
            \sum_{L} V_{Lpq} V_{Lrs} \rightarrow \sum_{KL} X_{Kp} X_{Kq} Z_{KL} Z_{Lr} Z_{Ls}

        Args:
            tensor: The 3-center integral tensor to hypercontract.
            mol: The molecular object.
            grids: The grid object containing coordinates and weights.
            coeffs: Optional coefficients to rotate into the basis of the last two indices of the
                tensor.
            cholesky_tol: The tolerance for the Cholesky decomposition.
            cluster_block_size: The block size for processing the coordinates.
            hilbert_sort_bits: The number of bits to use for Hilbert curve sorting.

        Returns:
            A tuple containing the collocation matrices and Coulomb matrix for the tensor
            hypercontracted integrals.

        Reference:
            https://arxiv.org/abs/2506.19392v1
        """
        if self.cderi.ndim != 3:
            raise ValueError("The tensor must be a 3-center integral tensor.")
        order = self.hilbert_argsort(self.grids.coords, bits=self._hilbert_sort_bits)
        coords = self.grids.coords[order]
        weights = self.grids.weights[order]

        npoints = coords.shape[0]
        for i in range(self._max_iterations):
            indices = np.zeros((0,), dtype=int)
            for start in range(0, coords.shape[0], self._cluster_block_size):
                end = min(start + self._cluster_block_size, coords.shape[0])
                pivots = self.get_pivots(coords[start:end], weights[start:end])
                indices = np.concatenate((indices, pivots + start))
            coords = coords[indices]
            weights = weights[indices]
            if npoints <= self._cluster_block_size or coords.shape[0] == npoints:
                break
            npoints = coords.shape[0]

        x = self.mol.eval_gto("GTOval_sph", coords)
        x *= np.abs(weights)[:, None] ** 0.5
        overlap = util.einsum("pi,qi->pq", x, x) ** 2

        if self.coeffs is not None:
            xi = x @ self.coeffs[0]
            xj = x @ self.coeffs[1]
        else:
            xi = xj = x

        a = np.einsum("pi,pj,Lij->pL", xi, xj, self.cderi) # SOMETHING WITH UTIL EINSUM HERE?????
        b = np.linalg.solve(overlap, a)
        v = util.einsum("pL,qL->pq", b, b)


        return x, v

    def hilbert_argsort(self, coords,  bits = 16):
        """Indirectly sort the coordinates using a Hilbert curve.

        Args:
            coords: The coordinates to sort.
            bits: The number of bits to use for the Hilbert curve.

        Returns:
            An array of indices that sorts the coordinates according to the Hilbert curve.
        """
        scoords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        indices = np.floor(scoords * (2 ** bits - 1)).astype(int)
        hilbert_curve = HilbertCurve(bits, 3)
        return np.argsort(hilbert_curve.distances_from_points(indices))

    def get_pivots(
            self,
            coords,
            weights,
    ):
        """Get the pivot indices for the coordinates based on a pivoted Cholesky decomposition.

        Args:
            mol: The molecular object.
            coords: The coordinates to process.
            weights: The weights associated with the coordinates.

        Returns:
            An array of pivot indices.
        """
        ao = self.mol.eval_gto("GTOval_sph", coords)
        x = util.einsum("pi,p->pi", ao, np.abs(weights) ** 0.5)
        s = util.einsum("pi,qi->pq", x, x) ** 2
        _, pivots = self.pivoted_cholesky(s)
        return pivots

    def pivoted_cholesky(self, matrix):
        """Perform a pivoted Cholesky decomposition.

        Args:
            matrix: The matrix to decompose.
            tol: The tolerance for the decomposition.

        Returns:
            A tuple containing the upper triangular matrix and the pivot indices.
        """
        tri, pivots, rankc, info = dpstrf(matrix, tol=self._cholesky_tol)
        pivots = pivots[:rankc] - 1
        return tri, pivots


