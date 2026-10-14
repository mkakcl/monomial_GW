"""Tensor hyper-contraction."""

import h5py
import numpy as np
from scipy.special import binom

from momentGW import ints, logging, mpi_helper, util
from momentGW.logging import init_logging
from momentGW.tda import dTDA as DFdTDA


class Integrals(ints.Integrals):
    """Container for the tensor-hypercontracted integrals required for GW methods.

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
        file_path=None,
    ):
        # Parameters
        self._with_df = with_df
        self._mo_coeff = mo_coeff
        self._mo_occ = mo_occ

        # Options
        self.file_path = file_path
        self.compression = None

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
        """Import a HDF5 file containing a dictionary.

        The keys
        `"collocation_matrix"` and a `"coulomb_matrix"` must exist, with
        shapes ``(MO, aux)`` and ``(aux, aux)``, respectively.
        """

        if self.file_path is None:
            raise ValueError("file path cannot be None for THC implementation")

        thc_eri = h5py.File(self.file_path, "r")
        coll = np.array(thc_eri["collocation_matrix"])[..., 0].T
        cou = np.array(thc_eri["coulomb_matrix"])[0, ..., 0]
        self._blocks["coll"] = coll
        self._blocks["cou"] = cou

        self._naux = self.cou.shape[0]

    @logging.with_status("Transforming integrals")
    def transform(self, do_Lpq=True, do_Lpx=True, do_Lia=True):
        """Transform the integrals in-place.

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


class dTDA(DFdTDA):
    """Compute the self-energy moments using dTDA and numerical integration with tensor-
    hypercontraction.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : BaseIntegrals
        Integrals object.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw.mo_occ`.
    """

    def kernel(self, exact=False):
        """Run the polarizability calculation to compute moments of the self-energy.

        Parameters
        ----------
        exact : bool, optional
            Has no effect and is only present for compatibility with
            `dRPA`. Default value is `False`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy for each spin channel.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy for each spin channel.
        """
        # Build the density-density response moments
        moments_dd = self.build_dd_moments()

        # Build the self-energy moments
        moments_occ, moments_vir = self.build_se_moments(moments_dd)

        return moments_occ, moments_vir

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        r"""Build the moments of the density-density response using tensor-hypercontraction.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.

        Notes
        -----
        Unlike the standard `momentGW.tda` implementation, this method
        scales as :math:`O(N^3)` with system size instead of
        :math:`O(N^4)`.

        The moments obey

        .. math::
            \zeta^{(m)} = \mathcal{D}^{(m)}
            + 2 \sum_{j<m} \mathcal{D}^{(m-1-j)} Z \zeta^{(j)},

        with :math:`\mathcal{D}^{(m)} = X^T D^m X` the auxiliary-space image of the
        particle-hole gaps to the mth power.  Because
        :math:`D^m = (\epsilon_a - \epsilon_i)^m` expands binomially into separable
        products of orbital energies, each :math:`\mathcal{D}^{(m)}` costs one pass
        over the collocation matrices; there are `nmom_max + 1` of them, not one per
        pair of moment orders.
        """

        ei = self.mo_energy_w[self.mo_occ_w > 0]
        ea = self.mo_energy_w[self.mo_occ_w == 0]
        orders = np.arange(self.nmom_max + 1)

        # Separable factors of D^m.  The occupied energies carry the sign of the gap, so
        # every binomial term below is a product of like-signed factors.
        cou_occ = util.einsum("mi,Pi,Qi->mPQ", np.power.outer(-ei, orders).T, self.Li, self.Li)
        cou_vir = util.einsum("ma,Pa,Qa->mPQ", np.power.outer(ea, orders).T, self.La, self.La)

        # Auxiliary-space images of D^m
        cou_d = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        for m in range(self.nmom_max + 1):
            for j in range(m + 1):
                cou_d[m] += binom(m, j) * cou_occ[j] * cou_vir[m - j]
        del cou_occ, cou_vir

        # Recursion
        zeta = cou_d.copy()
        for m in range(1, self.nmom_max + 1):
            for j in range(m):
                zeta[m] += np.linalg.multi_dot((cou_d[m - 1 - j], self.cou, zeta[j])) * 2.0

        return zeta

    @logging.with_timer("Green's function collocation moments")
    @logging.with_status("Constructing Green's function collocation moments")
    def build_gf_collocation_moments(self, occupied):
        r"""Build the energy-weighted collocation moments of one Green's function sector.

        Parameters
        ----------
        occupied : bool
            Whether to build the occupied sector.

        Returns
        -------
        cou_g : numpy.ndarray
            Moments, shape ``(nmom_max + 1, naux, naux)``.

        Notes
        -----
        These are

        .. math::
            G^{(j)}_{PQ} = \sum_{x \in \text{sector}} \epsilon_x^j\, X_{xP} X_{xQ} ,

        the only place the Green's function orbital index appears in the self-energy
        build.  Summing it out here rather than carrying it is what keeps
        `build_se_moments` cubic; see its notes.  One sector is built at a time because
        both at once would double the largest array the routine holds.

        Operations are ``O(nmom_max * naux^2 * nmo)`` and memory
        ``O(nmom_max * naux^2)``.
        """

        q0, q1 = self.mpi_slice(self.mo_energy_g.size)
        energies = self.mo_energy_g[q0:q1]
        mask = (self.mo_occ_g[q0:q1] > 0) == occupied
        Lx = self.integrals.Lx[:, q0:q1][:, mask]

        if not np.any(mask):
            return mpi_helper.allreduce(np.zeros((self.nmom_max + 1, self.naux, self.naux)))

        weights = np.power.outer(energies[mask], np.arange(self.nmom_max + 1)).T
        return mpi_helper.allreduce(util.einsum("jx,Px,Qx->jPQ", weights, Lx, Lx))

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, zeta):
        r"""Build the moments of the self-energy via convolution with tensor-hypercontraction.

        Parameters
        ----------
        zeta : numpy.ndarray
            Moments of the density-density response.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.

        Notes
        -----
        Operations are ``O(nmom_max^2 * (naux^2 * nmo + naux * nmo^2))`` and memory
        ``O(nmom_max * (naux^2 + nmo^2))``: cubic and quadratic in the system size, which
        is the whole reason a tensor-hypercontracted path is worth having.

        Reaching that needs the Green's function orbital index summed out before the
        rotation rather than after.  Written the direct way, the convolution asks for

        .. math::
            \eta^{(t)}_{x,pq} = 2 \sum_{PQ} X_{pP} X_{xP} X_{qQ} X_{xQ}\, \zeta'^{(t)}_{PQ}

        for every `x`, which is an ``O(naux^2 nmo^2)`` loop over an ``O(nmo^3)`` array.  But
        `x` only ever enters the output through the weighted sums the convolution takes,

        .. math::
            \Sigma^{(n)}_{pq} = \sum_{x} \sum_{t \le n} \binom{n}{t} (\pm 1)^t
                \epsilon_x^{\,n-t}\, \eta^{(t)}_{x,pq} ,

        with the sign negative for the occupied sector.  Moving that sum inside the
        auxiliary contraction leaves

        .. math::
            \Sigma^{(n)} = \sum_{t \le n} \binom{n}{t} (\pm 1)^t\;
                X^T \Big( \zeta'^{(t)} \circ G^{(n-t)} \Big) X ,

        with :math:`\circ` the elementwise product and :math:`G^{(j)}` the collocation
        moments of `build_gf_collocation_moments`.  Neither `eta` nor the loop over `x`
        survives, and the reduction over ranks moves onto the ``(naux, naux)`` moments,
        which are smaller than the ``(nmo, nmo)`` self-energy moments they replace.

        The whole ``t`` sum is accumulated in the auxiliary space before being rotated, so
        there are ``2 (nmom_max + 1)`` rotations rather than one per ``(n, t)`` pair.  The
        rotation is the only term carrying a factor of `nmo`, so doing it
        ``(nmom_max + 2) / 2`` times fewer is most of the gain.
        """

        Lp = self.integrals.Lp
        moments = {
            True: np.zeros((self.nmom_max + 1, self.nmo, self.nmo)),
            False: np.zeros((self.nmom_max + 1, self.nmo, self.nmo)),
        }

        # Factor two as in the density-density to screened-Coulomb rotation.  Written as a
        # loop rather than one `einsum` so that no intermediate of the same size as the
        # result is allocated alongside it.
        zeta_prime = np.empty_like(zeta)
        for t in range(self.nmom_max + 1):
            zeta_prime[t] = np.linalg.multi_dot((self.cou, zeta[t], self.cou))
        zeta_prime *= 2.0

        for occupied in (True, False):
            cou_g = self.build_gf_collocation_moments(occupied)
            sign = -1.0 if occupied else 1.0

            for n in range(self.nmom_max + 1):
                accumulated = np.zeros((self.naux, self.naux))
                for t in range(n + 1):
                    accumulated += (binom(n, t) * sign**t) * (zeta_prime[t] * cou_g[n - t])

                if self.gw.diagonal_se:
                    moments[occupied][n] = np.diag(util.einsum("PQ,Pp,Qp->p", accumulated, Lp, Lp))
                else:
                    moments[occupied][n] = np.linalg.multi_dot((Lp.T, accumulated, Lp))

            del cou_g

        moments_occ, moments_vir = moments[True], moments[False]

        # Numerical integration can lead to small non-hermiticity
        moments_occ = 0.5 * (moments_occ + moments_occ.swapaxes(1, 2).conj())
        moments_vir = 0.5 * (moments_vir + moments_vir.swapaxes(1, 2).conj())

        return moments_occ, moments_vir

    @property
    def Li(self):
        """Get the ``(aux, W occ)`` array."""
        return self.integrals.Li

    @property
    def La(self):
        """Get the ``(aux, W vir)`` array."""
        return self.integrals.La

    @property
    def cou(self):
        """Get the ``(aux, aux)`` Coulomb array."""
        return self.integrals.cou
