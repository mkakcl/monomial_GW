"""
Tensor hyper-contraction with periodic boundary conditions.
"""

import h5py
import numpy as np

from momentGW import logging, util, mpi_helper
from momentGW.pbc.ints import KIntegrals as DFKIntegrals

from momentGW.thc import Integrals



class KIntegrals(Integrals, DFKIntegrals):
    """
    Container for the tensor-hypercontracted integrals required for GW
    methods with periodic boundary conditions.

    Parameters
    ----------
    with_df : pyscf.pbc.df.DF
        Density fitting object.
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients at each k-point.
    mo_occ : numpy.ndarray
        Molecular orbital occupations at each k-point.
    file_path : str, optional
        Path to the HDF5 file containing the integrals. Default value is
        `None`.
    """

    def __init__(
        self,
        with_df,
        kpts,
        mo_coeff,
        mo_occ,
        file_path=None,
        store_full=False,
    ):
        Integrals.__init__(
            self,
            with_df,
            mo_coeff,
            mo_occ,
            file_path=file_path,
        )

        # Parameters
        self.kpts = kpts
        self.store_full = store_full

        # Options
        self.compression = None

    def import_thc_components(self):
        """
        Import a HDF5 file containing a dictionary. The keys
        `"collocation_matrix"` and a `"coulomb_matrix"` must exist, with
        shapes ``(MO, aux)`` and ``(aux, aux)``, respectively.
        """

        if self.file_path is None:
            raise ValueError("file path cannot be None for THC implementation")

        thc_eri = h5py.File(self.file_path, "r")

        kpts_imp = np.array(thc_eri["kpts"])

        if kpts_imp.shape[0] != len(self.kpts):
            raise ValueError("Number of kpts imported differs from pyscf")
        if not np.allclose(kpts_imp, self.kpts._kpts) and not np.allclose(
            kpts_imp, -self.kpts._kpts
        ):
            raise ValueError("Different kpts imported to those in pyscf")

        cou = {}
        coll = {}
        for ki in range(len(self.kpts)):
            cou_arr = np.array(thc_eri["coulomb_matrix"][ki])
            cou[ki] = cou_arr[:, :, 0] + 1j * cou_arr[:, :,  1]
            coll_arr = np.array(thc_eri["collocation_matrix"][0,ki])
            coll[ki] = coll_arr[:, :, 0] + 1j * coll_arr[:, :,  1]

        self._blocks["coll"] = coll
        self._blocks["cou"] = cou

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

        Lp = {}
        Lx = {}
        Li = {}
        La = {}

        do_Lpq = self.store_full if do_Lpq is None else do_Lpq

        for ki in range(self.nkpts):
            # Transform the (L|pq) array
            if do_Lpq:
                Lp[ki] = util.einsum("Lp,pq->Lq", self.coll[ki], self.mo_coeff[ki])

            # Transform the (L|px) array
            if do_Lpx:
                Lx[ki] = util.einsum("Lp,pq->Lq", self.coll[ki], self.mo_coeff_g[ki])

            # Transform the (L|ia) and (L|ai) arrays
            if do_Lia:
                ci = self.mo_coeff_w[ki][:, self.mo_occ_w[ki] > 0]
                ca = self.mo_coeff_w[ki][:, self.mo_occ_w[ki] == 0]

                Li[ki] = util.einsum("Lp,pi->Li", self.coll[ki], ci)
                La[ki] = util.einsum("Lp,pa->La", self.coll[ki], ca)

        if do_Lpq:
            self._blocks["Lp"] = Lp
        if do_Lpx:
            self._blocks["Lx"] = Lx
        if do_Lia:
            self._blocks["Li"] = Li
            self._blocks["La"] = La

    @logging.with_timer("J matrix")
    @logging.with_status("Building J matrix")
    def get_j(self, dm, basis="mo"):
        """Build the J matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point.
        basis : str, optional
            Basis in which to build the J matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vj : numpy.ndarray
            J matrix at each k-point.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        vj = np.zeros_like(dm, dtype=complex)
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        buf = 0.0
        for ki in range(self.nkpts):
            tmp = util.einsum("pq,Kp,Kq->K", dm[ki], Lp[ki], Lp[ki].conj())
            tmp = util.einsum("K,KL->L", tmp, cou[0])
            buf += tmp

        buf /= self.nkpts

        for kj in range(self.nkpts):
            vj[kj] = util.einsum("L,Lr,Ls->rs", buf, Lp[kj].conj(), Lp[kj])

        return vj

    @logging.with_timer("K matrix")
    @logging.with_status("Building K matrix")
    def get_k(self, dm, basis="mo", ewald=False):
        """Build the K matrix.

        Parameters
        ----------
        dm : numpy.ndarray
            Density matrix at each k-point.
        basis : str, optional
            Basis in which to build the K matrix. One of
            `("ao", "mo")`. Default value is `"mo"`.

        Returns
        -------
        vk : numpy.ndarray
            K matrix at each k-point.

        Notes
        -----
        The basis of `dm` must be the same as `basis`.
        """

        # Check the input
        assert basis in ("ao", "mo")

        # Get the components
        vk = np.zeros_like(dm, dtype=complex)
        if basis == "ao":
            if self.coll is None and self.cou is None:
                self.import_thc_components()
            Lp = self.coll
            cou = self.cou
        else:
            Lp = self.Lp
            cou = self.cou

        rho_cou = np.zeros((self.nkpts, self.naux, self.naux), dtype=complex)
        for kk in range(self.nkpts):
            tmp = util.einsum("pq,Kp->Kq", dm[kk], Lp[kk].conj())
            rho_cou[kk] = util.einsum("Kq,Lq->KL", tmp, Lp[kk])

        for ki in range(self.nkpts):
            buf = np.zeros((self.naux, self.naux), dtype=complex)
            for kk in range(self.nkpts):
                q = self.kpts.member(self.kpts.wrap_around(self.kpts[ki] + self.kpts[kk]))
                buf += util.einsum("KL,KL->KL", rho_cou[kk], cou[q])
            buf /= self.nkpts
            tmp = util.einsum("KL,Ks->Ls", buf, Lp[ki].conj())
            vk[ki] += util.einsum("Ls,Lr->rs", tmp, Lp[ki])

        return vk

    @property
    def nkpts(self):
        """Get the number of k-points"""
        return len(self.kpts)

    @property
    def naux(self):
        """Get the number of auxiliary basis functions."""
        return self.cou[0].shape[0]



