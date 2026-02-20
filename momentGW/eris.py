"""Full 4-index ERI implementation"""

import h5py
import numpy as np
from scipy.special import binom
import pyscf.ao2mo

from momentGW import ints, logging, util
from momentGW.logging import init_logging
from momentGW.tda import dTDA as DFdTDA


class Integrals(ints.Integrals):
    """Container for the 4 index integrals required for GW methods.

    Parameters
    ----------
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    mo_occ : numpy.ndarray
        Molecular orbital occupations.
    """

    def __init__(
        self,
        with_df,
        mo_coeff,
        mo_occ,
        mf=None
    ):
        if mf is None:
            raise ValueError("Integrals class requires a mean-field object for 4-index ERIs.")
        # Parameters
        self._mo_coeff = mo_coeff
        self._mo_occ = mo_occ
        self._eris = None
        self._mf = mf

        # Options
        self.compression = None

        # Logging
        init_logging()

        # Attributes
        self._mo_coeff_g = None
        self._mo_coeff_w = None
        self._mo_occ_w = None
        self._rot = None

    def get_compression_metric(self):
        """Return the compression metric - not currently used in 4 index ERIs."""
        return None

    @logging.with_status("Transforming integrals")
    def transform(self):
        """Transform the integrals to MO basis.

        """
        self._eris = pyscf.ao2mo.kernel(self._mf.mol.intor('int2e'), self._mo_coeff)
        

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
        if basis == "ao":
            raise NotImplementedError("AO basis not implemented for 4-index ERIs")

        # Build the J matrix
        vj = util.einsum("pqrs,rs->pq", self._eris, dm)

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
        if basis == "ao":
            raise NotImplementedError("AO basis not implemented for 4-index ERIs")

        # Build the K matrix
        vk = util.einsum("prqs,rs->pq", self._eris, dm)

        return vk


    @property
    def nmo(self):
        """Number of molecular orbitals."""
        return self._mo_coeff.shape[1]

    @property
    def nocc(self):
        """Number of occupied molecular orbitals."""
        return (self._mo_occ > 0).sum()

    @property
    def nvir(self):
        """Number of virtual molecular orbitals."""
        return self.nmo - self.nocc

    @property
    def ovov(self):
        occ = np.s_[:self.nocc]
        vir = np.s_[self.nocc:]
        return self._eris[occ, vir, occ, vir]

