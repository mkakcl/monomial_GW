import numpy as np
from scipy.special import binom

from momentGW import logging, util
from momentGW.tda import dTDA as DFdTDA

from scipy.linalg import cholesky

import tracemalloc
import time
from momentGW import metrics

# from scalene_test import t2


class dTDA(DFdTDA):
    """
    Compute the self-energy moments using dTDA and numerical integration
    with tensor-hypercontraction.

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

    def get_D_list(self, max, ei=None, ea=None):
        if ei is None:
            ei = self.mo_energy_w[self.mo_occ_w > 0]
        if ea is None:
            ea = self.mo_energy_w[self.mo_occ_w == 0]

        ea_range = np.power.outer(ea, np.arange(max))
        cou_vir = util.einsum("am,Pa,Qa->mPQ", ea_range, self.La, self.La)
        # print("Pre D size memory", tracemalloc.get_traced_memory())
        D_list = np.zeros((max, self.naux, self.naux))
        # print("post D size memory", tracemalloc.get_traced_memory())


        for i in range(max):
            cou_occ = util.einsum("i,Pi,Qi->PQ", ei ** i, self.Li, self.Li) * pow(-1, i)
            coeff = np.asarray([binom(j, i) for j in range(i,max)])
            # print(i," memory", tracemalloc.get_traced_memory())
            for m in range(i,max):
                D_list[m] += coeff[m-i]*util.einsum("PQ,PQ->PQ", cou_occ, cou_vir[m-i])

        del cou_vir, cou_occ

        # print("Pre cou memory", tracemalloc.get_traced_memory())
        for i in range(max):
            D_list[i] = util.einsum("PQ,QR->PR", self.cou, D_list[i])
        # print("post cou memory", tracemalloc.get_traced_memory())

        return D_list


    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """
        Build the moments of the density-density response using
        tensor-hypercontraction.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.

        Notes
        -----
        Unlike the standard `momentGW.tda` implementation, this method
        scales as :math:`O(N^3)` with system size instead of
        :math:`O(N^4)`.
        """

        # Build the energy difference matrices
        print("Pre_d_list memory", tracemalloc.get_traced_memory())
        t1 = time.time()
        D_list = self.get_D_list(self.nmom_max + 1)
        t2 = time.time()
        print("D_list time", t2 - t1)
        metrics.record("D_list_time", t2 - t1)
        print("Post d_list memory", tracemalloc.get_traced_memory())

        moments = D_list.copy()
        D_list *= 2.0

        # Get the higher order moments
        for i in range(1, self.nmom_max + 1):
            # moments[i] += util.einsum("mPQ, mQR -> PR", D_list[:i][::-1], moments[:i])
            for m in range(i, self.nmom_max + 1):
                moments[m] += util.einsum("PQ,QR->PR", D_list[m - i], moments[i - 1])
        del D_list
        # moments = util.einsum("mPR, RS -> mPS", moments, self.cou) # CHANGED FOR USING SPLIT MIDDLE cou


        moments = util.einsum("mPQ,QR->mPR", moments, self.cou) # add this back in for new se

        return moments

    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, mo_energy_g=None, mo_occ_g=None):
        """
        Build the moments of the self-energy via convolution with
        tensor-hypercontraction.

        Parameters
        ----------
        moments_dd : numpy.ndarray
            Moments of the density-density response.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy.
        """
        t1 = time.time()
        W = self.build_dd_moments()
        t2 = time.time()
        dt = t2 - t1
        print("dd_moments time", dt)
        metrics.record("dd_moments_time", dt)
        print("Post dd_moments memory", tracemalloc.get_traced_memory())

        if mo_energy_g is None:
            mo_energy_g = self.mo_energy_g
        if mo_occ_g is None:
            mo_occ_g = self.mo_occ_g

        # Setup dependent on diagonal SE and initialise moments
        if self.gw.diagonal_se:
            chars = ("p", "p", "p", lambda x: np.diag(x))
        else:
            chars = ("pq", "p", "q", lambda x: x)

        # Convolve the occupied and virtual moments separately
        print("")
        t3 = time.time()
        moments_occ = self.convolve_occ(W, mo_energy_g, mo_occ_g, chars)
        t4 = time.time()
        dt = t4 - t3
        print("moments_occ time", dt)
        metrics.record("moments_occ_time", dt)
        print("Post moments_occ memory", tracemalloc.get_traced_memory())
        print("")
        moments_vir = self.convolve_vir(W, mo_energy_g, mo_occ_g, chars)
        t5 = time.time()
        dt = t5 - t4
        print("moments_vir time", dt)
        metrics.record("moments_vir_time", dt)
        print("Post moments_vir memory", tracemalloc.get_traced_memory())
        print("")

        return moments_occ, moments_vir

    @logging.with_timer("Occupied moment convolution")
    @logging.with_status("Convoluting occupied moments")
    def convolve_occ(self, W, mo_energy_g, mo_occ_g, chars):
        """
        Handle the convolution of the moments of the Green's function
        and screened Coulomb interaction.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction at each
            k-point.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function at each k-point. If
            `None`, use `self.mo_energy_g`. Default value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function at each k-point. If
            `None`, use `self.mo_occ_g`. Default value is `None`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        """
        # TODO fix this doc string
        (pqchar, pchar, qchar, fproc) = chars

        max_nmom = self.nmom_max + 1
        moms = np.arange(max_nmom)
        nmo = self.Lp.shape[-1]

        moments_occ = np.zeros((max_nmom, nmo, nmo))

        t1 = time.time()
        ea_range = np.power.outer(mo_energy_g[mo_occ_g > 0], moms) * 2.0
        cou_G = util.einsum("am,Pa,Qa->mPQ", ea_range, self.Lx[:,mo_occ_g > 0], self.Lx[:,mo_occ_g > 0])
        t2 = time.time()
        dt = t2 - t1
        print("cou_G occ time", dt)
        metrics.record("cou_G_occ_time", dt)

        for n in moms:
            eta_orders = np.arange(n+1)
            fh = binom(n, eta_orders)* (-1) ** eta_orders
            GW_occ = np.zeros((self.naux,self.naux))
            for t in range(n+1):
                GW_occ += fh[t]*util.einsum("PQ,PQ->PQ", W[t], cou_G[n-t])
            moments_occ[n] = fproc(util.einsum(f"P{pchar},PQ,Q{qchar}->{pqchar}", self.Lp,
                                       GW_occ, self.Lp))

        # Numerical integration can lead to small non-hermiticity
        moments_occ = 0.5 * (moments_occ + moments_occ.swapaxes(1, 2).conj())

        return moments_occ

    @logging.with_timer("Occupied moment convolution")
    @logging.with_status("Convoluting occupied moments")
    def convolve_vir(self, W, mo_energy_g, mo_occ_g, chars):
        """
        Handle the convolution of the moments of the Green's function
        and screened Coulomb interaction.

        Parameters
        ----------
        eta : numpy.ndarray
            Moments of the density-density response partly transformed
            into moments of the screened Coulomb interaction at each
            k-point.
        mo_energy_g : numpy.ndarray, optional
            Energies of the Green's function at each k-point. If
            `None`, use `self.mo_energy_g`. Default value is `None`.
        mo_occ_g : numpy.ndarray, optional
            Occupancies of the Green's function at each k-point. If
            `None`, use `self.mo_occ_g`. Default value is `None`.

        Returns
        -------
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """
        # TODO fix this doc string
        (pqchar, pchar, qchar, fproc) = chars

        max_nmom = self.nmom_max + 1
        moms = np.arange(max_nmom)
        nmo = self.Lp.shape[-1]

        moments_vir = np.zeros((max_nmom, nmo, nmo))

        t1 = time.time()
        ea_range = np.power.outer(mo_energy_g[mo_occ_g == 0], moms)* 2.0
        cou_G = util.einsum("am,Pa,Qa->mPQ", ea_range, self.Lx[:,mo_occ_g == 0], self.Lx[:,mo_occ_g == 0])
        t2 = time.time()
        dt = t2 - t1
        print("cou_G vir time", dt)
        metrics.record("cou_G_vir_time", dt)

        for n in moms:
            eta_orders = np.arange(n + 1)
            fp = binom(n, eta_orders)
            GW_vir = np.zeros((self.naux, self.naux))
            for t in range(n + 1):
                GW_vir += fp[t]*util.einsum("PQ,PQ->PQ", W[t], cou_G[n-t])
            moments_vir[n] = fproc(util.einsum(f"P{pchar},PQ,Q{qchar}->{pqchar}", self.Lp,
                                       GW_vir, self.Lp))

        # Numerical integration can lead to small non-hermiticity
        moments_vir = 0.5 * (moments_vir + moments_vir.swapaxes(1, 2).conj())

        return moments_vir

    @property
    def Li(self):
        """Get the ``(aux, W occ)`` array."""
        return self.integrals.Li

    @property
    def La(self):
        """Get the ``(aux, W vir)`` array."""
        return self.integrals.La

    @property
    def Lx(self):
        """Get the ``(aux, W occ)`` array."""
        return self.integrals.Lx

    @property
    def Lp(self):
        """Get the ``(aux, W vir)`` array."""
        return self.integrals.Lp

    @property
    def cou(self):
        """Get the ``(aux, aux)`` Coulomb array."""
        return self.integrals.cou

    @property
    def decou(self):
        """Get the ``(aux, aux)`` Coulomb array."""
        return cholesky(self.integrals.cou, lower=True)