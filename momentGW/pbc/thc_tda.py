import numpy as np
from scipy.special import binom

from momentGW import logging, util, mpi_helper
from momentGW import metrics

from momentGW.pbc.tda import dTDA as DFdTDA
from momentGW.thc_tda import dTDA as MoldTDA

import tracemalloc
import time

class dTDA(MoldTDA, DFdTDA):
    """
    Compute the self-energy moments using dTDA with tensor
    hyper-contraction and periodic boundary conditions.

    Parameters
    ----------
    gw : BaseKGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : KIntegrals
        Density-fitted integrals.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies at each k-point. If a tuple is passed,
        the first element corresponds to the Green's function basis and
        the second to the screened Coulomb interaction. Default value is
        that of `gw.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies at each k-point. If a tuple is
        passed, the first element corresponds to the Green's function basis
        and the second to the screened Coulomb interaction. Default value
        is that of `gw.mo_occ`.
    """

    def get_D_list_parts(self, max):
        cou_occ = np.zeros((self.nkpts, max, self.naux, self.naux), dtype=np.complex128)
        cou_vir = np.zeros((self.nkpts, max, self.naux, self.naux), dtype=np.complex128)

        # Build all energy ERI pairings - scaling n_k N^3_orb
        for kj in self.kpts.loop(1, mpi=True):
            ei_range = np.power.outer(-1*self.mo_energy_w[kj][self.mo_occ_w[kj] > 0], np.arange(max))
            cou_occ[kj] = util.einsum("im,Pi,Qi->mPQ", ei_range, self.Li[kj], self.Li[kj].conj())
            ea_range = np.power.outer(self.mo_energy_w[kj][self.mo_occ_w[kj] == 0], np.arange(max))
            cou_vir[kj] = util.einsum("am,Pa,Qa->mPQ", ea_range, self.La[kj], self.La[kj].conj())
        del ei_range, ea_range
        cou_occ = mpi_helper.allreduce(cou_occ)
        cou_vir = mpi_helper.allreduce(cou_vir)

        return cou_occ, cou_vir


    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        """Build the moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response at each k-point.

        Notes
        -----
        Unlike the standard `momentGW.tda` implementation, this method
        scales as :math:`O(N^3)` with system size instead of
        :math:`O(N^4)`.
        """

        kpts = self.kpts
        naux = self.naux
        max = self.nmom_max + 1

        t1 = time.time()
        cou_occ, cou_vir = self.get_D_list_parts(max)
        t2 = time.time()
        D_list_parts_time = t2 - t1
        D_list_parts_mem = tracemalloc.get_traced_memory()[1]
        print(f"Max memory for D list parts: {D_list_parts_mem / 1e6} MB")
        print(f"Time to get D list parts: {D_list_parts_time} seconds")
        # record metrics
        metrics.record("D_list_parts_time", D_list_parts_time)
        metrics.record("D_list_parts_mem", int(D_list_parts_mem))

        moments = np.zeros((self.nkpts,max, naux, naux), dtype = np.complex128)

        t3 = time.time()
        for q in kpts.loop(1, mpi=True):
            D_list = np.zeros((max, naux, naux),
                              dtype=np.complex128)
            for i in range(max):
                coeff = [binom(i, j) for j in range(i + 1)]
                for kj in kpts.loop(1): # Construct all powers of D matricies
                    ka = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
                    D_list[i] += util.einsum("m, mQR, mQR->QR", coeff,
                                             cou_occ[kj,:i + 1][::-1],
                                             cou_vir[ka,:i + 1]) # - scaling n^2_k N^2_orb

            D_list /= self.nkpts
            D_list = util.einsum("PQ,mQR->mPR", self.cou[q],D_list) # - scaling n_k N^3_orb
            D_list = mpi_helper.allreduce(D_list)
            moments[q] += D_list.copy()
            D_list *= 2.0
            for i in range(max):
                # Get the higher order moments
                moments[q][i] += util.einsum("mPQ, mQR -> PR", D_list[:i][::-1], moments[q][:i])  # - scaling n_k N^3_orb

            moments[q] = util.einsum("mPR, RS -> mPS", moments[q], self.cou[q])
            del D_list
        t4 = time.time()
        dd_moments_time = t4 - t3
        dd_moments_mem = tracemalloc.get_traced_memory()[1]
        print(f"Max memory for dd moments: {dd_moments_mem / 1e6} MB")
        print(f"Time to get dd moments: {dd_moments_time} seconds")
        # record metrics
        metrics.record("dd_moments_time", dd_moments_time)
        metrics.record("dd_moments_mem", int(dd_moments_mem))

        moments /= self.nkpts

        return moments, D_list_parts_time, D_list_parts_mem, dd_moments_time, dd_moments_mem


    @logging.with_timer("Self-energy moments")
    @logging.with_status("Constructing self-energy moments")
    def build_se_moments(self, mo_energy_g=None, mo_occ_g=None):
        """
        Build the moments of the self-energy via convolution.

        Parameters
        ----------
        W : numpy.ndarray
            Moments of the density-density response at each k-point.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy at each k-point.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """

        # t1 = time.time()
        W, D_list_parts_time, D_list_parts_mem, dd_moments_time, dd_moments_mem = self.build_dd_moments()
        # t2 = time.time()
        # build_all_dd_moments_time = t2 - t1
        # print(f"Time to build all dd moments: {build_all_dd_moments_time} seconds")
        # metrics.record("build_all_dd_moments_time", build_all_dd_moments_time)
        # also record the dd parts already returned
        metrics.record("D_list_parts_time", D_list_parts_time)
        metrics.record("D_list_parts_mem", int(D_list_parts_mem))
        metrics.record("dd_moments_time", dd_moments_time)
        metrics.record("dd_moments_mem", int(dd_moments_mem))

        # Get the orbitals
        if mo_energy_g is None:
            mo_energy_g = self.mo_energy_g
        if mo_occ_g is None:
            mo_occ_g = self.mo_occ_g

        # Setup dependent on diagonal SE and initialise moments
        if self.gw.diagonal_se:
            chars = ("p", "p", "p", lambda x: np.diag(x))
        else:
            chars = ("pq", "p", "q", lambda x: x)

        tracemalloc.reset_peak()
        #Convolve the occupied and virtual moments separately
        t3 = time.time()
        moments_occ = self.convolve_occ(W, mo_energy_g, mo_occ_g, chars)
        t4 = time.time()
        occ_convolution_time = t4 - t3
        occ_convolution_mem = tracemalloc.get_traced_memory()[1]
        print(f"Time for occupied convolution: {occ_convolution_time} seconds")
        print(f"Max memory for occupied convolution: {occ_convolution_mem / 1e6} MB")
        metrics.record("occ_convolution_time", occ_convolution_time)
        metrics.record("occ_convolution_mem", int(occ_convolution_mem))
        tracemalloc.reset_peak()
        moments_vir = self.convolve_vir(W, mo_energy_g, mo_occ_g, chars)
        t5 = time.time()
        vir_convolution_time = t5 - t4
        vir_convolution_mem = tracemalloc.get_traced_memory()[1]
        print(f"Time for virtual convolution: {vir_convolution_time} seconds")
        print(f"Max memory for virtual convolution: {vir_convolution_mem / 1e6} MB")
        metrics.record("vir_convolution_time", vir_convolution_time)
        metrics.record("vir_convolution_mem", int(vir_convolution_mem))
        metrics.record("convolution_time", occ_convolution_time + vir_convolution_time)

        return moments_occ, moments_vir


    # def build_se_moments(self):
    #     """
    #     Build the moments of the self-energy via convolution with
    #     tensor-hypercontraction in k-space.
    #
    #     Parameters
    #     ----------
    #     zeta : numpy.ndarray
    #         Moments of the density-density response.
    #
    #     Returns
    #     -------
    #     moments_occ : numpy.ndarray
    #         Moments of the occupied self-energy.
    #     moments_vir : numpy.ndarray
    #         Moments of the virtual self-energy.
    #     """
    #
    #     zeta = 2*self.build_dd_moments()
    #
    #     kpts = self.kpts
    #
    #     if self.gw.diagonal_se:
    #         pqchar = pchar = qchar = "p"
    #         eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo)
    #     else:
    #         pqchar, pchar, qchar = "pq", "p", "q"
    #         eta_shape = lambda k: (self.mo_energy_g[k].size, self.nmom_max + 1, self.nmo, self.nmo)
    #     eta = np.zeros((self.nkpts, self.nkpts), dtype=object)
    #
    #     # Get the moments in (aux|aux) and rotate to (mo|mo)
    #     for i in range(self.nmom_max + 1):
    #         for q in kpts.loop(1):
    #             zeta_prime = zeta[q][i]
    #             # for kj in kpts.loop(1):
    #             #     kb = kpts.member(kpts.wrap_around(kpts[q] + kpts[kj]))
    #             #     zeta_prime += np.linalg.multi_dot((self.cou[q], zeta[q, kb, i], self.cou[q]))
    #             zeta_prime *= 2.0
    #             zeta_prime /= self.nkpts
    #
    #             for kp in range(self.nkpts):
    #                 kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))
    #
    #                 if not isinstance(eta[kp, q], np.ndarray):
    #                     eta[kp, q] = np.zeros(eta_shape(kx), dtype=zeta_prime.dtype)
    #
    #                 for x in range(self.mo_energy_g[kx].size):
    #                     Lpx = util.einsum(
    #                         "Pp,P->Pp", self.integrals.Lp[kp], self.integrals.Lx[kx][:, x]
    #                     )
    #                     subscript = f"P{pchar},Q{qchar},PQ->{pqchar}"
    #                     eta[kp, q][x, i] += util.einsum(subscript, Lpx, Lpx.conj(), zeta_prime)
    #
    #     # Construct the self-energy moments
    #     moments_occ, moments_vir = self.convolve(eta)
    #
    #     return moments_occ, moments_vir

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
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy at each k-point.
        """
        # TODO fix this doc string
        (pqchar, pchar, qchar, fproc) = chars

        kpts = self.kpts
        max_nmom = self.nmom_max + 1
        nmo = self.Lp[0].shape[-1]
        moms = np.arange(max_nmom)

        moments_occ = np.zeros((self.nkpts, max_nmom, nmo, nmo), dtype=complex)
        g_cou_occ = np.zeros((self.nkpts, max_nmom, self.naux, self.naux),
                             dtype=np.complex128)
        t1 = time.time()
        for kx in self.kpts.loop(1, mpi=True):
            eo = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] > 0], moms) * 2.0
            g_cou_occ[kx] = util.einsum("xm,Px,Qx->mPQ", eo, self.Lx[kx][:, mo_occ_g[kx] > 0].conj(), self.Lx[kx][:, mo_occ_g[kx] > 0])
        t2 = time.time()
        g_cou_occ_time = t2 - t1
        g_cou_occ_mem = tracemalloc.get_traced_memory()[1]
        print(f"Time to build g_cou_occ: {g_cou_occ_time} seconds")
        print(f"Max memory for g_cou_occ: {g_cou_occ_mem / 1e6} MB")
        metrics.record("g_cou_occ_time", g_cou_occ_time)
        metrics.record("g_cou_occ_mem", int(g_cou_occ_mem))

        for n in moms:
            eta_orders = np.arange(n + 1)
            fh = binom(n, eta_orders) * (-1) ** eta_orders
            for kp in kpts.loop(1, mpi=True):
                GW_occ = np.zeros((self.naux, self.naux), dtype=complex)
                for q in kpts.loop(1):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))
                    GW_occ += util.einsum("t,tPQ,tPQ->PQ", fh, W[q, :n+1], g_cou_occ[kx, :n+1][::-1])
                moments_occ[kp, n] = util.einsum(f"P{pchar},PQ,Q{qchar}->{pqchar}", self.Lp[kp],
                                                GW_occ, self.Lp[kp].conj())

                # Numerical integration can lead to small non-hermiticity
                moments_occ[kp, n] = 0.5 * (moments_occ[kp, n] + moments_occ[kp, n].T.conj())

        moments_occ = mpi_helper.allreduce(moments_occ)
        return moments_occ

    @logging.with_timer("Virtual moment convolution")
    @logging.with_status("Convoluting virtual moments")
    def convolve_vir(self, W, mo_energy_g, mo_occ_g, chars):
        (pqchar, pchar, qchar, fproc) = chars

        kpts = self.kpts
        max_nmom = self.nmom_max + 1
        moms = np.arange(max_nmom)
        nmo = self.Lp[0].shape[-1]

        moments_vir = np.zeros((self.nkpts, max_nmom, nmo, nmo), dtype=complex)
        g_cou_vir = np.zeros((self.nkpts, max_nmom, self.naux, self.naux),
                             dtype=np.complex128)

        t1 = time.time()
        # Build G_aux_aux - scaling n_k N^3_orb
        for kx in self.kpts.loop(1, mpi=True):
            ev = np.power.outer(mo_energy_g[kx][mo_occ_g[kx] == 0], moms) * 2.0
            g_cou_vir[kx] = util.einsum("xm,Px,Qx->mPQ", ev, self.Lx[kx][:, mo_occ_g[kx] == 0].conj(),self.Lx[kx][:, mo_occ_g[kx] == 0])
        t2 = time.time()
        g_cou_vir_time = t2 - t1
        g_cou_vir_mem = tracemalloc.get_traced_memory()[1]
        print(f"Time to build g_cou_vir: {g_cou_vir_time} seconds")
        print(f"Max memory for g_cou_vir: {g_cou_vir_mem / 1e6} MB")
        metrics.record("g_cou_vir_time", g_cou_vir_time)
        metrics.record("g_cou_vir_mem", int(g_cou_vir_mem))

        for n in moms:
            eta_orders = np.arange(n + 1)
            fp = binom(n, eta_orders)
            for kp in kpts.loop(1, mpi=True):
                GW_vir = np.zeros((self.naux, self.naux), dtype=complex)
                for q in kpts.loop(1):
                    kx = kpts.member(kpts.wrap_around(kpts[kp] - kpts[q]))
                    GW_vir += util.einsum("t,tPQ,tPQ->PQ", fp, W[q, :n+1], g_cou_vir[kx, :n+1][::-1])
                moments_vir[kp, n] = util.einsum(f"P{pchar},PQ,Q{qchar}->{pqchar}", self.Lp[kp],
                                               GW_vir, self.Lp[kp].conj())
                # Numerical integration can lead to small non-hermiticity
                moments_vir[kp, n] = 0.5 * (moments_vir[kp, n] + moments_vir[kp, n].T.conj())

            moments_vir = mpi_helper.allreduce(moments_vir)

        return moments_vir

