"""Spin-restricted one-shot GW via self-energy moment constraints for molecular systems."""

import numpy as np
from dyson import MBLSE, Lehmann, Spectral

from momentGW import energy, logging, thc, util
from momentGW.base import BaseGW
from momentGW.fock import FockLoop, search_chempot
from momentGW.ints import Integrals
from momentGW.rpa import dRPA
from momentGW.tda import dTDA


def nelec_tolerance(gw, fock_loop):
    """Return the tolerance the particle-number error is judged against.

    A one-shot calculation only searches for a chemical potential, so its
    particle-number error is limited by where the poles fall. Shifting the
    self-energy poles or running the Fock loop makes the particle number
    something the calculation actually solves for, and the error is held to
    the Fock loop's tolerance instead.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    fock_loop : momentGW.fock.FockLoop
        Fock loop solver. Its tolerance is read from the solver rather than
        from `gw.fock_opts`, which may override only some of the defaults.

    Returns
    -------
    tol : float
        Tolerance on the error in the number of electrons.
    """
    if gw.fock_loop or gw.optimise_chempot:
        return fock_loop.conv_tol_nelec
    return 1e-1


def realization_record(solver, se_moments):
    """Summarise what a Dyson solver realized, as distinct from what it was asked for.

    Dyson stops the recurrence when an iteration produces an off-diagonal
    square with no real square root, and solves at the last order it
    completed. Everything here is read at the achieved order, because that is
    the realization the Dyson solve used; the requested order is recorded
    beside it, not in place of it.

    Parameters
    ----------
    solver : dyson.MBLSE
        Solver, after `kernel` has been called.
    se_moments : numpy.ndarray
        Moments the solver was handed.

    Returns
    -------
    record : dict
        Realization diagnostics for one sector.
    """
    if solver.max_cycle_achieved is None:
        achieved = solver.max_cycle
    else:
        achieved = solver.max_cycle_achieved

    return {
        "moments_supplied": int(np.asarray(se_moments).shape[0]),
        "max_cycle": int(solver.max_cycle),
        "max_cycle_achieved": int(achieved),
        "order_reduced": int(achieved) != int(solver.max_cycle),
        "nmom_conserved_requested": int(solver.nmom_conserved(solver.max_cycle)),
        "nmom_conserved_achieved": int(solver.nmom_conserved(achieved)),
        "n_poles": int(np.asarray(solver.result.eigvals).size),
        "errors": solver.moment_errors() if solver.calculate_errors else None,
    }


def kernel(
    gw,
    nmom_max,
    moments=None,
    integrals=None,
):
    """Moment-constrained one-shot GW.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    moments : tuple of numpy.ndarray, optional
        Tuple of (hole, particle) moments, if passed then they will
        be used instead of calculating them. Default value is `None`.
    integrals : BaseIntegrals, optional
        Integrals object. If `None`, generate from scratch. Default
        value is `None`.

    Returns
    -------
    conv : bool
        Convergence flag. A one-shot calculation has no outer loop, so
        this reports whether the numerical gates in
        `gw.dyson_diagnostics` passed: the realization delivered every
        moment order it was asked for, and the particle-number error is
        within tolerance.
    gf : dyson.Lehmann
        Green's function object.
    se : dyson.Lehmann
        Self-energy object.
    qp_energy : numpy.ndarray
        Quasiparticle energies. Always `None` for GW, returned for
        compatibility with other GW methods.

    Notes
    -----
    This approach is described in [1]_.

    References
    ----------
    .. [1] C. J. C. Scott, O. J. Backhouse, and G. H. Booth, 158, 12,
        2023.
    """

    # Get the integrals
    if integrals is None:
        integrals = gw.ao2mo()

    # Get the static part of the SE
    se_static = gw.build_se_static(integrals)

    # Get the moments of the SE
    if moments is None:
        th, tp = gw.build_se_moments(
            nmom_max,
            integrals,
            mo_energy=dict(
                g=gw.mo_energy,
                w=gw.mo_energy,
            ),
        )
    else:
        th, tp = moments

    # Solve the Dyson equation
    gf, se = gw.solve_dyson(th, tp, se_static, integrals=integrals)

    # A single-shot calculation has no outer loop to converge, but it can still fail
    # its numerical gates: report those rather than an unconditional `True`. The
    # unrestricted and periodic solvers share this kernel but override `solve_dyson`
    # without gating it yet, and keep the old unconditional flag until they do.
    if gw.dyson_diagnostics is None:
        conv = True
    else:
        conv = gw.dyson_diagnostics["converged"]

    return conv, gf, se, None


class GW(BaseGW):
    """Spin-restricted one-shot GW via self-energy moment constraints for molecules.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field class.
    diagonal_se : bool, optional
        If `True`, use a diagonal approximation in the self-energy.
        Default value is `False`.
    polarizability : str, optional
        Type of polarizability to use, can be one of `("drpa",
        "drpa-exact", "dtda", "thc-dtda"). Default value is `"drpa"`.
    npoints : int, optional
        Number of numerical integration points. Default value is `48`.
    optimise_chempot : bool, optional
        If `True`, optimise the chemical potential by shifting the
        position of the poles in the self-energy relative to those in
        the Green's function. Default value is `False`.
    fock_loop : bool, optional
        If `True`, self-consistently renormalise the density matrix
        according to the updated Green's function. Default value is
        `False`.
    fock_opts : dict, optional
        Dictionary of options passed to the Fock loop. For more details
        see `momentGW.fock`.
    compression : str, optional
        Blocks of the ERIs to use as a metric for compression. Can be
        one or more of `("oo", "ov", "vv", "ia")` which can be passed as
        a comma-separated string. `"oo"`, `"ov"` and `"vv"` refer to
        compression on the initial ERIs, whereas `"ia"` refers to
        compression on the ERIs entering RPA, which may change under a
        self-consistent scheme. Default value is `"ia"`.
    compression_tol : float, optional
        Tolerance for the compression. Default value is `1e-10`.
    thc_opts : dict, optional
        Dictionary of options to be used for THC calculations. Current
        implementation requires a filepath to import the THC integrals.

    Notes
    -----
    This approach is described in [1]_.

    References
    ----------
    .. [1] C. J. C. Scott, O. J. Backhouse, and G. H. Booth, 158, 12,
        2023.
    """

    _kernel = kernel

    @property
    def name(self):
        """Get the method name."""
        polarizability = self.polarizability.upper().replace("DTDA", "dTDA").replace("DRPA", "dRPA")
        return f"{polarizability}-G0W0"

    @logging.with_timer("Static self-energy")
    @logging.with_status("Building static self-energy")
    def build_se_static(self, integrals):
        """Build the static part of the self-energy, including the Fock matrix.

        Parameters
        ----------
        integrals : Integrals
            Integrals object.

        Returns
        -------
        se_static : numpy.ndarray
            Static part of the self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        """

        # Get intermediates
        mask = self.active
        dm = self._scf.make_rdm1(mo_coeff=self._mo_coeff)

        # Get the contribution from the exchange-correlation potential
        if getattr(self._scf, "xc", "hf") == "hf":
            se_static = np.zeros_like(dm)
            se_static = se_static[..., mask, :][..., :, mask]
        else:
            with util.SilentSCF(self._scf):
                veff = self._scf.get_veff(None, dm)[..., mask, :][..., :, mask]
                vj = self._scf.get_j(None, dm)[..., mask, :][..., :, mask]

            vhf = integrals.get_veff(dm, j=vj, basis="ao")
            se_static = vhf - veff
            se_static = util.einsum(
                "...pq,...pi,...qj->...ij", se_static, np.conj(self.mo_coeff), self.mo_coeff
            )

        # If diagonal approximation, set non-diagonal elements to zero
        if self.diagonal_se:
            se_static = util.einsum("...pq,pq->...pq", se_static, np.eye(se_static.shape[-1]))

        # Add the Fock matrix contribution
        se_static += util.einsum("...p,...pq->...pq", self.mo_energy, np.eye(se_static.shape[-1]))

        return se_static

    def build_se_moments(self, nmom_max, integrals, **kwargs):
        """Build the moments of the self-energy.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        integrals : Integrals
            Integrals object.
        **kwargs : dict, optional
           Additional keyword arguments passed to polarizability class.

        Returns
        -------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy. If `self.diagonal_se`,
            non-diagonal elements are set to zero.

        See Also
        --------
        momentGW.rpa.dRPA
        momentGW.tda.dTDA
        momentGW.thc.dTDA
        """

        if self.polarizability.lower() == "drpa":
            rpa = dRPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel()

        elif self.polarizability.lower() == "drpa-exact":
            rpa = dRPA(self, nmom_max, integrals, **kwargs)
            return rpa.kernel(exact=True)

        elif self.polarizability.lower() == "dtda":
            tda = dTDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()

        elif self.polarizability.lower() == "thc-dtda":
            tda = thc.dTDA(self, nmom_max, integrals, **kwargs)
            return tda.kernel()

        else:
            raise NotImplementedError

    @logging.with_timer("Integral construction")
    @logging.with_status("Constructing integrals")
    def ao2mo(self, transform=True):
        """Get the integrals object.

        Parameters
        ----------
        transform : bool, optional
            Whether to transform the integrals object.

        Returns
        -------
        integrals : Integrals
            Integrals object.

        See Also
        --------
        momentGW.ints.Integrals
        momentGW.thc.Integrals
        """

        # Get the integrals class
        if self.polarizability.lower().startswith("thc"):
            cls = thc.Integrals
            kwargs = self.thc_opts
        else:
            cls = Integrals
            kwargs = dict(
                compression=self.compression,
                compression_tol=self.compression_tol,
                # Note: `pyscf.pbc.df` methods don't use `self.prange`
                # so the MPI solution won't work. Storing the full
                # tensor is a workaround.
                store_full=self.fock_loop or hasattr(self.with_df, "kpts"),
            )

        # Get the integrals
        integrals = cls(
            self.with_df,
            self.mo_coeff,
            self.mo_occ,
            **kwargs,
        )

        # Transform the integrals
        if transform:
            integrals.transform()

        return integrals

    def solve_dyson(self, se_moments_hole, se_moments_part, se_static, integrals=None):
        """Solve the Dyson equation due to a self-energy resulting from a list of hole and particle
        moments, along with a static contribution.

        Also finds a chemical potential best satisfying the physical
        number of electrons. If `self.optimise_chempot`, this will
        shift the self-energy poles relative to the Green's function,
        which is a partial self-consistency that better conserves the
        particle number.

        If `self.fock_loop`, this function will also require that the
        outputted Green's function is self-consistent with respect to
        the corresponding density and Fock matrix.

        Parameters
        ----------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy.
        se_static : numpy.ndarray
            Static part of the self-energy.
        integrals : Integrals
            Integrals object. Required if `self.fock_loop` is `True`.
            Default value is `None`.

        Returns
        -------
        gf : dyson.Lehmann
            Green's function object.
        se : dyson.Lehmann
            Self-energy object.

        See Also
        --------
        momentGW.fock.FockLoop
        """

        # Solve the Dyson equation for the moments
        with logging.with_modifiers(status="Solving Dyson equation", timer="Dyson equation"):
            solver_occ = MBLSE(se_static, np.array(se_moments_hole), **self.dyson_opts)
            solver_occ.kernel()

            solver_vir = MBLSE(se_static, np.array(se_moments_part), **self.dyson_opts)
            solver_vir.kernel()

            result = Spectral.combine_for_self_energy(solver_occ.result, solver_vir.result)
            se = result.get_self_energy()

        # Record what each sector realized. The solvers are the only place this is
        # known, so it is read off here rather than rebuilt by whoever wants it.
        realization = {
            "hole": realization_record(solver_occ, se_moments_hole),
            "particle": realization_record(solver_vir, se_moments_part),
        }
        for sector, record in realization.items():
            if record["order_reduced"]:
                logging.warn(
                    f"[red]Realization stepped down[/] ({sector}): conserving "
                    f"{record['nmom_conserved_achieved']} of "
                    f"{record['nmom_conserved_requested']} moments"
                )

        # Initialise the solver
        solver = FockLoop(self, se=se, **self.fock_opts)

        # Shift the self-energy poles relative to the Green's function
        # to better conserve the particle number
        if self.optimise_chempot:
            se = solver.auxiliary_shift(se_static)

        # Find the error in the moments
        moment_error = self.moment_error(se_moments_hole, se_moments_part, se)
        logging.write(
            f"Error in moments:  "
            f"[{logging.rate(sum(moment_error), 1e-12, 1e-8)}]{sum(moment_error):.3e}[/] "
            f"(hole = [{logging.rate(moment_error[0], 1e-12, 1e-8)}]{moment_error[0]:.3e}[/], "
            f"particle = [{logging.rate(moment_error[1], 1e-12, 1e-8)}]{moment_error[1]:.3e}[/])"
        )

        # Solve the Dyson equation for the self-energy
        gf, error = solver.solve_dyson(se_static, se=se)
        chempot = gf.chempot
        se = se.copy(chempot=chempot)

        # Self-consistently renormalise the density matrix
        fock_conv = None
        if self.fock_loop:
            logging.write("")
            solver.gf = gf
            solver.se = se
            fock_conv, gf, se = solver.kernel(integrals=integrals)
            _, error = solver.search_chempot(gf)

        # Print the error in the number of electrons
        nelec_tol = nelec_tolerance(self, solver)
        logging.write("")
        style = logging.rate(abs(error), 1e-6, nelec_tol)
        logging.write(f"Error in number of electrons:  [{style}]{error:.3e}[/]")
        logging.write(f"Chemical potential:  {gf.chempot:.6f}")

        # Record the gates the calculation is judged against, so that the caller can
        # ask whether it converged instead of assuming that it did
        gates = {
            "realization": not any(r["order_reduced"] for r in realization.values()),
            "nelec": bool(abs(error) <= nelec_tol),
        }
        if fock_conv is not None:
            gates["fock_loop"] = bool(fock_conv)
        self.dyson_diagnostics = {
            "realization": realization,
            "moment_error": {"hole": moment_error[0], "particle": moment_error[1]},
            "nelec_error": float(error),
            "nelec_tol": float(nelec_tol),
            "chempot": float(gf.chempot),
            "gates": gates,
            "converged": all(gates.values()),
        }

        return gf, se

    def kernel(
        self,
        nmom_max,
        moments=None,
        integrals=None,
    ):
        """Driver for the method.

        Parameters
        ----------
        nmom_max : int
            Maximum moment number to calculate.
        moments : tuple of numpy.ndarray, optional
            Tuple of (hole, particle) moments, if passed then they will
            be used instead of calculating them. Default value is
            `None`.
        integrals : Integrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        converged : bool
            Whether the solver converged. For single-shot calculations,
            this is whether the numerical gates in `dyson_diagnostics`
            passed.
        gf : dyson.Lehmann
            Green's function object.
        se : dyson.Lehmann
            Self-energy object.
        qp_energy : NoneType
            Quasiparticle energies. For most GW methods, this is `None`.
        """
        return super().kernel(nmom_max, moments=moments, integrals=integrals)

    def make_rdm1(self, gf=None):
        """Get the first-order reduced density matrix.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function object. If `None`, use either `self.gf`, or
            the mean-field Green's function. Default value is `None`.

        Returns
        -------
        rdm1 : numpy.ndarray
            First-order reduced density matrix.
        """

        # Get the Green's function
        if gf is None:
            gf = self.gf
        if gf is None:
            gf = self.init_gf()

        return gf.occupied().moment(0) * 2.0

    def moment_error(self, se_moments_hole, se_moments_part, se):
        """Return the error in the moments.

        Parameters
        ----------
        se_moments_hole : numpy.ndarray
            Moments of the hole self-energy.
        se_moments_part : numpy.ndarray
            Moments of the particle self-energy.
        se : dyson.Lehmann
            Self-energy object.

        Returns
        -------
        eh : float
            Error in the hole moments.
        ep : float
            Error in the particle moments.
        """
        eh = self._moment_error(
            se_moments_hole,
            se.occupied().moment(range(len(se_moments_hole))),
        )
        ep = self._moment_error(
            se_moments_part,
            se.virtual().moment(range(len(se_moments_part))),
        )
        return eh, ep

    def energy_nuc(self):
        """Calculate the nuclear repulsion energy.

        Returns
        -------
        e_nuc : float
            Nuclear repulsion energy.
        """
        with util.SilentSCF(self._scf):
            return self._scf.energy_nuc()

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_hf(self, gf=None, integrals=None):
        """Calculate the one-body (Hartree--Fock) energy.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function object. If `None`, use either `self.gf`, or
            the mean-field Green's function. Default value is `None`.
        integrals : Integrals, optional
            Integrals object. If `None`, generate from scratch. Default
            value is `None`.

        Returns
        -------
        e_1b : float
            One-body energy.
        """

        # Get the Green's function
        if gf is None:
            gf = self.gf

        # Get the integrals
        if integrals is None:
            integrals = self.ao2mo()

        # Find the Fock matrix
        with util.SilentSCF(self._scf):
            h1e = util.einsum(
                "pq,pi,qj->ij", self._scf.get_hcore(), self.mo_coeff.conj(), self.mo_coeff
            )
        rdm1 = self.make_rdm1(gf=gf)
        fock = integrals.get_fock(rdm1, h1e)

        return energy.hartree_fock(rdm1, fock, h1e)

    @logging.with_timer("Energy")
    @logging.with_status("Calculating energy")
    def energy_gm(self, gf=None, se=None, g0=True):
        r"""Calculate the two-body (Galitskii--Migdal) energy.

        Parameters
        ----------
        gf : dyson.Lehmann, optional
            Green's function object. If `None`, use `self.gf`. Default
            value is `None`.
        se : dyson.Lehmann, optional
            Self-energy object. If `None`, use `self.se`. Default value
            is `None`.
        g0 : bool, optional
            If `True`, use the mean-field Green's function. Default
            value is `True`.

        Returns
        -------
        e_2b : float
            Two-body energy.
        """

        # Get the Green's function and self-energy
        if gf is None:
            gf = self.gf
        if se is None:
            se = self.se

        # Calculate the Galitskii--Migdal energy
        if g0:
            e_2b = energy.galitskii_migdal_g0(self.mo_energy, self.mo_occ, se)
        else:
            e_2b = energy.galitskii_migdal(gf, se)

        return e_2b

    def init_gf(self, mo_energy=None):
        """Initialise the mean-field Green's function.

        Parameters
        ----------
        mo_energy : numpy.ndarray, optional
            Molecular orbital energies. Default value is
            `self.mo_energy`.

        Returns
        -------
        gf : dyson.Lehmann
            Mean-field Green's function.
        """

        # Get the MO energies
        if mo_energy is None:
            mo_energy = self.mo_energy

        # Build the Green's function
        gf = Lehmann(mo_energy, np.eye(self.nmo))

        # Find the chemical potential
        chempot = search_chempot(gf.energies, gf.couplings, self.nmo, self.nocc * 2)[0]
        gf = gf.copy(chempot=chempot)

        return gf
