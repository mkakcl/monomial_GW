"""Construct RPA moments."""

import numpy as np
import scipy.linalg
import scipy.optimize

from momentGW import eta0, logging, mpi_helper, util
from momentGW.tda import dTDA


class dRPA(dTDA):
    """Compute the self-energy moments using dRPA and numerical integration.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : BaseIntegrals
        Integrals object.
    mo_energy : dict, optional
        Molecular orbital energies. Keys are "g" and "w" for the Green's
        function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_energy` for both. Default value is `None`.
    mo_occ : dict, optional
        Molecular orbital occupancies. Keys are "g" and "w" for the
        Green's function and screened Coulomb interaction, respectively.
        If `None`, use `gw.mo_occ` for both. Default value is `None`.

    Notes
    -----
    See `momentGW.tda.dTDA.__init__` for initialisation details and
    `momentGW.tda.dTDA.kernel` for calculation run details.
    """

    @logging.with_timer("Numerical integration")
    @logging.with_status("Performing numerical integration")
    def build_zeroth_dd_moment(self, m0=None):
        """Build the zeroth moment of the density-density response.

        Dispatches on `gw.eta0_method`: `"clencur"` optimises the legacy
        Clenshaw-Curtis quadrature and performs the numerical integration,
        while `"hht"` applies a certified rational approximation of the
        inverse square root (see `momentGW.eta0`).  Both produce the same
        RI-projected object.

        Returns
        -------
        zeroth moment : numpy.ndarray
            Zeroth moment of the density-density response.
        """

        method = getattr(self.gw, "eta0_method", "clencur")
        if method == "hht":
            return self._build_zeroth_dd_moment_hht()
        if method != "clencur":
            raise ValueError(f"Unknown eta0_method {method!r}: expected one of ('clencur', 'hht')")

        p0, p1 = self.mpi_slice(self.nov)

        # Construct energy differences
        d_full = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()

        # Calculate diagonal part of ERI
        diag_eri = np.zeros((self.nov,))
        diag_eri[p0:p1] = util.einsum("np,np->p", self.integrals.Lia, self.integrals.Lia)
        diag_eri = mpi_helper.allreduce(diag_eri)

        # Get the main integral quadrature
        quad = self.optimise_main_quad(d_full, diag_eri)

        # Perform the main integral
        integral = self.eval_main_integral(quad)

        # Report quadrature error
        if self.report_quadrature_error:
            a = np.sum((integral[0] - integral[2]) ** 2)
            b = np.sum((integral[0] - integral[1]) ** 2)
            a, b = mpi_helper.allreduce(np.array([a, b]))
            a, b = a**0.5, b**0.5
            err = self.estimate_error_clencur(a, b)
            style_half = logging.rate(a, 1e-4, 1e-3)
            style_quar = logging.rate(b, 1e-8, 1e-6)
            style_full = logging.rate(err, 1e-12, 1e-9)
            logging.write(
                f"Error in integral:  [{style_full}]{err:.3e}[/] "
                f"(half = [{style_half}]{a:.3e}[/], quarter = [{style_quar}]{b:.3e}[/])",
            )
            integral = np.delete(integral, [1, 2], 0)
        return integral[0]

    def _hht_apply(self, shifts, weights, d, Lia, collect_residuals=False):
        """Apply the rational approximation of the inverse square root through Woodbury.

        Each pole contributes ``w_j (I + 4 G_j)^{-1} Y_j`` with
        ``Y_j = Lia * f_j`` and the auxiliary-space Gram
        ``G_j = Y_j Lia^T``, where ``f_j = d / (d^2 + s_j)``.  Only the Gram
        is reduced across ranks; no particle-hole squared intermediate is
        formed, and there is no bare-`Lia` term to cancel against.

        Parameters
        ----------
        shifts : numpy.ndarray
            Pole shifts.
        weights : numpy.ndarray
            Weights.
        d : numpy.ndarray
            Local slice of the particle-hole energy differences.
        Lia : numpy.ndarray
            Local slice of the ``(aux, W occ, W vir)`` array.
        collect_residuals : bool, optional
            Whether to compute the relative residual of every
            auxiliary-space Cholesky solve.  Default value is `False`.

        Returns
        -------
        integral : numpy.ndarray
            RI-projected zeroth moment contribution, shape
            ``(naux, nov_local)``.
        residuals : list of float
            Per-pole relative solve residuals (empty unless
            `collect_residuals`).
        """

        naux = Lia.shape[0]
        integral = np.zeros_like(Lia)
        eye = np.eye(naux)
        d_sq = d * d
        residuals = []

        for shift, weight in zip(shifts, weights):
            f = d / (d_sq + shift)
            Y = Lia * f[None]
            gram = np.dot(Y, Lia.T)
            gram = mpi_helper.allreduce(gram)
            A = eye + 4.0 * gram
            cho = scipy.linalg.cho_factor(A)
            X = scipy.linalg.cho_solve(cho, Y)

            if collect_residuals:
                norms = np.array([np.sum((np.dot(A, X) - Y) ** 2), np.sum(Y * Y)])
                norms = mpi_helper.allreduce(norms)
                residuals.append(float(np.sqrt(norms[0] / norms[1])) if norms[1] > 0 else 0.0)

            integral += weight * X

        return integral, residuals

    def _estimate_mtilde_lambda_max(self, d, Lia, n_iter=12):
        """Estimate the largest eigenvalue of ``Mtilde`` by power iteration.

        Reported only to show how loose the rigorous upper bound of the
        certified interval is; it plays no role in the certificate itself.
        The action ``v -> d^2 v + 4 W (W^T v)`` is evaluated without forming
        any particle-hole squared matrix.

        Parameters
        ----------
        d : numpy.ndarray
            Local slice of the particle-hole energy differences.
        Lia : numpy.ndarray
            Local slice of the ``(aux, W occ, W vir)`` array.
        n_iter : int, optional
            Number of power iterations.  Default value is `12`.

        Returns
        -------
        lambda_max : float
            Estimate of (and lower bound on) the largest eigenvalue.
        """

        p0, p1 = self.mpi_slice(self.nov)
        sqrt_d = np.sqrt(d)

        # Deterministic start vector, replicated identically on every rank
        v = np.random.default_rng(0).standard_normal(self.nov)
        v /= np.linalg.norm(v)

        lambda_max = 0.0
        for _ in range(n_iter):
            # W^T v, reduced over the distributed particle-hole index
            t = np.dot(Lia * sqrt_d[None], v[p0:p1])
            t = mpi_helper.allreduce(t)

            # Local rows of Mtilde v, then reassemble the replicated vector
            u = np.zeros_like(v)
            u[p0:p1] = d * d * v[p0:p1] + 4.0 * sqrt_d * np.dot(t, Lia)
            u = mpi_helper.allreduce(u)

            lambda_max = float(np.dot(v, u))
            norm = np.linalg.norm(u)
            if norm == 0:
                break
            v = u / norm

        return lambda_max

    @logging.with_status("Performing certified rational approximation")
    def _build_zeroth_dd_moment_hht(self):
        """Build the zeroth moment through a certified HHT inverse square root.

        Computes the same RI-projected object as the Clenshaw-Curtis route,
        ``eta0 V = D^{1/2} Mtilde^{-1/2} W``, through the rational
        approximation layer in `momentGW.eta0`: a rigorous spectral enclosure
        of ``Mtilde``, pole selection against `gw.eta0_tol` certified by the
        measured scalar error, and one auxiliary-space Cholesky solve per
        pole with its residual checked.  Structured diagnostics are stored on
        `gw.eta0_diagnostics`.

        Returns
        -------
        integral : numpy.ndarray
            Zeroth moment of the density-density response, shape
            ``(naux, nov_local)``.
        """

        p0, p1 = self.mpi_slice(self.nov)
        naux = self.naux

        # Restricted molecular scope only.  The unrestricted and periodic
        # solvers override `build_zeroth_dd_moment` wholesale and never reach
        # this dispatch; the guard is here so a refactor cannot silently
        # widen the scope.
        if type(self) is not dRPA:
            raise NotImplementedError(
                "eta0_method='hht' is validated for the restricted molecular dRPA only"
            )

        # An empty particle-hole space has nothing to screen
        if self.nov == 0:
            return np.zeros((naux, 0))

        if self.d is None:
            self._build_d()
        d = self.d
        Lia = self.integrals.Lia

        # Full gap vector, replicated on every rank as in the legacy route
        d_full = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()

        # Rigorous norm bounds on the coupling term 4 W W^T with
        # W = D^{1/2} V:  ||W W^T||_2 <= min(||W||_F^2, ||W||_1 ||W||_inf)
        sqrt_d = np.sqrt(d)
        w_frob_sq = float(mpi_helper.allreduce(np.asarray(np.sum(d * np.sum(Lia * Lia, axis=0)))))
        col_sums = mpi_helper.allreduce(np.sum(np.abs(Lia) * sqrt_d[None], axis=1))
        w_one = float(np.max(col_sums)) if col_sums.size else 0.0
        row_sums = sqrt_d * np.sum(np.abs(Lia), axis=0)
        w_inf_local = float(np.max(row_sums)) if row_sums.size else 0.0
        w_inf = float(
            mpi_helper.allreduce(np.asarray(w_inf_local), op=getattr(mpi_helper.mpi, "MAX", None))
        )
        coupling_bound = 4.0 * min(w_frob_sq, w_one * w_inf)

        # Certified spectral enclosure and pole selection against the
        # requested tolerance.  A fixed pole count is an instruction, so a
        # tolerance miss there is reported, not raised.
        lmin, lmax = eta0.certified_interval(d_full, coupling_bound)
        # Milestone 3.1: where a frontier accuracy is requested, the tolerance is derived
        # from it rather than given, and the pole count then follows from the tolerance.
        moment_tol = getattr(self.gw, "moment_tol", None)
        qp_tol = getattr(self.gw, "qp_tol", None)
        if moment_tol is not None and qp_tol is not None:
            raise ValueError(
                "moment_tol and qp_tol both set; they derive the same eta0 tolerance "
                "from different targets, so one of them would be silently ignored"
            )
        if qp_tol is not None:
            tol, clamped = eta0.eta0_tol_for_qp_tol(qp_tol)
            if clamped:
                logging.warn(
                    f"Requested qp_tol = {qp_tol:.3e} eV implies an eta0 tolerance below "
                    f"[bad]{eta0.ETA0_TOL_FLOOR:.3e}[/]; clamped, so the frontier is not "
                    f"bounded at the requested accuracy"
                )
        elif moment_tol is not None:
            tol, clamped = eta0.eta0_tol_for_moment_tol(moment_tol)
            if clamped:
                logging.warn(
                    f"Requested moment_tol = {moment_tol:.3e} implies an eta0 tolerance "
                    f"below [bad]{eta0.ETA0_TOL_FLOOR:.3e}[/]; clamped, so the moments are "
                    f"not bounded at the requested accuracy"
                )
        else:
            tol = self.gw.eta0_tol
        n_poles_fixed = self.gw.eta0_n_poles
        shifts, weights, delta, n_estimate = eta0.select_poles(
            lmin, lmax, tol, n_poles=n_poles_fixed
        )
        if n_poles_fixed is not None and delta > tol:
            logging.warn(
                f"Fixed eta0_n_poles = {n_poles_fixed} achieves scalar error "
                f"[bad]{delta:.3e}[/] against tolerance {tol:.3e}"
            )

        # Apply the rational approximation, checking every solve
        integral, residuals = self._hht_apply(shifts, weights, d, Lia, collect_residuals=True)

        if integral.shape != (naux, p1 - p0):
            raise RuntimeError(
                f"Unexpected local eta0 shape {integral.shape}: expected {(naux, p1 - p0)}"
            )

        # Optional secondary regression signal: repeat with four more poles.
        # This is not the accuracy certificate -- the measured scalar error
        # is -- it exists to catch a defect the scalar layer cannot see.
        refinement = None
        if self.gw.eta0_check_refinement:
            shifts_ref, weights_ref = eta0.hht_coefficients(lmin, lmax, shifts.size + 4)
            delta_ref = eta0.scalar_error(shifts_ref, weights_ref, lmin, lmax)
            integral_ref, _ = self._hht_apply(shifts_ref, weights_ref, d, Lia)
            diff = np.array([np.max(np.abs(integral - integral_ref))])
            diff = mpi_helper.allreduce(diff, op=getattr(mpi_helper.mpi, "MAX", None))
            refinement = {
                "n_poles": int(shifts_ref.size),
                "scalar_error": float(delta_ref),
                "max_abs_diff": float(diff[0]),
            }

        # Power-iteration estimate, reported only to show how loose the
        # rigorous upper bound is
        lambda_max_estimate = self._estimate_mtilde_lambda_max(d, Lia)

        diagnostics = {
            "method": "hht",
            "interval": (float(lmin), float(lmax)),
            "condition_number": float(lmax / lmin),
            "coupling_bound": {
                "frobenius_sq": w_frob_sq,
                "one_norm_times_inf_norm": w_one * w_inf,
                "used": coupling_bound,
            },
            "lambda_max_estimate": lambda_max_estimate,
            "upper_bound_looseness": float(lmax / lambda_max_estimate)
            if lambda_max_estimate > 0
            else np.inf,
            "n_poles": int(shifts.size),
            "n_poles_estimate": int(n_estimate),
            "n_poles_fixed": n_poles_fixed is not None,
            "npoints_legacy": int(self.gw.npoints),
            "scalar_error": float(delta),
            "tol": float(tol),
            # Which route set the tolerance. Without this a result cannot say whether
            # `tol` was asked for directly or derived from a frontier target.
            "tol_source": (
                "qp_tol"
                if qp_tol is not None
                else "moment_tol"
                if moment_tol is not None
                else "eta0_tol"
            ),
            "moment_tol": float(moment_tol) if moment_tol is not None else None,
            "qp_tol": float(qp_tol) if qp_tol is not None else None,
            "frontier_bound_ev": float(delta) * eta0.ETA0_FRONTIER_AMPLIFICATION,
            "cholesky_residuals": {
                "max": float(np.max(residuals)),
                "per_pole": [float(r) for r in residuals],
            },
            "refinement": refinement,
            "intermediate_shapes": [
                ("d_local", d.shape),
                ("Lia_local", Lia.shape),
                ("weighted_rhs", Lia.shape),
                ("gram", (naux, naux)),
                ("cholesky_lhs", (naux, naux)),
                ("solve", Lia.shape),
                ("integral", integral.shape),
            ],
        }
        # The frontier constant is an observed maximum, not a bound, so what was asked
        # for is checked against what was achieved rather than assumed from the request.
        if qp_tol is not None and diagnostics["frontier_bound_ev"] > qp_tol:
            logging.warn(
                f"Achieved eta0 error implies a frontier bound of "
                f"[bad]{diagnostics['frontier_bound_ev']:.3e}[/] eV against the requested "
                f"{qp_tol:.3e} eV"
            )
        self.gw.eta0_diagnostics = diagnostics
        eta0.report(diagnostics)

        return integral

    @logging.with_timer("Nth density-density moments")
    @logging.with_status("Constructing nth density-density moment")
    def build_nth_dd_moment(self, n, recursion_term=None, zeroth_mom=None):
        """Build the nth moment of the density-density response.

        Parameters
        ----------
        n : int
            Moment order to be built.
        recursion_term : numpy.ndarray, optional
            Previous recursion term required to build the next moment. In the case of RPA this is
            the appropriate [(A+B)(A-B)]^(n-2/2) for the nth moment. These are only calculated on
            even moments, odd moments use the previous even moment value.
        zeroth_mom : numpy.ndarray, optional
            Zeroth moment of the density-density response.

        Returns
        -------
        recursion_term : numpy.ndarray
            Term required for the next moment. In the case of RPA this is [(A+B)(A-B)]^(n/2)
        eta_aux : numpy.ndarray
            The nth density-density response moment in (N_aux,N_aux) form
        """
        if n % 2 == 0:
            if zeroth_mom is None:
                zeroth_mom = self.build_zeroth_dd_moment()
            if n != 0:
                tmp = np.dot(self.integrals.Lia * self.d[None], recursion_term) * 4.0
                tmp = mpi_helper.allreduce(tmp)
                recursion_term = util.einsum("i, iP->iP", self.d**2, recursion_term)
                recursion_term += util.einsum("Pi,PQ->iQ", self.integrals.Lia, tmp)
                del tmp
            elif n == 0 and recursion_term is None:
                recursion_term = self.integrals.Lia.T
            return recursion_term, np.dot(zeroth_mom, recursion_term)

        else:
            if recursion_term is None:
                raise AttributeError(
                    f"To build the {n}th dd-moment, a recursion_term must be provided"
                )
            return recursion_term, np.dot(self.integrals.Lia * self.d[None], recursion_term)

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self, integral=None):
        """Build the moments of the density-density response.

        Parameters
        ----------
        integral : numpy.ndarray, optional
            Integral array. If `None`, calculate from scratch. Default is `None`.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """
        if self.d is None:
            self._build_d()

        if integral is None:
            integral = self.build_zeroth_dd_moment()

        p0, p1 = self.mpi_slice(self.nov)
        moments = np.zeros((self.nmom_max + 1, self.naux, p1 - p0))

        # Construct energy differences
        d_full = util.build_1h1p_energies(self.mo_energy_w, self.mo_occ_w).ravel()
        d = d_full[p0:p1]

        # Get the zeroth order moment
        moments[0] = integral

        # Get the first order moment
        moments[1] = self.integrals.Lia * d[None]

        # Get the higher order moments
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * d[None] ** 2
            tmp = np.dot(moments[i - 2], self.integrals.Lia.T)  # aux^2 o v
            tmp = mpi_helper.allreduce(tmp)
            moments[i] += np.dot(tmp, moments[1]) * 4.0  # aux^2 o v
            del tmp

        return moments

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments_exact(self):
        """Build the exact moments of the density-density response.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response.
        """

        import sys

        sys.argv.append("--silent")
        from vayesta.rpa import ssRPA

        rpa = ssRPA(self.gw._scf)
        rpa.kernel()

        rot = np.concatenate([self.integrals.Lia, self.integrals.Lia], axis=-1)

        moments = rpa.gen_moms(self.nmom_max)
        moments = util.einsum("nij,Pi->nPj", moments, rot)

        return moments[:, :, : self.nov]

    def build_dp_moments(self):
        """Build the moments of the dynamic polarizability for optical spectra calculations.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the dynamic polarizability.
        """
        raise NotImplementedError

    # --- Numerical integration functions:

    @staticmethod
    def rescale_quad(bare_quad, a):
        """Rescale quadrature for grid space `a`.

        Parameters
        ----------
        bare_quad : tuple
            The quadrature points and weights.
        a : float
            Grid spacing.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """
        return bare_quad[0] * a, bare_quad[1] * a

    def optimise_main_quad(self, d, diag_eri, name="main"):
        """Optimise the grid spacing of Clenshaw-Curtis quadrature for the main integral.

        Parameters
        ----------
        d : numpy.ndarray
            Orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.
        name : str, optional
            Name of the integral. Default value is `"main"`.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        # Generate the bare quadrature
        bare_quad = self.gen_clencur_quad_semiinf()

        # Calculate the exact value of the integral for the diagonal
        exact = np.sum(d * (d * (d + diag_eri)) ** -0.5)

        # Define the integrand
        integrand = lambda quad: self.eval_diag_main_integral(quad, d, diag_eri)

        # Get the optimal quadrature
        quad = self.get_optimal_quad(bare_quad, integrand, exact, name=name)

        return quad

    def get_optimal_quad(self, bare_quad, integrand, exact, name=None):
        """Get the optimal quadrature.

        Parameters
        ----------
        bare_quad : tuple
            The quadrature points and weights.
        integrand : function
            The integrand function.
        exact : float
            The exact value of the integral.
        name : str, optional
            Name of the integral. Default value is `None`.

        Returns
        -------
        points : numpy.ndarray
            The quadrature points.
        weights : numpy.ndarray
            The quadrature weights.
        """

        def diag_err(spacing):
            """Calculate the error in the diagonal integral."""
            return np.abs(integrand(self.rescale_quad(bare_quad, 10**spacing)) - exact)

        # Optimise the grid spacing
        res = scipy.optimize.minimize_scalar(diag_err, bounds=(-2, 4), method="bounded")
        if not res.success:
            raise RuntimeError("Could not optimise `a` value.")

        # Get the scale
        solve = 10**res.x

        # Report the result
        full_name = f"{f'{name} ' if name else ''}quadrature".capitalize()
        style = logging.rate(res.fun, 1e-14, 1e-10)
        logging.write(f"{full_name} scale:  {solve:.2e} (error = [{style}]{res.fun:.2e}[/])")

        return self.rescale_quad(bare_quad, solve)

    def eval_diag_main_integral(self, quad, d, diag_eri):
        """Evaluate the diagonal of the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray
            Orbital energy differences.
        diag_eri : numpy.ndarray
            Diagonal of the ERIs.

        Returns
        -------
        integral : numpy.ndarray
            Main integral.
        """

        integral = 0.0

        for point, weight in zip(*quad):
            contrib = (d + diag_eri) * d + point**2
            contrib = np.sum(d * contrib ** (-1))

            integral += weight * contrib * 2 / np.pi

        return integral

    def eval_main_integral(self, quad, d=None, Lia=None, include_spin_factor=False):
        """Evaluate the main integral.

        Parameters
        ----------
        quad : tuple
            The quadrature points and weights.
        d : numpy.ndarray, optional
            Orbital energy differences. If `None`, use `self.d`.
            Default value is `None`.
        Lia : numpy.ndarray, optional
            The (aux, W occ, W vir) integral array. If `None`, use
            `self.integrals.Lia`. Keyword argument allows for the use of
            this function with `uhf` and `pbc` modules.
        include_spin_factor : bool, optional
            If `True`, use spin factor of 2 (for unrestricted with combined
            spin channels). If `False`, use spin factor of 4 (for restricted).
            Default value is `False`.

        Returns
        -------
        integral : numpy.ndarray
            Main integral.
        """

        # Get the integral intermediates
        if d is None:
            d = self.d

        if include_spin_factor:
            spin_factor = 2.0
        else:
            spin_factor = 4.0

        if Lia is None:
            Lia = self.integrals.Lia
        naux, nov = Lia.shape  # This `nov` is actually self.mpi_size(nov)

        # Initialise the integral
        dim = 3 if self.report_quadrature_error else 1
        integral = np.zeros((dim, naux, nov))
        integral[:] += Lia

        # Calculate the integral for each point
        for i, (point, weight) in enumerate(zip(*quad)):
            f = d / (d**2 + point**2)
            q = np.dot(Lia * f[None], Lia.T) * spin_factor  # aux^2 o v
            q = mpi_helper.allreduce(q)
            tmp = np.linalg.inv(np.eye(naux) + q) - np.eye(naux)
            del q

            contrib = weight * np.dot(tmp, Lia * f[None]) * (2 / (np.pi))

            integral[0] += contrib
            if i % 2 == 0 and self.report_quadrature_error:
                integral[1] += 2 * contrib
            if i % 4 == 0 and self.report_quadrature_error:
                integral[2] += 4 * contrib

        return integral

    def gen_clencur_quad_semiinf(self):
        """Generate quadrature points and weights for Clenshaw-Curtis quadrature over semiinfinite
        range (0 to +inf)
        """
        j = np.arange(1, self.gw.npoints + 1)
        tvals = np.pi * j / (self.gw.npoints + 1)
        points = 1.0 / np.tan(tvals / 2) ** 2
        # Vectorize the inner sum computation
        j_mesh, t_mesh = np.meshgrid(j, tvals, indexing="ij")
        jsums = np.sum(np.sin(j_mesh * t_mesh) * (1 - np.cos(j_mesh * np.pi)) / j_mesh, axis=0)
        weights = (4 * np.sin(tvals) / ((self.gw.npoints + 1) * (1 - np.cos(tvals)) ** 2)) * jsums
        return points, weights

    def gen_gausslag_quad_semiinf(self):
        """Generate quadrature points and weights for Gauss-Laguerre quadrature over an ``(0,
        +inf)``.

        Returns
        -------
        points : numpy.ndarray
            Quadrature points.
        weights : numpy.ndarray
            Quadrature weights.
        """
        points, weights = np.polynomial.laguerre.laggauss(self.gw.npoints)
        weights *= np.exp(points)
        return points, weights

    def estimate_error_clencur(self, i4, i2, imag_tol=1e-10):
        """Estimate the quadrature error for Clenshaw-Curtis quadrature.

        Parameters
        ----------
        i4 : numpy.ndarray
            Integral at one-quarter the number of points.
        i2 : numpy.ndarray
            Integral at one-half the number of points.
        imag_tol : float, optional
            Threshold to consider the imaginary part of a root to be zero.
            Default value is `1e-10`.

        Returns
        -------
        error : numpy.ndarray
            Estimated error.
        """

        if (i4 - i2) < 1e-14:
            return 0.0

        # Eq. 103 from https://arxiv.org/abs/2301.09107
        roots = np.roots([1, 0, i4 / (i4 - i2), -i2 / (i4 - i2)])

        # Require a real root between 0 and 1
        real_roots = roots[np.abs(roots.imag) < 1e-10].real

        # Check how many there are
        if len(real_roots) > 1:
            logging.warn(
                "Nested quadrature error estimation gives [bad]%d real roots[/]. "
                "Taking smallest positive root." % len(real_roots),
            )
        else:
            logging.write(
                f"Nested quadrature error estimation gives {len(real_roots)} "
                f"real root{'s' if len(real_roots) != 1 else ''}.",
            )

        # Check if there is a root between 0 and 1
        if not np.any(np.logical_and(real_roots > 0, real_roots < 1)):
            logging.warn(
                "Nested quadrature error estimation gives [bad]no root between 0 and 1[/]."
            )
            return np.nan
        else:
            root = np.min(real_roots[np.logical_and(real_roots > 0, real_roots < 1)])

        # Calculate the error
        error = i2 / (1.0 + root**-2)

        return error
