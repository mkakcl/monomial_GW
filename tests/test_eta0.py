"""Tests for `eta0.py` and the HHT zeroth-moment route in `rpa.py`.

Covers the Milestone 2 acceptance gate: the scalar layer from
well-conditioned through extreme condition numbers against an
extended-precision oracle, explicit failure when the requested tolerance is
not representable, dense comparison against an eigendecomposition of
``Mtilde``, agreement of the projected kernel with the legacy quadrature and
the dense oracle, invariance of downstream moments and quasiparticle
energies, compression on/off, frozen core, zero coupling, the empty and
degenerate edge cases, and shape instrumentation proving that no
particle-hole squared matrix is formed.
"""

import unittest

import numpy as np
import scipy.linalg
import scipy.special
from pyscf import dft, gto, scf

from momentGW import eta0
from momentGW.gw import GW
from momentGW.rpa import dRPA
from momentGW.uhf.gw import UGW

LD = np.longdouble


def rational(shifts, weights, x):
    """Evaluate the rational approximation in extended precision."""
    x = np.asarray(x, dtype=LD)
    r = np.zeros_like(x)
    for s, w in zip(np.asarray(shifts, dtype=LD), np.asarray(weights, dtype=LD)):
        r += w / (x + s)
    return r


class Test_ScalarLayer(unittest.TestCase):
    def test_elliptic_building_blocks_vs_scipy(self):
        # The extended-precision AGM and Landen constructions must agree
        # with scipy's float64 elliptic functions to float64 precision
        for k2 in (0.9, 0.5, 1e-2, 1e-6, 1e-12):
            kp_agm = float(eta0._PI / (2 * eta0._agm(LD(1), np.sqrt(LD(k2)))))
            kp_scipy = scipy.special.ellipkm1(k2)
            self.assertAlmostEqual(kp_agm / kp_scipy, 1.0, places=14)

            u = np.linspace(0.1, 2.0, 7)
            sn, cn, dn = eta0._sncndn(u, LD(k2))
            sn_s, cn_s, dn_s, _ = scipy.special.ellipj(u, 1 - k2)
            np.testing.assert_allclose(sn.astype(float), sn_s, atol=1e-14)
            np.testing.assert_allclose(cn.astype(float), cn_s, atol=1e-14)
            np.testing.assert_allclose(dn.astype(float), dn_s, atol=1e-14)

    def test_selection_meets_tolerance_across_conditions(self):
        # Well-conditioned through extreme intervals: the selected pole
        # count must meet the requested tolerance as measured, and the
        # count must grow with the condition number
        last_n = 0
        for kappa in (1.0, 1e2, 1e6, 1e10, 1e14):
            shifts, weights, error, n_estimate = eta0.select_poles(1.0, kappa, 1e-14)
            self.assertLessEqual(error, 1e-14)
            self.assertTrue(np.all(shifts > 0))
            self.assertTrue(np.all(weights > 0))
            self.assertGreaterEqual(shifts.size, last_n)
            last_n = shifts.size

    def test_scalar_error_against_independent_oracle(self):
        # The reported supremum must bound pointwise errors measured on an
        # independent random grid in extended precision
        rng = np.random.default_rng(7)
        for kappa in (1e2, 1e6, 1e10):
            shifts, weights, error, _ = eta0.select_poles(1.0, kappa, 1e-12)
            x = np.exp(rng.uniform(0, np.log(kappa), 2000)).astype(LD)
            pointwise = np.abs(1 - np.sqrt(x) * rational(shifts, weights, x))
            self.assertLessEqual(float(np.max(pointwise)), 2 * error + 1e-17)

    def test_unrepresentable_tolerance_raises(self):
        with self.assertRaises(ValueError):
            eta0.select_poles(1.0, 1e4, 1e-18)

    def test_unreachable_tolerance_raises(self):
        with self.assertRaises(RuntimeError):
            eta0.select_poles(1.0, 1e16, 1e-14, n_poles_max=20)

    def test_fixed_pole_count(self):
        # An explicit pole count is an instruction: honoured exactly, with
        # the measured error returned rather than raised
        shifts, weights, error, _ = eta0.select_poles(1.0, 1e6, 1e-14, n_poles=8)
        self.assertEqual(shifts.size, 8)
        self.assertGreater(error, 1e-14)

    def test_degenerate_interval(self):
        # lmin == lmax: the map degenerates to the trigonometric limit and
        # remains exact
        shifts, weights, error, _ = eta0.select_poles(4.0, 4.0, 1e-14)
        self.assertLessEqual(error, 1e-14)
        self.assertLessEqual(shifts.size, 4)

    def test_certified_interval_validation(self):
        with self.assertRaises(ValueError):
            eta0.certified_interval(np.array([]), 0.0)
        with self.assertRaises(ValueError):
            eta0.certified_interval(np.array([1.0, -1.0]), 0.0)
        with self.assertRaises(ValueError):
            eta0.certified_interval(np.array([1.0, 0.0]), 0.0)
        with self.assertRaises(ValueError):
            eta0.certified_interval(np.array([1.0, np.inf]), 0.0)
        with self.assertRaises(ValueError):
            eta0.certified_interval(np.array([1.0, 2.0]), np.nan)

    def test_certified_interval_padding_is_outward(self):
        d = np.array([0.5, 2.0])
        lmin, lmax = eta0.certified_interval(d, 3.0)
        self.assertLess(lmin, 0.25)
        self.assertGreater(lmax, 7.0)
        self.assertAlmostEqual(lmin, 0.25, places=6)
        self.assertAlmostEqual(lmax, 7.0, places=5)

    def test_zero_coupling_interval(self):
        d = np.array([0.5, 2.0])
        lmin, lmax = eta0.certified_interval(d, 0.0)
        shifts, weights, error, _ = eta0.select_poles(lmin, lmax, 1e-14)
        self.assertLessEqual(error, 1e-14)


class Test_DenseKernel(unittest.TestCase):
    def test_dense_vs_eigendecomposition(self):
        # Synthetic positive measure: the Woodbury route through the
        # auxiliary space must reproduce the eigendecomposition of Mtilde
        # at the level of the scalar certificate
        rng = np.random.default_rng(1)
        nov, naux = 150, 20
        d = np.exp(rng.uniform(np.log(0.05), np.log(20.0), nov))
        Lia = rng.standard_normal((naux, nov)) * 0.3
        W = (Lia * np.sqrt(d)[None]).T
        mtilde = np.diag(d**2) + 4.0 * W @ W.T
        evals, evecs = np.linalg.eigh(mtilde)
        oracle = (np.sqrt(d)[:, None] * (evecs @ ((evals**-0.5)[:, None] * (evecs.T @ W)))).T

        w_frob_sq = np.sum(W**2)
        w_norm_bound = min(w_frob_sq, np.linalg.norm(W, 1) * np.linalg.norm(W, np.inf))
        lmin, lmax = eta0.certified_interval(d, 4.0 * w_norm_bound)
        self.assertLessEqual(lmin, evals.min())
        self.assertGreaterEqual(lmax, evals.max())

        for tol in (1e-6, 1e-10, 1e-14):
            shifts, weights, error, _ = eta0.select_poles(lmin, lmax, tol)
            out = np.zeros_like(Lia)
            eye = np.eye(naux)
            for s, w in zip(shifts, weights):
                f = d / (d * d + s)
                Y = Lia * f[None]
                A = eye + 4.0 * np.dot(Y, Lia.T)
                out += w * scipy.linalg.cho_solve(scipy.linalg.cho_factor(A), Y)
            rel = np.max(np.abs(out - oracle)) / np.max(np.abs(oracle))
            # The kernel error tracks the scalar certificate until float64
            # rounding takes over
            self.assertLessEqual(rel, 10 * error + 1e-12)


class Test_Molecular(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="cc-pvdz",
            verbose=0,
        )
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def _dense_oracle(self, rpa):
        """Dense eigendecomposition oracle for the projected zeroth moment."""
        if rpa.d is None:
            rpa._build_d()
        d = rpa.d
        Lia = rpa.integrals.Lia
        W = (Lia * np.sqrt(d)[None]).T
        mtilde = np.diag(d**2) + 4.0 * W @ W.T
        evals, evecs = np.linalg.eigh(mtilde)
        return (np.sqrt(d)[:, None] * (evecs @ ((evals**-0.5)[:, None] * (evecs.T @ W)))).T

    def _compare_routes(self, mf):
        """Compare HHT against the legacy quadrature and the dense oracle."""
        gw_leg = GW(mf)
        integrals = gw_leg.ao2mo()
        rpa_leg = dRPA(gw_leg, 1, integrals)
        rpa_leg._build_d()
        legacy = rpa_leg.build_zeroth_dd_moment()

        gw_hht = GW(mf, eta0_method="hht")
        rpa_hht = dRPA(gw_hht, 1, integrals)
        hht = rpa_hht.build_zeroth_dd_moment()

        oracle = self._dense_oracle(rpa_hht)
        scale = np.max(np.abs(oracle))

        # The HHT route must sit on the dense oracle at the level of its
        # certificate, and the legacy quadrature agrees with both at its
        # own (uncertified) accuracy
        delta = gw_hht.eta0_diagnostics["scalar_error"]
        self.assertLessEqual(np.max(np.abs(hht - oracle)) / scale, 100 * delta + 1e-13)
        self.assertLessEqual(np.max(np.abs(legacy - oracle)) / scale, 1e-10)
        self.assertLessEqual(np.max(np.abs(hht - legacy)) / scale, 1e-10)

    def test_hht_vs_legacy_and_dense_oracle(self):
        self._compare_routes(self.mf)

    def test_hht_vs_legacy_and_dense_oracle_h2(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        self._compare_routes(mf)

    def test_downstream_invariance(self):
        # Full G0W0: density-response moments, self-energy moments, and
        # quasiparticle energies must be invariant within the certificate
        th = {}
        tp = {}
        qp = {}
        for method in ("clencur", "hht"):
            gw = GW(self.mf, eta0_method=method)
            integrals = gw.ao2mo()
            th[method], tp[method] = gw.build_se_moments(
                3, integrals, mo_energy=dict(g=gw.mo_energy, w=gw.mo_energy)
            )
            conv, gf, se, _ = gw.kernel(nmom_max=3, integrals=integrals)
            qp[method] = gw.qp_energy.copy()
            self.assertTrue(conv)

        for n in range(len(th["clencur"])):
            scale = max(np.max(np.abs(th["clencur"][n])), 1.0)
            self.assertLessEqual(np.max(np.abs(th["clencur"][n] - th["hht"][n])) / scale, 1e-10)
            scale = max(np.max(np.abs(tp["clencur"][n])), 1.0)
            self.assertLessEqual(np.max(np.abs(tp["clencur"][n] - tp["hht"][n])) / scale, 1e-10)
        self.assertLessEqual(np.max(np.abs(qp["clencur"] - qp["hht"])), 1e-10)

    def test_compression_on_off(self):
        # The route must not depend on whether the auxiliary space was
        # compressed: each compression choice agrees with its own dense
        # oracle, and the QP energies agree across the choice
        qp = {}
        for compression in ("ia", ""):
            gw = GW(self.mf, eta0_method="hht", compression=compression)
            integrals = gw.ao2mo()
            rpa = dRPA(gw, 1, integrals)
            hht = rpa.build_zeroth_dd_moment()
            oracle = self._dense_oracle(rpa)
            scale = np.max(np.abs(oracle))
            self.assertLessEqual(np.max(np.abs(hht - oracle)) / scale, 1e-12)

            gw.kernel(nmom_max=1, integrals=integrals)
            qp[compression] = gw.qp_energy.copy()

        # Compression itself moves the result at the compression tolerance,
        # not at the eta0 certificate
        self.assertLessEqual(np.max(np.abs(qp["ia"] - qp[""])), 1e-6)

    def test_frozen_core(self):
        qp = {}
        for method in ("clencur", "hht"):
            gw = GW(self.mf, eta0_method=method)
            gw.frozen = [0]
            conv, gf, se, _ = gw.kernel(nmom_max=1)
            qp[method] = gw.qp_energy.copy()
        self.assertLessEqual(np.max(np.abs(qp["clencur"] - qp["hht"])), 1e-10)

    def test_refinement_check(self):
        gw = GW(self.mf, eta0_method="hht", eta0_check_refinement=True)
        integrals = gw.ao2mo()
        rpa = dRPA(gw, 1, integrals)
        rpa.build_zeroth_dd_moment()
        refinement = gw.eta0_diagnostics["refinement"]
        self.assertIsNotNone(refinement)
        self.assertEqual(refinement["n_poles"], gw.eta0_diagnostics["n_poles"] + 4)
        self.assertLessEqual(refinement["max_abs_diff"], 1e-12)

    def test_no_particle_hole_squared_intermediates(self):
        # Shape instrumentation: every intermediate must have at most one
        # dimension of particle-hole size, and the local output shape is
        # asserted by the kernel itself
        gw = GW(self.mf, eta0_method="hht")
        integrals = gw.ao2mo()
        rpa = dRPA(gw, 1, integrals)
        integral = rpa.build_zeroth_dd_moment()
        nov = rpa.nov
        self.assertGreater(nov, integrals.naux)  # the test is vacuous otherwise
        for name, shape in gw.eta0_diagnostics["intermediate_shapes"]:
            self.assertLessEqual(sum(dim == nov for dim in shape), 1, f"{name} has shape {shape}")
        self.assertEqual(integral.shape, (integrals.naux, nov))

    def test_diagnostics_are_complete(self):
        # The roadmap's definition of a trustworthy calculation asks for
        # the interval, condition number, pole count, scalar error, and
        # solve residuals
        gw = GW(self.mf, eta0_method="hht")
        integrals = gw.ao2mo()
        rpa = dRPA(gw, 1, integrals)
        rpa.build_zeroth_dd_moment()
        diag = gw.eta0_diagnostics
        lmin, lmax = diag["interval"]
        self.assertGreater(lmin, 0)
        self.assertGreater(lmax, lmin)
        self.assertAlmostEqual(diag["condition_number"], lmax / lmin, places=8)
        self.assertGreater(diag["n_poles"], 0)
        self.assertLessEqual(diag["scalar_error"], diag["tol"])
        self.assertEqual(len(diag["cholesky_residuals"]["per_pole"]), diag["n_poles"])
        self.assertLessEqual(diag["cholesky_residuals"]["max"], 1e-12)
        # The rigorous upper bound encloses the power-iteration estimate
        self.assertLessEqual(diag["lambda_max_estimate"], lmax)
        self.assertGreaterEqual(diag["upper_bound_looseness"], 1.0)


class Test_SmallGap(unittest.TestCase):
    """Ozone from a PBE starting point: the stiffest case in the baseline set.

    Its ``Mtilde`` condition number is around 1e5, where the legacy
    quadrature's true error has already risen to 1e-9 while its reported
    estimate is uninformative (see `baseline/README.md`).  The certified
    route must hold its certificate here.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom=(
                "O 0.000000 0.000000 0.000000; "
                "O 1.086900 0.000000 0.660000; "
                "O -1.086900 0.000000 0.660000"
            ),
            basis="ccpvdz",
            verbose=0,
        )
        mf = dft.RKS(mol, xc="pbe").density_fit(auxbasis="weigend")
        mf.conv_tol = 1e-11
        mf.kernel()
        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_small_gap_certificate_holds(self):
        gw_leg = GW(self.mf)
        integrals = gw_leg.ao2mo()
        rpa_leg = dRPA(gw_leg, 1, integrals)
        rpa_leg._build_d()
        legacy = rpa_leg.build_zeroth_dd_moment()

        gw_hht = GW(self.mf, eta0_method="hht")
        rpa_hht = dRPA(gw_hht, 1, integrals)
        hht = rpa_hht.build_zeroth_dd_moment()

        diag = gw_hht.eta0_diagnostics
        self.assertGreater(diag["condition_number"], 1e4)
        self.assertLessEqual(diag["scalar_error"], diag["tol"])
        self.assertLessEqual(diag["cholesky_residuals"]["max"], 1e-12)

        # Dense oracle on the stiff system
        d = rpa_hht.d
        Lia = integrals.Lia
        W = (Lia * np.sqrt(d)[None]).T
        mtilde = np.diag(d**2) + 4.0 * W @ W.T
        evals, evecs = np.linalg.eigh(mtilde)
        oracle = (np.sqrt(d)[:, None] * (evecs @ ((evals**-0.5)[:, None] * (evecs.T @ W)))).T
        scale = np.max(np.abs(oracle))

        # The conditioning admits some float64 amplification; the
        # certificate plus a condition-scaled rounding floor must hold,
        # and the legacy route is known-good here to ~1e-9
        floor = np.finfo(float).eps * np.sqrt(diag["condition_number"])
        self.assertLessEqual(
            np.max(np.abs(hht - oracle)) / scale, 100 * (diag["scalar_error"] + floor)
        )
        self.assertLessEqual(np.max(np.abs(hht - legacy)) / scale, 1e-7)


class Test_Edges(unittest.TestCase):
    def test_empty_particle_hole_space(self):
        # A screened-interaction space with every orbital occupied has
        # nov = 0.  (`Integrals.transform` cannot build such a system from
        # scratch, so the occupancies are overridden on a real molecule.)
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        gw = GW(mf, eta0_method="hht")
        integrals = gw.ao2mo()
        rpa = dRPA(
            gw,
            1,
            integrals,
            mo_energy=dict(g=gw.mo_energy, w=gw.mo_energy),
            mo_occ=dict(g=gw.mo_occ, w=np.full_like(gw.mo_occ, 2.0)),
        )
        self.assertEqual(rpa.nov, 0)
        integral = rpa.build_zeroth_dd_moment()
        self.assertEqual(integral.shape, (integrals.naux, 0))

    def test_zero_coupling(self):
        # With the RI couplings zeroed the projected moment is exactly
        # zero, and the route must survive the degenerate Gram
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        gw = GW(mf, eta0_method="hht")
        integrals = gw.ao2mo()
        rpa = dRPA(gw, 1, integrals)
        integrals._blocks["Lia"] = np.zeros_like(integrals.Lia)
        integral = rpa.build_zeroth_dd_moment()
        self.assertEqual(integral.shape, integrals.Lia.shape)
        self.assertTrue(np.all(integral == 0))
        self.assertTrue(np.isfinite(gw.eta0_diagnostics["condition_number"]))

    def test_unknown_method_raises_at_construction(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).density_fit()
        with self.assertRaises(ValueError):
            GW(mf, eta0_method="typo")

    def test_unrestricted_guard(self):
        # The unrestricted solver overrides the zeroth-moment build, so the
        # option would be silently inert: it must fail at construction
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.UHF(mol).density_fit()
        with self.assertRaises(NotImplementedError):
            UGW(mf, eta0_method="hht")
        UGW(mf)  # the default still constructs


if __name__ == "__main__":
    unittest.main()
