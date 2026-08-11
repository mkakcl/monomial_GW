"""Tests for `gw.py`."""

import unittest

import numpy as np
import pytest
from pyscf import dft, gto, gw, lib, tdscf
from pyscf.agf2 import mpi_helper

from momentGW import GW


class Test_GW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = "hf"
        mf.conv_tol = 1e-11
        mf.kernel()

        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)

        gw_exact = gw.GW(mf, freq_int="exact")
        gw_exact.kernel()

        mf = mf.density_fit(auxbasis="cc-pv5z-ri")
        mf.with_df.build()

        cls.mol, cls.mf, cls.gw_exact = mol, mf, gw_exact

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.gw_exact

    def test_vs_pyscf_vhf_df(self):
        gw = GW(self.mf)
        gw.diagonal_se = True
        conv, gf, se, _ = gw.kernel(nmom_max=7)
        gf = gf.physical(weight=1e-8)
        self.assertAlmostEqual(
            gf.occupied().energies.max(),
            self.gw_exact.mo_energy[self.gw_exact.mo_occ > 0].max(),
            2,
        )
        self.assertAlmostEqual(
            gf.virtual().energies.min(),
            self.gw_exact.mo_energy[self.gw_exact.mo_occ == 0].min(),
            2,
        )

    def test_vs_pyscf_no_vhf_df(self):
        gw = GW(self.mf)
        gw.diagonal_se = True
        conv, gf, se, _ = gw.kernel(nmom_max=7)
        gf = gf.physical(weight=1e-8)
        self.assertAlmostEqual(
            gf.occupied().energies.max(),
            self.gw_exact.mo_energy[self.gw_exact.mo_occ > 0].max(),
            2,
        )
        self.assertAlmostEqual(
            gf.virtual().energies.min(),
            self.gw_exact.mo_energy[self.gw_exact.mo_occ == 0].min(),
            2,
        )

    def test_nelec(self):
        gw = GW(self.mf)
        gw.diagonal_se = True
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.occupied().moment(0).trace() * 2,
            self.mol.nelectron,
            1,
        )
        gw.optimise_chempot = True
        conv, gf, se, _ = gw.kernel(nmom_max=1)
        self.assertAlmostEqual(
            gf.occupied().moment(0).trace() * 2,
            self.mol.nelectron,
            8,
        )

    def test_moments(self):
        gw = GW(self.mf)
        gw.diagonal_se = True
        th1, tp1 = gw.build_se_moments(5, gw.ao2mo())
        conv, gf, se, _ = gw.kernel(nmom_max=5)
        th2 = se.occupied().moment(range(5))
        tp2 = se.virtual().moment(range(5))

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def test_moments_vs_tdscf_rpa(self):
        if mpi_helper.size > 1:
            pytest.skip("Doesn't work with MPI")

        gw = GW(self.mf)
        gw.diagonal_se = True
        nocc, nvir = gw.nocc, gw.nmo - gw.nocc
        th1, tp1 = gw.build_se_moments(5, gw.ao2mo())

        td = tdscf.dRPA(self.mf)
        td.nstates = nocc * nvir
        td.kernel()
        z = np.sum(np.array(td.xy) * 2, axis=1).reshape(len(td.e), nocc, nvir)
        integrals = gw.ao2mo()
        Lpq = integrals.Lpx
        Lia = integrals.Lia
        z = z.reshape(-1, nocc * nvir)

        m = lib.einsum("Qx,vx,Qpj->vpj", Lia, z, Lpq[:, :, :nocc])
        e = lib.direct_sum("j-v->jv", self.mf.mo_energy[:nocc], td.e)
        th2 = []
        for n in range(6):
            t = lib.einsum("vpj,jv,vqj->pq", m, np.power(e, n), m)
            if gw.diagonal_se:
                t = np.diag(np.diag(t))
            th2.append(t)

        m = lib.einsum("Qx,vx,Qqb->vqb", Lia, z, Lpq[:, :, nocc:])
        e = lib.direct_sum("b+v->bv", self.mf.mo_energy[nocc:], td.e)
        tp2 = []
        for n in range(6):
            t = lib.einsum("vpj,jv,vqj->pq", m, np.power(e, n), m)
            if gw.diagonal_se:
                t = np.diag(np.diag(t))
            tp2.append(t)

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def test_moments_vs_tdscf_tda(self):
        if mpi_helper.size > 1:
            pytest.skip("Doesn't work with MPI")

        gw = GW(self.mf)
        gw.diagonal_se = True
        gw.polarizability = "dtda"
        nocc, nvir = gw.nocc, gw.nmo - gw.nocc
        th1, tp1 = gw.build_se_moments(5, gw.ao2mo())

        td = tdscf.dTDA(self.mf)
        td.nstates = nocc * nvir
        td.kernel()
        xy = np.array([x[0] for x in td.xy])
        z = xy * 2
        integrals = gw.ao2mo()
        Lpq = integrals.Lpx
        Lia = integrals.Lia
        z = z.reshape(-1, nocc * nvir)

        m = lib.einsum("Qx,vx,Qpj->vpj", Lia, z, Lpq[:, :, :nocc])
        e = lib.direct_sum("j-v->jv", self.mf.mo_energy[:nocc], td.e)
        th2 = []
        for n in range(6):
            t = lib.einsum("vpj,jv,vqj->pq", m, np.power(e, n), m)
            if gw.diagonal_se:
                t = np.diag(np.diag(t))
            th2.append(t)

        m = lib.einsum("Qx,vx,Qqb->vqb", Lia, z, Lpq[:, :, nocc:])
        e = lib.direct_sum("b+v->bv", self.mf.mo_energy[nocc:], td.e)
        tp2 = []
        for n in range(6):
            t = lib.einsum("vpj,jv,vqj->pq", m, np.power(e, n), m)
            if gw.diagonal_se:
                t = np.diag(np.diag(t))
            tp2.append(t)

        for a, b in zip(th1, th2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)
        for a, b in zip(tp1, tp2):
            dif = np.max(np.abs(a - b)) / np.max(np.abs(a))
            self.assertAlmostEqual(dif, 0, 8)

    def _test_regression(self, xc, kwargs, nmom_max, ip, ea, name=""):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis="weigend").run()
        mf.mo_coeff = mpi_helper.bcast_dict(mf.mo_coeff, root=0)
        mf.mo_energy = mpi_helper.bcast_dict(mf.mo_energy, root=0)
        gw = GW(mf, **kwargs)
        gw.kernel(nmom_max)
        gf = gw.gf.physical(weight=0.1)
        self.assertAlmostEqual(gf.occupied().energies[-1], ip, 6, msg=name)
        self.assertAlmostEqual(gf.virtual().energies[0], ea, 6, msg=name)

    def test_regression_simple(self):
        ip = -0.2775530233069981
        ea = 0.0055297771164524835
        self._test_regression("hf", dict(), 3, ip, ea, "simple")

    def test_regression_pbe(self):
        ip = -0.23326311518556142
        ea = 0.002757793304710979
        self._test_regression("pbe", dict(), 3, ip, ea, "pbe")

    def test_regression_fock_loop(self):
        ip = -0.28546340568433876
        ea = 0.006558884450397966
        self._test_regression("hf", dict(fock_loop=True), 1, ip, ea, "fock loop")

    def test_moment_order_convergence_off_by_default(self):
        """The truncation estimate costs a second solve, so it is opt-in."""
        gw = GW(self.mf)
        gw.kernel(3)
        self.assertIsNone(gw.dyson_diagnostics["moment_order_convergence"])

    def test_moment_order_convergence_matches_an_explicit_pair(self):
        """The estimate equals the shift between two independent calculations.

        The whole point of reusing the moments is that truncating them to `nmom_max - 1`
        entries is identical to having built them at `nmom_max - 2`. This checks that
        against the two-calculation version rather than assuming it.
        """
        gw = GW(self.mf, moment_order_convergence=True)
        gw.kernel(5)
        record = gw.dyson_diagnostics["moment_order_convergence"]

        self.assertEqual(record["nmom_max"], 5)
        self.assertEqual(record["nmom_max_compared"], 3)
        self.assertFalse(record["self_consistent_excluded"])

        lower = GW(self.mf)
        lower.kernel(3)
        upper = GW(self.mf)
        upper.kernel(5)
        for name, index, sector in (("homo", -1, "occupied"), ("lumo", 0, "virtual")):
            expected = (
                getattr(getattr(upper.gf.physical(weight=0.1), sector)(), "energies")[index]
                - getattr(getattr(lower.gf.physical(weight=0.1), sector)(), "energies")[index]
            )
            self.assertAlmostEqual(record[f"{name}_shift"], expected, 10, msg=name)

    def test_moment_order_convergence_reports_the_dominant_orbital(self):
        """A level crossing has to be visible, not read as a large shift."""
        gw = GW(self.mf, moment_order_convergence=True)
        gw.kernel(5)
        record = gw.dyson_diagnostics["moment_order_convergence"]

        for name in ("homo", "lumo"):
            self.assertIn(f"{name}_orbital", record["frontier"])
            self.assertIn(f"{name}_orbital", record["frontier_compared"])
            self.assertIsInstance(record[f"{name}_orbital_changed"], bool)

    def test_moment_order_convergence_needs_a_lower_order(self):
        """`nmom_max = 1` has nothing to compare against and says so."""
        gw = GW(self.mf, moment_order_convergence=True)
        gw.kernel(1)
        self.assertIsNone(gw.dyson_diagnostics["moment_order_convergence"])

    def test_moment_order_convergence_does_not_change_the_result(self):
        """Switching the estimate on is a diagnostic, not a change of calculation."""
        off = GW(self.mf)
        off.kernel(3)
        on = GW(self.mf, moment_order_convergence=True)
        on.kernel(3)
        np.testing.assert_allclose(on.qp_energy, off.qp_energy, rtol=0, atol=0)

    def test_nmom_max_tol_off_by_default(self):
        """Without a tolerance, `nmom_max` means exactly what it always meant."""
        gw = GW(self.mf)
        gw.kernel(5)
        self.assertIsNone(gw.dyson_diagnostics["moment_order"])
        self.assertNotIn("moment_order", gw.dyson_diagnostics["gates"])

    def test_nmom_max_tol_stops_early_and_matches_that_order(self):
        """A loose tolerance stops below the cap, and gives that order's answer.

        The point of reusing the moments is that a truncated set is the same as one
        built at the lower order, so the adaptive result must match a plain run at the
        order it settled on. They agree to roundoff rather than bit-for-bit: `convolve`
        contracts every output order in one GEMM, whose row count depends on `nmom_max`,
        so the two builds sum in a slightly different order.
        """
        gw = GW(self.mf, nmom_max_tol=1e-1)
        gw.kernel(7)
        record = gw.dyson_diagnostics["moment_order"]

        self.assertTrue(record["converged"])
        self.assertLess(record["order"], 7)
        self.assertEqual(record["cap"], 7)
        self.assertTrue(gw.dyson_diagnostics["gates"]["moment_order"])

        plain = GW(self.mf)
        plain.kernel(record["order"])
        np.testing.assert_allclose(gw.qp_energy, plain.qp_energy, rtol=0, atol=1e-11)

    def test_nmom_max_tol_needs_two_consecutive_orders(self):
        """One small shift is not convergence; the shift is not monotonic in the order."""
        gw = GW(self.mf, nmom_max_tol=1e-1)
        gw.kernel(7)
        record = gw.dyson_diagnostics["moment_order"]

        settled = [x for x in record["shifts"] if x is not None][-2:]
        self.assertEqual(len(settled), 2)
        for shift in settled:
            self.assertLess(shift, record["tol"])

    def test_nmom_max_tol_unmet_is_reported_as_unconverged(self):
        """Reaching the cap without meeting the tolerance fails the gate."""
        gw = GW(self.mf, nmom_max_tol=1e-12)
        gw.kernel(5)
        record = gw.dyson_diagnostics["moment_order"]

        self.assertFalse(record["converged"])
        self.assertEqual(record["order"], 5)
        self.assertFalse(gw.dyson_diagnostics["gates"]["moment_order"])
        self.assertFalse(gw.dyson_diagnostics["converged"])
        self.assertFalse(gw.converged)

    def test_nmom_max_tol_carries_the_other_quantities(self):
        """The walk records more than the frontier, one entry per order tried."""
        gw = GW(self.mf, nmom_max_tol=1e-1)
        gw.kernel(7)
        record = gw.dyson_diagnostics["moment_order"]

        n = len(record["orders"])
        for key in ("shifts", "nelec_errors", "spectral_weight_deficits", "spectral_weight_min"):
            self.assertEqual(len(record[key]), n, msg=key)

    def test_nmom_max_tol_requires_the_particle_number_too(self):
        """The order it settles on must satisfy the particle-number gate as well."""
        gw = GW(self.mf, nmom_max_tol=1e-1)
        gw.kernel(7)
        record = gw.dyson_diagnostics["moment_order"]

        self.assertTrue(record["converged"])
        chosen = record["frontiers"][-1]
        self.assertLessEqual(abs(chosen["nelec_error"]), chosen["nelec_tol"])

    def test_spectral_weight_sum_rule_holds_at_every_order(self):
        """`Tr[G(0)] = nmo` with non-negative residues is a sum rule, not a limit.

        It is recorded as a validity check rather than a convergence criterion because it
        holds at every order by construction; the value of the check is that a future
        change breaking it would be caught.
        """
        gw = GW(self.mf, nmom_max_tol=1e-1)
        gw.kernel(7)
        record = gw.dyson_diagnostics["moment_order"]

        for deficit, smallest in zip(
            record["spectral_weight_deficits"], record["spectral_weight_min"]
        ):
            self.assertAlmostEqual(deficit, 0.0, 10)
            self.assertGreater(smallest, -1e-10)

    def test_error_budget_collects_every_measured_contribution(self):
        """Each source that is measured somewhere appears once, with its own unit."""
        gw = GW(self.mf, moment_order_convergence=True)
        gw.kernel(5)
        budget = gw.error_budget()

        for name in ("eta0", "cholesky", "realization", "particle_number", "moment_truncation"):
            self.assertIn(name, budget["contributions"], msg=name)
        for name, entry in budget["contributions"].items():
            self.assertIn("unit", entry, msg=name)
            self.assertIn("source", entry, msg=name)

    def test_error_budget_ranks_only_what_converts_to_ev(self):
        """Contributions without a measured amplification are not ranked or summed."""
        gw = GW(self.mf, moment_order_convergence=True)
        gw.kernel(5)
        budget = gw.error_budget()

        ranked = [row["frontier_ev"] for row in budget["ranked"]]
        self.assertEqual(ranked, sorted(ranked, reverse=True))
        for row in budget["ranked"]:
            self.assertIsNotNone(budget["contributions"][row["name"]]["frontier_ev"])
        for name in ("cholesky", "realization", "particle_number"):
            self.assertIsNone(budget["contributions"][name]["frontier_ev"], msg=name)
        self.assertNotIn("total", budget)

    def test_error_budget_is_dominated_by_truncation(self):
        """The whole reason for the budget: truncation dwarfs every numerical term."""
        gw = GW(self.mf, moment_order_convergence=True)
        gw.kernel(5)
        budget = gw.error_budget()

        self.assertEqual(budget["dominant"], "moment_truncation")
        truncation = budget["contributions"]["moment_truncation"]["frontier_ev"]
        eta0 = budget["contributions"]["eta0"]["frontier_ev"]
        self.assertGreater(truncation, 1e6 * eta0)

    def test_error_budget_names_what_it_cannot_measure(self):
        """A contribution that is not measured is listed, not silently omitted."""
        gw = GW(self.mf)
        gw.kernel(5)
        budget = gw.error_budget()

        self.assertNotIn("moment_truncation", budget["contributions"])
        self.assertTrue(any("moment_truncation" in item for item in budget["unquantified"]))
        self.assertTrue(any("compression" in item for item in budget["unquantified"]))

    def test_closure_spread_off_by_default(self):
        """It is a second realization, so it is opt-in."""
        gw = GW(self.mf)
        gw.kernel(5)
        self.assertIsNone(gw.dyson_diagnostics["closure_spread"])

    def test_closure_spread_pins_each_sector_at_its_own_edge(self):
        """One shared pin would land outside one sector; the two must differ."""
        gw = GW(self.mf, closure_spread=True)
        gw.kernel(5)
        record = gw.dyson_diagnostics["closure_spread"]

        hole_edge, particle_edge = record["sector_edges"]
        low, high = record["pins"]
        self.assertGreater(low, hole_edge[1])
        self.assertLess(high, particle_edge[0])
        self.assertLess(low, high)

    def test_closure_spread_is_not_called_a_bound(self):
        """The roadmap forbids labelling the spread a bound without a theorem."""
        gw = GW(self.mf, closure_spread=True)
        gw.kernel(5)
        record = gw.dyson_diagnostics["closure_spread"]

        self.assertFalse(record["is_a_bound"])
        self.assertIn("homo_spread", record)

    def test_closure_spread_is_independent_of_the_order_estimate(self):
        """The two indicators must not be the same number by another name."""
        gw = GW(self.mf, closure_spread=True, moment_order_convergence=True)
        gw.kernel(5)
        closure = gw.dyson_diagnostics["closure_spread"]["homo_spread"]
        differencing = gw.dyson_diagnostics["moment_order_convergence"]["homo_shift"]

        self.assertNotAlmostEqual(closure, differencing, 6)

    def test_closure_spread_unavailable_with_a_single_block(self):
        """One block has no leading part to pin against, so there is no second closure."""
        gw = GW(self.mf, closure_spread=True)
        gw.kernel(1)
        self.assertIsNone(gw.dyson_diagnostics["closure_spread"])

    def test_closure_spread_records_whether_self_consistency_was_excluded(self):
        """Both frontiers must get the same treatment or the difference is not the closure."""
        gw = GW(self.mf, closure_spread=True)
        gw.kernel(5)
        self.assertFalse(gw.dyson_diagnostics["closure_spread"]["self_consistent_excluded"])

        loop = GW(self.mf, closure_spread=True, fock_loop=True)
        loop.kernel(5)
        self.assertTrue(loop.dyson_diagnostics["closure_spread"]["self_consistent_excluded"])

    def test_closure_spread_compares_the_frontier_labels(self):
        """Two closures can order the frontier differently; that is not a spread."""
        gw = GW(self.mf, closure_spread=True)
        gw.kernel(5)
        record = gw.dyson_diagnostics["closure_spread"]

        self.assertIn("homo_orbital_changed", record)
        self.assertIsInstance(record["homo_orbital_changed"], bool)
        self.assertIn("homo_orbital", record["frontier_gauss"])
        self.assertIn("homo_orbital", record["frontier_radau"])

    def test_regression_fock_loop_nmom3(self):
        # Dyson's `Spectral` is shared with the Fock loop, which the recorded baseline
        # never exercises: `baseline/run.py` stores `fock_loop` as provenance but does not
        # vary it. `test_regression_fock_loop` above covers `nmom_max = 1`, where the
        # upfolded supermatrix is only `3 * nmo` and each sector contributes one block;
        # this covers an order where it is `5 * nmo` and the self-energy is built by
        # combining two sectors, which is the path a change to `Spectral` would move.
        ip = -0.280154354540313
        ea = 0.006296394214700909
        self._test_regression("hf", dict(fock_loop=True), 3, ip, ea, "fock loop, nmom_max=3")

    def test_diagonal_pbe0(self):
        ip = -0.26182940618925504
        ea = 0.008140559373415861
        self._test_regression("pbe0", dict(diagonal_se=True), 5, ip, ea, "diagonal pbe0")

    def test_regression_tda(self):
        ip = -0.27310320793161513
        ea = 0.005268331141340351
        self._test_regression("hf", dict(polarizability="dtda"), 7, ip, ea, "tda")


if __name__ == "__main__":
    print("Running tests for GW")
    unittest.main()
