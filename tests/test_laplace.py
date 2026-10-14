"""Tests for `laplace.py`, the separable representation of the particle-hole resolvent.

Covers what the module promises: that the measured residual is realised on the true gaps
rather than a bound, that the grid search returns the smallest size meeting a tolerance,
that an unrepresentable tolerance fails immediately, that the selection is genuinely
better than a geometric layout, and -- the claim the whole design rests on -- that one
scalar residual certifies the relative operator error of `Q^(m+1)` at every moment order,
not just the zeroth.
"""

import unittest

import numpy as np

from momentGW import laplace


def gaps(n_occ=6, n_vir=24, gap=0.4, spread=18.0, seed=0):
    """Build a synthetic but realistically spread set of particle-hole gaps."""
    rng = np.random.default_rng(seed)
    e_occ = -np.sort(rng.uniform(0.0, spread, n_occ))[::-1] - gap / 2
    e_vir = np.sort(rng.uniform(0.0, spread, n_vir)) + gap / 2
    return (e_vir[None, :] - e_occ[:, None]).ravel()


def vertex(nov, naux=40, seed=1):
    """Build a synthetic vertex with the shape momentGW would supply."""
    return np.random.default_rng(seed).standard_normal((naux, nov)) / np.sqrt(naux)


def shift_set(d, n=12):
    """Build shifts spanning the range an HHT pole set would cover."""
    return np.geomspace(d.min() ** 2, d.max() ** 2 * 4.0, n)


class Test_Laplace(unittest.TestCase):
    def test_residual_is_realised_not_a_bound(self):
        # The reported residual must be exactly the largest relative deviation on the
        # gaps that were fitted -- recompute it independently.
        d = gaps()
        shifts = shift_set(d)
        t, coeffs, diagnostics = laplace.build(d, shifts, 1e-9)

        basis = np.exp(-np.outer(d, t))
        worst = 0.0
        for j, shift in enumerate(shifts):
            g = d / (d**2 + shift)
            worst = max(worst, np.max(np.abs(basis @ coeffs[j] - g) / g))

        self.assertAlmostEqual(worst, diagnostics["residual"], places=14)
        self.assertLessEqual(diagnostics["residual"], 1e-9)

    def test_operator_bound_holds_at_every_order(self):
        # ||dQ^(m+1)|| <= residual * ||Q^(m+1)|| for every m, which is what lets one
        # scalar number certify the whole moment set.
        d = gaps()
        shifts = shift_set(d)
        v = vertex(d.size)
        t, coeffs, diagnostics = laplace.build(d, shifts, 1e-10)
        basis = np.exp(-np.outer(d, t))

        for j, shift in enumerate(shifts):
            g = d / (d**2 + shift)
            fit = basis @ coeffs[j]
            for m in range(0, 10, 2):
                exact = (v * (d**m * g)[None]) @ v.T
                approx = (v * (d**m * fit)[None]) @ v.T
                relative = np.linalg.norm(approx - exact, 2) / np.linalg.norm(exact, 2)
                self.assertLessEqual(relative, diagnostics["residual"] * 1.5)

    def test_error_does_not_degrade_with_moment_order(self):
        # The point of the binomial route: unlike the polynomial-division route, the
        # relative error at order 8 is no worse than at order 0.
        d = gaps()
        shifts = shift_set(d)
        v = vertex(d.size)
        t, coeffs, _ = laplace.build(d, shifts, 1e-10)
        basis = np.exp(-np.outer(d, t))

        errors = []
        for m in range(0, 10, 2):
            worst = 0.0
            for j, shift in enumerate(shifts):
                g = d / (d**2 + shift)
                exact = (v * (d**m * g)[None]) @ v.T
                approx = (v * (d**m * (basis @ coeffs[j]))[None]) @ v.T
                worst = max(worst, np.linalg.norm(approx - exact, 2) / np.linalg.norm(exact, 2))
            errors.append(worst)

        self.assertLess(max(errors), 10.0 * errors[0])

    def test_search_returns_the_smallest_grid(self):
        d = gaps()
        shifts = shift_set(d)
        t, _, diagnostics = laplace.build(d, shifts, 1e-8)

        _, residual_smaller, _ = laplace.fit(d, shifts, laplace.select_exponents(d, t.size - 1))
        self.assertGreater(residual_smaller, 1e-8)
        self.assertLessEqual(diagnostics["residual"], 1e-8)

    def test_tighter_tolerance_needs_more_points(self):
        d = gaps()
        shifts = shift_set(d)
        sizes = [laplace.build(d, shifts, tol)[2]["n_points"] for tol in (1e-6, 1e-8, 1e-10)]
        self.assertEqual(sizes, sorted(sizes))

    def test_selection_beats_a_geometric_grid(self):
        # The claim in the module docstring: pivoting reaches the floor with fewer points.
        d = gaps()
        shifts = shift_set(d)
        n_points = 30

        _, selected, _ = laplace.fit(d, shifts, laplace.select_exponents(d, n_points))
        geometric = np.geomspace(1e-2 / d.max(), 30.0 / d.min(), n_points)
        _, laid_out, _ = laplace.fit(d, shifts, geometric)

        self.assertLess(selected, laid_out)

    def test_unrepresentable_tolerance_raises(self):
        d = gaps()
        with self.assertRaises(ValueError):
            laplace.build(d, shift_set(d), laplace.TOL_FLOOR / 10.0)

    def test_fixed_grid_size_is_an_instruction(self):
        # A fixed size must be honoured and its miss reported, not raised.
        d = gaps()
        shifts = shift_set(d)
        t, coeffs, diagnostics = laplace.build(d, shifts, 1e-12, n_points=8)

        self.assertEqual(t.size, 8)
        self.assertEqual(coeffs.shape, (shifts.size, 8))
        self.assertTrue(diagnostics["n_points_fixed"])
        self.assertGreater(diagnostics["residual"], 1e-12)

    def test_unreachable_tolerance_raises(self):
        d = gaps()
        with self.assertRaises(RuntimeError):
            laplace.build(d, shift_set(d), 1e-12, n_points_max=10)

    def test_single_shift(self):
        d = gaps()
        t, coeffs, diagnostics = laplace.build(d, np.array([1.0]), 1e-9)
        self.assertEqual(coeffs.shape, (1, t.size))
        self.assertEqual(diagnostics["n_shifts"], 1)

    def test_gap_validation(self):
        with self.assertRaises(ValueError):
            laplace.candidate_exponents(np.array([1.0, 0.0, 2.0]))
        with self.assertRaises(ValueError):
            laplace.candidate_exponents(np.array([1.0, -1.0]))
        with self.assertRaises(ValueError):
            laplace.candidate_exponents(np.array([1.0, np.inf]))

    def test_exponents_are_sorted_and_positive(self):
        d = gaps()
        t = laplace.select_exponents(d, 24)
        self.assertEqual(t.size, 24)
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertTrue(np.all(t > 0))

    def test_selection_saturates_at_the_candidate_count(self):
        d = gaps()
        t = laplace.select_exponents(d, laplace.N_CANDIDATE + 50)
        self.assertEqual(t.size, laplace.N_CANDIDATE)

    def test_diagnostics_are_complete(self):
        d = gaps()
        t, _, diagnostics = laplace.build(d, shift_set(d), 1e-9)
        for key in (
            "n_points",
            "residual",
            "tol",
            "amplification",
            "n_shifts",
            "gap_range",
            "exponent_range",
            "n_points_fixed",
        ):
            self.assertIn(key, diagnostics)
        self.assertEqual(diagnostics["gap_range"], (float(d.min()), float(d.max())))
        self.assertEqual(diagnostics["exponent_range"], (float(t[0]), float(t[-1])))

    def test_report_runs(self):
        d = gaps()
        _, _, diagnostics = laplace.build(d, shift_set(d), 1e-9)
        laplace.report(diagnostics)
        diagnostics["n_points_fixed"] = True
        diagnostics["tol"] = diagnostics["residual"] / 10.0
        laplace.report(diagnostics)

    def test_wide_gap_range(self):
        # A core orbital pushes the range out by three decades; the search has to cope.
        d = gaps(spread=200.0, gap=0.1)
        t, _, diagnostics = laplace.build(d, shift_set(d), 1e-8)
        self.assertLessEqual(diagnostics["residual"], 1e-8)
        self.assertLess(t.size, laplace.N_POINTS_MAX)


if __name__ == "__main__":
    print("Running tests for laplace")
    unittest.main()
