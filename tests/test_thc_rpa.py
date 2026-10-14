"""Tests for `thc_rpa.py`: the separable assembly, and the dRPA solver built on it.

The reference throughout is an explicit `O(N^4)` double loop over the particle-hole pairs
-- the thing the module exists not to do -- so that the separable assembly is checked
against the definition rather than against a cleverer version of itself.

Two claims are separated deliberately.  Against the *same* fitted scalar function of the
gaps, the assembly must agree with the definition to machine precision: the binomial
expansion of ``D^m`` is exact, so anything above roundoff there is a bug in the expansion,
the sector factorisation or the blocking.  Against the *true* resolvent it must agree to
the residual `laplace` measured, at every moment order, which is the property that lets one
scalar certify the whole set.
"""

import unittest

import numpy as np
import scipy.linalg

from momentGW import eta0, laplace, thc_rpa


def gaps(n_occ=6, n_vir=24, gap=0.4, spread=18.0, seed=0):
    """Build a synthetic but realistically spread set of orbital energies.

    The spread is what makes the test meaningful: a core-like occupied energy is where the
    exponential factor underflows and where ``(-ebar_i)**p`` is largest, so a sign or
    reference-point error shows up there and nowhere else.
    """
    rng = np.random.default_rng(seed)
    e_occ = -np.sort(rng.uniform(0.0, spread, n_occ))[::-1] - gap / 2
    e_vir = np.sort(rng.uniform(0.0, spread, n_vir)) + gap / 2
    return e_occ, e_vir


def collocation(n_occ=6, n_vir=24, naux=30, seed=1):
    """Build synthetic collocation arrays with the shape momentGW would supply."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(naux)
    return rng.standard_normal((naux, n_occ)) * scale, rng.standard_normal((naux, n_vir)) * scale


def shift_set(d, n=12):
    """Build shifts spanning the range an HHT pole set would cover."""
    return np.geomspace(d.min() ** 2, d.max() ** 2 * 4.0, n)


def reference(coll_occ, coll_vir, f):
    r"""Evaluate ``sum_ia X_iP X_iQ X_aP X_aQ f_ia`` by an explicit loop over the pairs.

    Quartic in operations and deliberately so.  `f` is ``(nocc, nvir)``.
    """
    naux = coll_occ.shape[0]
    out = np.zeros((naux, naux))
    for i in range(coll_occ.shape[1]):
        occ = np.outer(coll_occ[:, i], coll_occ[:, i])
        for a in range(coll_vir.shape[1]):
            out += occ * np.outer(coll_vir[:, a], coll_vir[:, a]) * f[i, a]
    return out


class Test_THCRPA(unittest.TestCase):
    def setUp(self):
        self.e_occ, self.e_vir = gaps()
        self.coll_occ, self.coll_vir = collocation()
        self.d = (self.e_vir[None, :] - self.e_occ[:, None]).ravel()
        self.shifts = shift_set(self.d)
        self.orders = (0, 2, 4, 6, 8)
        self.t, self.coeffs, self.diagnostics = laplace.build(self.d, self.shifts, 1e-10)

    def fitted(self):
        """Get the fitted resolvent on the gaps, shape ``(n_shift, nocc, nvir)``."""
        basis = np.exp(-np.outer(self.d, self.t))
        return (basis @ self.coeffs.T).T.reshape(-1, *self.d.reshape(6, 24).shape)

    def test_assembly_is_exact_against_the_definition(self):
        # Against the same fitted scalar, the assembly is an identity: D^m expands
        # binomially into separable products, so only roundoff separates the two.
        q = thc_rpa.build_q(
            self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, self.t, self.coeffs, self.orders
        )
        d = self.d.reshape(6, 24)
        fit = self.fitted()

        for j in range(self.shifts.size):
            for mi, m in enumerate(self.orders):
                exact = reference(self.coll_occ, self.coll_vir, d**m * fit[j])
                error = np.max(np.abs(q[j, mi] - exact)) / np.max(np.abs(exact))
                self.assertLess(error, 1e-12, msg=f"shift {j}, order {m}")

    def test_matches_the_resolvent_within_the_measured_residual(self):
        # Against the true D^(m+1)/(D^2+s), the error is the one laplace measured, at
        # every order -- which is the whole design claim.
        q = thc_rpa.build_q(
            self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, self.t, self.coeffs, self.orders
        )
        d = self.d.reshape(6, 24)
        residual = self.diagnostics["residual"]

        for j, shift in enumerate(self.shifts):
            for mi, m in enumerate(self.orders):
                exact = reference(self.coll_occ, self.coll_vir, d ** (m + 1) / (d**2 + shift))
                relative = np.linalg.norm(q[j, mi] - exact, 2) / np.linalg.norm(exact, 2)
                self.assertLessEqual(relative, residual * 1.5, msg=f"shift {j}, order {m}")

    def test_error_is_flat_in_the_moment_order(self):
        # Route B rather than polynomial division: order 8 is no worse than order 0.
        # Division loses about 1.5 decades per order, so this fails loudly if it creeps in.
        q = thc_rpa.build_q(
            self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, self.t, self.coeffs, self.orders
        )
        d = self.d.reshape(6, 24)

        errors = []
        for mi, m in enumerate(self.orders):
            worst = 0.0
            for j, shift in enumerate(self.shifts):
                exact = reference(self.coll_occ, self.coll_vir, d ** (m + 1) / (d**2 + shift))
                worst = max(worst, np.linalg.norm(q[j, mi] - exact, 2) / np.linalg.norm(exact, 2))
            errors.append(worst)

        self.assertLess(max(errors), 10.0 * errors[0])

    def test_blocking_changes_nothing(self):
        # The stream is the production path and the unblocked form is the convenience, so
        # they have to be the same object, not merely close.
        args = (
            self.coll_occ,
            self.coll_vir,
            self.e_occ,
            self.e_vir,
            self.t,
            self.coeffs,
            self.orders,
        )
        whole = thc_rpa.build_q(*args, block=self.coll_occ.shape[0])

        for block in (1, 7, 13, 1000):
            other = thc_rpa.build_q(*args, block=block)
            self.assertTrue(np.allclose(whole, other, rtol=0.0, atol=1e-14 * np.max(np.abs(whole))))

    def test_stream_covers_every_column_once(self):
        naux = self.coll_occ.shape[0]
        seen = np.zeros(naux, dtype=int)
        for cols, chunk in thc_rpa.q_blocks(
            self.coll_occ,
            self.coll_vir,
            self.e_occ,
            self.e_vir,
            self.t,
            self.coeffs,
            self.orders,
            block=7,
        ):
            seen[cols] += 1
            self.assertEqual(
                chunk.shape,
                (self.shifts.size, len(self.orders), naux, cols.stop - cols.start),
            )
        self.assertTrue(np.all(seen == 1))

    def test_block_size_respects_the_budget(self):
        # The point of deriving the width from bytes rather than from a shape: the block
        # is the widest one that fits, and a single column is the floor below which the
        # budget simply cannot be honoured.
        for naux in (32, 500, 4000):
            for budget in (1 << 20, 1 << 26, 1 << 34):
                block = thc_rpa.block_size(naux, 33, 18, 5, 10, budget=budget)
                self.assertGreaterEqual(block, 1)
                self.assertLessEqual(block, naux)

                per_column = 8 * naux * ((33 + 18) * 5 + 2 * 10)
                if block > 1:
                    self.assertLessEqual(per_column * block, budget)
                if block < naux:
                    self.assertGreater(per_column * (block + 1), budget)

    def test_operator_bound_carries_through_the_factorisation(self):
        # Q_X = W diag(f) W^T with W_{P,ia} = X_iP X_aP, so it is positive semidefinite
        # and laplace's relative bound applies to it exactly as it does to the RI vertex.
        q = thc_rpa.build_q(
            self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, self.t, self.coeffs, self.orders
        )
        w = (self.coll_occ[:, :, None] * self.coll_vir[:, None, :]).reshape(
            self.coll_occ.shape[0], -1
        )

        for j, shift in enumerate(self.shifts):
            g = self.d / (self.d**2 + shift)
            for mi, m in enumerate(self.orders):
                exact = (w * (self.d**m * g)[None]) @ w.T
                self.assertGreater(np.min(np.linalg.eigvalsh(exact)), -1e-10 * np.abs(exact).max())
                error = np.linalg.norm(q[j, mi] - exact, 2)
                self.assertLessEqual(
                    error, self.diagnostics["residual"] * np.linalg.norm(exact, 2) * 1.5
                )

    def test_against_a_certified_pole_set(self):
        # The shifts a calculation actually sees, rather than a geometric stand-in.
        w = (self.coll_occ[:, :, None] * self.coll_vir[:, None, :]).reshape(
            self.coll_occ.shape[0], -1
        )
        coupling = 4.0 * float(np.sum(self.d * np.sum(w * w, axis=0)))
        lmin, lmax = eta0.certified_interval(self.d, coupling)
        shifts, weights, *_ = eta0.select_poles(lmin, lmax, 1e-12)
        t, coeffs, diagnostics = laplace.build(self.d, shifts, 1e-10)

        q = thc_rpa.build_q(
            self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, t, coeffs, self.orders
        )
        d = self.d.reshape(6, 24)
        for j, shift in enumerate(shifts):
            for mi, m in enumerate(self.orders):
                exact = reference(self.coll_occ, self.coll_vir, d ** (m + 1) / (d**2 + shift))
                relative = np.linalg.norm(q[j, mi] - exact, 2) / np.linalg.norm(exact, 2)
                self.assertLessEqual(relative, diagnostics["residual"] * 1.5)

    def test_odd_orders_are_not_special_cased(self):
        # The dRPA only needs even m, but nothing in the assembly assumes it, and the
        # recursion that follows will want the shift-independent odd images too.
        orders = (1, 3, 5)
        q = thc_rpa.build_q(
            self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, self.t, self.coeffs, orders
        )
        d = self.d.reshape(6, 24)
        fit = self.fitted()
        for mi, m in enumerate(orders):
            exact = reference(self.coll_occ, self.coll_vir, d**m * fit[0])
            self.assertLess(np.max(np.abs(q[0, mi] - exact)) / np.max(np.abs(exact)), 1e-12)

    def test_potential_must_separate_the_sectors(self):
        with self.assertRaises(ValueError):
            thc_rpa.build_q(
                self.coll_occ,
                self.coll_vir,
                self.e_occ,
                self.e_vir,
                self.t,
                self.coeffs,
                self.orders,
                mu=float(self.e_vir[3]) + 1.0,
            )

    def test_argument_validation(self):
        args = (self.coll_occ, self.coll_vir, self.e_occ, self.e_vir)
        with self.assertRaises(ValueError):  # coefficients against the wrong grid
            thc_rpa.build_q(*args, self.t[:-1], self.coeffs, self.orders)
        with self.assertRaises(ValueError):  # collocation arrays of different rank
            thc_rpa.build_q(
                self.coll_occ[:-1],
                self.coll_vir,
                self.e_occ,
                self.e_vir,
                self.t,
                self.coeffs,
                self.orders,
            )
        with self.assertRaises(ValueError):  # negative moment order
            thc_rpa.build_q(*args, self.t, self.coeffs, (-1,))
        with self.assertRaises(ValueError):  # an empty sector
            thc_rpa.build_q(
                self.coll_occ[:, :0],
                self.coll_vir,
                self.e_occ[:0],
                self.e_vir,
                self.t,
                self.coeffs,
                self.orders,
            )


if __name__ == "__main__":
    unittest.main()


def pole_set(d, v, tol=1e-14):
    """Get the certified pole set for a vertex, the way a solver would."""
    coupling = 4.0 * float(np.sum(d * np.sum(v * v, axis=0)))
    lmin, lmax = eta0.certified_interval(d, coupling)
    shifts, weights, *_ = eta0.select_poles(lmin, lmax, tol)
    return shifts, weights


class Test_THCdRPA(unittest.TestCase):
    """The solver layer: the pole solve, the recursion, and the identity behind both."""

    def setUp(self):
        self.e_occ, self.e_vir = gaps()
        self.coll_occ, self.coll_vir = collocation()
        self.d = (self.e_vir[None, :] - self.e_occ[:, None]).ravel()
        self.naux = self.coll_occ.shape[0]
        self.nmom_max = 7

        # A positive definite Coulomb matrix, so that the Cholesky factor the
        # push-through identity removes actually exists to compare against.
        rng = np.random.default_rng(3)
        a = rng.standard_normal((self.naux, self.naux))
        self.cou = a @ a.T / self.naux + np.eye(self.naux)

        self.w = (self.coll_occ[:, :, None] * self.coll_vir[:, None, :]).reshape(self.naux, -1)

    def families(self, v, shifts, weights, cou=None):
        """Build both families densely from a pair-index vertex.

        With `cou` given the vertex is hypercontracted and the solve carries `Z`; without
        it the vertex is the density-fitted one and the solve is the plain form.
        """
        naux = v.shape[0]
        eye = np.eye(naux)
        a_x = np.zeros((self.nmom_max + 1, naux, naux))

        def q(k, s):
            return (v * (self.d**k / (self.d**2 + s))[None]) @ v.T

        for m in range(self.nmom_max + 1):
            if m % 2:
                a_x[m] = (v * (self.d**m)[None]) @ v.T
            else:
                for s, weight in zip(shifts, weights):
                    inner = q(1, s) @ cou if cou is not None else q(1, s)
                    a_x[m] += weight * scipy.linalg.solve(eye + 4.0 * inner, q(m + 1, s))
        return a_x

    def test_recursion_matches_the_pair_space_form(self):
        # `combine` against the recursion `momentGW.rpa.dRPA` runs over the pair index --
        # a reference that shares none of the auxiliary-space route.
        rng = np.random.default_rng(11)
        v = rng.standard_normal((self.naux, self.d.size)) / np.sqrt(self.naux)
        shifts, weights = pole_set(self.d, v)

        a_x = self.families(v, shifts, weights)
        zeta = thc_rpa.combine(a_x, np.eye(self.naux))

        # The pair-space form: moments carrying the particle-hole index throughout
        eye = np.eye(self.naux)
        moments = np.zeros((self.nmom_max + 1, self.naux, self.d.size))
        for s, weight in zip(shifts, weights):
            f = self.d / (self.d**2 + s)
            y = v * f[None]
            moments[0] += weight * scipy.linalg.solve(eye + 4.0 * (y @ v.T), y)
        moments[1] = v * self.d[None]
        for i in range(2, self.nmom_max + 1):
            moments[i] = moments[i - 2] * self.d[None] ** 2
            moments[i] += 4.0 * ((moments[i - 2] @ v.T) @ moments[1])

        for m in range(self.nmom_max + 1):
            exact = moments[m] @ v.T
            relative = np.linalg.norm(zeta[m] - exact) / np.linalg.norm(exact)
            self.assertLess(relative, 1e-11, msg=f"order {m}")

    def test_push_through_removes_the_cholesky_factor(self):
        # The identity the whole solver convention rests on:
        #   (I + 4 L^T Q_X L)^-1 L^T Q_X' L  ==  L^T [(I + 4 Q_X Z)^-1 Q_X'] L
        # `cholesky(lower=False)` returns U with Z = U^T U, so L = U^T and L^T = U.
        upper = scipy.linalg.cholesky(self.cou, lower=False)
        self.assertTrue(np.allclose(self.cou, upper.T @ upper))

        shifts, weights = pole_set(self.d, upper @ self.w)
        with_factor = self.families(upper @ self.w, shifts, weights)
        without = self.families(self.w, shifts, weights, cou=self.cou)

        for m in range(0, self.nmom_max + 1, 2):
            rotated = upper @ without[m] @ upper.T
            relative = np.linalg.norm(rotated - with_factor[m]) / np.linalg.norm(with_factor[m])
            self.assertLess(relative, 1e-10, msg=f"order {m}")

    def test_screened_moments_match_a_dense_solve(self):
        # The streamed, LU-factored, blocked route against an explicit dense one.
        shifts, weights = pole_set(self.d, self.w)
        t, coeffs, diagnostics = laplace.build(self.d, shifts, 1e-10)
        orders = [0, 2, 4, 6]

        got = thc_rpa.screened_moments(
            self.coll_occ,
            self.coll_vir,
            self.e_occ,
            self.e_vir,
            self.cou,
            weights,
            t,
            coeffs,
            orders,
        )

        # Dense reference from the already-verified assembly, so only the solve is new
        q = thc_rpa.build_q(self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, t, coeffs, orders)
        eye = np.eye(self.naux)
        for mi, m in enumerate(orders):
            exact = np.zeros((self.naux, self.naux))
            for j, weight in enumerate(weights):
                exact += weight * scipy.linalg.solve(eye + 4.0 * (q[j, 0] @ self.cou), q[j, mi])
            relative = np.linalg.norm(got[mi] - exact) / np.linalg.norm(exact)
            self.assertLess(relative, 1e-11, msg=f"order {m}")
        self.assertLessEqual(diagnostics["residual"], 1e-10)

    def test_screened_moments_blocking_changes_nothing(self):
        shifts, weights = pole_set(self.d, self.w)
        t, coeffs, _ = laplace.build(self.d, shifts, 1e-9)
        args = (
            self.coll_occ,
            self.coll_vir,
            self.e_occ,
            self.e_vir,
            self.cou,
            weights,
            t,
            coeffs,
            [0, 2, 4],
        )
        whole = thc_rpa.screened_moments(*args, block=self.naux)
        for block in (1, 7, 13):
            other = thc_rpa.screened_moments(*args, block=block)
            self.assertTrue(np.allclose(whole, other, rtol=0.0, atol=1e-13 * np.abs(whole).max()))

    def test_screened_moments_accept_a_strided_destination(self):
        # The solver writes into the even slots of the combined array, which is a view
        # with a stride of two -- so that has to be an accepted destination, not an
        # accident that happens to work.
        shifts, weights = pole_set(self.d, self.w)
        t, coeffs, _ = laplace.build(self.d, shifts, 1e-9)
        args = (
            self.coll_occ,
            self.coll_vir,
            self.e_occ,
            self.e_vir,
            self.cou,
            weights,
            t,
            coeffs,
            [0, 2, 4, 6],
        )
        a_x = np.zeros((self.nmom_max + 1, self.naux, self.naux))
        thc_rpa.screened_moments(*args, out=a_x[0::2])
        self.assertTrue(np.allclose(a_x[0::2], thc_rpa.screened_moments(*args)))
        self.assertTrue(np.all(a_x[1::2] == 0.0))

    def test_screened_moments_need_the_zeroth_order(self):
        shifts, weights = pole_set(self.d, self.w)
        t, coeffs, _ = laplace.build(self.d, shifts, 1e-9)
        with self.assertRaises(ValueError):
            thc_rpa.screened_moments(
                self.coll_occ,
                self.coll_vir,
                self.e_occ,
                self.e_vir,
                self.cou,
                weights,
                t,
                coeffs,
                [2, 4],
            )

    def test_polynomial_moments_are_the_plain_binomial(self):
        orders = [1, 3, 5, 7]
        got = thc_rpa.polynomial_moments(
            self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, orders
        )
        d = self.d.reshape(6, 24)
        for mi, m in enumerate(orders):
            exact = reference(self.coll_occ, self.coll_vir, d**m)
            self.assertLess(np.max(np.abs(got[mi] - exact)) / np.max(np.abs(exact)), 1e-12)

    def test_combine_is_the_identity_at_the_lowest_orders(self):
        # zeta^(0) and zeta^(1) carry no recursion term, and zeta^(2) carries exactly one.
        rng = np.random.default_rng(5)
        a_x = rng.standard_normal((3, self.naux, self.naux))
        zeta = thc_rpa.combine(a_x, self.cou)
        self.assertTrue(np.allclose(zeta[0], a_x[0]))
        self.assertTrue(np.allclose(zeta[1], a_x[1]))
        self.assertTrue(
            np.allclose(zeta[2], a_x[2] + 4.0 * np.linalg.multi_dot((a_x[0], self.cou, a_x[1])))
        )

    def test_combine_handles_a_single_moment(self):
        a_x = np.random.default_rng(6).standard_normal((1, self.naux, self.naux))
        self.assertTrue(np.allclose(thc_rpa.combine(a_x, self.cou), a_x))

    def test_every_moment_order_is_filled_at_every_nmom_max(self):
        # The two families are interleaved by parity, so an off-by-one in either slice
        # would leave a slot at zero rather than raise.  Check both slices cover the
        # array, at every length, including the degenerate ones.
        shifts, weights = pole_set(self.d, self.w)
        t, coeffs, _ = laplace.build(self.d, shifts, 1e-9)

        for nmom_max in range(0, 8):
            even = list(range(0, nmom_max + 1, 2))
            odd = list(range(1, nmom_max + 1, 2))
            a_x = np.zeros((nmom_max + 1, self.naux, self.naux))

            polynomial = thc_rpa.polynomial_moments(
                self.coll_occ, self.coll_vir, self.e_occ, self.e_vir, odd or [1]
            )
            if odd:
                a_x[1::2] = polynomial
            thc_rpa.screened_moments(
                self.coll_occ,
                self.coll_vir,
                self.e_occ,
                self.e_vir,
                self.cou,
                weights,
                t,
                coeffs,
                even,
                out=a_x[0::2],
            )

            self.assertEqual(len(even) + len(odd), nmom_max + 1, msg=f"nmom_max {nmom_max}")
            for m in range(nmom_max + 1):
                self.assertTrue(np.any(a_x[m] != 0.0), msg=f"nmom_max {nmom_max}, order {m}")

            zeta = thc_rpa.combine(a_x, self.cou)
            self.assertEqual(zeta.shape, (nmom_max + 1, self.naux, self.naux))
            self.assertTrue(np.all(np.isfinite(zeta)))
