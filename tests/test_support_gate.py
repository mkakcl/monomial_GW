"""Tests for the support gate on `converged`.

A realization can conserve every order it was asked for, reproduce nothing, and place a
pole far outside the support its own moments imply. The residual gate catches the second
of those after the fact; this one catches the third directly, and names the sector.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import dft, gto

from momentGW import GW
from momentGW.gw import SUPPORT_MAX, moment_support_bound, realization_record


@pytest.fixture(scope="module")
def mf():
    """Build a small restricted mean field, density fitted."""
    mol = gto.M(atom="H 0 0 0; Li 0 0 1.6", basis="sto3g", verbose=0)
    mean_field = dft.RKS(mol, xc="hf").density_fit()
    mean_field.conv_tol = 1e-11
    mean_field.kernel()
    return mean_field


def _run(mf, nmom_max=3, **kwargs):
    """Run a one-shot GW and hand back its diagnostics."""
    solver = GW(mf, polarizability="drpa", **kwargs)
    solver.verbose = 0
    solver.kernel(nmom_max=nmom_max)
    return solver.dyson_diagnostics


def _patch_record(monkeypatch, transform):
    """Route `realization_record` through `transform` before it is stored."""
    original = realization_record

    def wrapped(solver, se_moments):
        return transform(original(solver, se_moments))

    monkeypatch.setattr("momentGW.gw.realization_record", wrapped)


class TestBound:
    """The bound is a property of the moments, and is a bound."""

    def test_bounds_a_known_spectrum_from_below(self):
        """Built from poles we choose, it must not exceed the largest of them."""
        energies = np.array([-3.0, -0.5, 0.25, 2.0])
        couplings = np.eye(4)
        moments = np.array([couplings @ np.diag(energies**n) @ couplings.T for n in range(8)])

        bound = moment_support_bound(moments)

        assert 0.0 < bound <= np.abs(energies).max()

    def test_tightens_with_order(self):
        """More even pairs can only improve the bound, never worsen it."""
        energies = np.array([-3.0, -0.5, 0.25, 2.0])
        couplings = np.eye(4)
        moments = np.array([couplings @ np.diag(energies**n) @ couplings.T for n in range(12)])

        assert moment_support_bound(moments[:4]) <= moment_support_bound(moments)

    def test_odd_orders_would_break_it(self):
        """The even-order restriction is load-bearing, not tidiness.

        Odd powers can be negative, so a near-cancelling odd trace makes the ratio
        arbitrarily large. This spectrum is chosen so that admitting odd pairs returns
        2.65 for a spectrum whose largest pole is 2.0 - no longer a bound at all.
        """
        energies = np.array([-1.0, 2.0])
        couplings = np.eye(2)
        moments = np.array([couplings @ np.diag(energies**n) @ couplings.T for n in range(6)])

        assert moment_support_bound(moments) <= np.abs(energies).max()

        traces = np.array([float(np.trace(m)) for m in moments])
        odd_admitted = max(
            float(np.sqrt(traces[n + 2] / traces[n]))
            for n in range(traces.size - 2)
            if traces[n] > 0.0 and traces[n + 2] > 0.0
        )
        assert odd_admitted > np.abs(energies).max()

    def test_unavailable_below_three_moments(self):
        """Two moments carry no even pair, so there is nothing to bound with."""
        assert moment_support_bound(np.ones((2, 2, 2))) == 0.0


class TestHealthy:
    """A sound realization passes and reports its ratio."""

    def test_gate_passes(self, mf):
        diagnostics = _run(mf)

        assert diagnostics["gates"]["support"] is True
        assert diagnostics["converged"] is True

    def test_ratio_is_reported_per_sector(self, mf):
        diagnostics = _run(mf)

        assert set(diagnostics["support_ratio"]) == {"hole", "particle"}
        for value in diagnostics["support_ratio"].values():
            assert value is not None
            assert 0.0 < value <= SUPPORT_MAX
        assert diagnostics["support_max_ratio"] == SUPPORT_MAX


class TestNotApplicable:
    """Below three moments the check does not exist, and must not vote."""

    def test_gate_passes_when_the_bound_is_unavailable(self, mf):
        diagnostics = _run(mf, nmom_max=1)

        assert all(value is None for value in diagnostics["support_ratio"].values())
        assert diagnostics["gates"]["support"] is True

    def test_not_applicable_is_none_not_a_number(self, mf):
        """`None` and a passing ratio must not be confused in the readout."""
        diagnostics = _run(mf, nmom_max=1)

        assert diagnostics["support_ratio"] == {"hole": None, "particle": None}


class TestSpuriousPole:
    """The failure the gate exists for."""

    @staticmethod
    def _spurious(record):
        """Push one sector's largest pole far outside the bound its moments imply."""
        record["support_ratio"] = float(SUPPORT_MAX * 4.0)
        record["support_max"] = record["support_bound"] * SUPPORT_MAX * 4.0
        return record

    def test_gate_fails(self, mf, monkeypatch):
        _patch_record(monkeypatch, self._spurious)

        diagnostics = _run(mf)

        assert diagnostics["gates"]["support"] is False
        assert diagnostics["converged"] is False

    def test_other_gates_still_pass(self, mf, monkeypatch):
        """The point of a separate gate: order and residual cannot see this."""
        _patch_record(monkeypatch, self._spurious)

        diagnostics = _run(mf)

        assert diagnostics["gates"]["realization"] is True
        assert diagnostics["gates"]["residual"] is True
        assert diagnostics["gates"]["support"] is False

    def test_one_bad_sector_is_enough(self, mf, monkeypatch):
        """A sound hole must not carry a spurious particle through the gate."""
        state = {"seen": 0}

        def only_second(record):
            state["seen"] += 1
            return self._spurious(record) if state["seen"] == 2 else record

        _patch_record(monkeypatch, only_second)

        diagnostics = _run(mf)

        ratios = diagnostics["support_ratio"]
        assert min(ratios.values()) <= SUPPORT_MAX < max(ratios.values())
        assert diagnostics["gates"]["support"] is False
