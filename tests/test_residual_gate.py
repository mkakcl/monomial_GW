"""Tests for the reconstructed-moment residual gate on `converged`.

A sector can conserve every order it was asked for and still not reproduce the moments it
was given. The realization gate reads the conserved order and cannot see that, so the
residual is gated separately; these tests pin that it is.
"""

from __future__ import annotations

import dataclasses

import pytest
from pyscf import dft, gto

from momentGW import GW
from momentGW.gw import RESIDUAL_MAX, realization_record


@pytest.fixture(scope="module")
def mf():
    """Build a small restricted mean field, density fitted."""
    mol = gto.M(atom="H 0 0 0; Li 0 0 1.6", basis="sto3g", verbose=0)
    mean_field = dft.RKS(mol, xc="hf").density_fit()
    mean_field.conv_tol = 1e-11
    mean_field.kernel()
    return mean_field


def _run(mf, **kwargs):
    """Run a one-shot GW and hand back its diagnostics."""
    solver = GW(mf, polarizability="drpa", **kwargs)
    solver.verbose = 0
    solver.kernel(nmom_max=1)
    return solver.dyson_diagnostics


def _patch_record(monkeypatch, transform):
    """Route `realization_record` through `transform` before it is stored."""
    original = realization_record

    def wrapped(solver, se_moments):
        return transform(original(solver, se_moments))

    monkeypatch.setattr("momentGW.gw.realization_record", wrapped)


class TestHealthy:
    """A sound realization passes, and says what it measured."""

    def test_gate_passes(self, mf):
        diagnostics = _run(mf)

        assert diagnostics["gates"]["residual"] is True
        assert diagnostics["converged"] is True

    def test_residual_is_reported_per_sector(self, mf):
        diagnostics = _run(mf)

        assert set(diagnostics["residual"]) == {"hole", "particle"}
        for value in diagnostics["residual"].values():
            assert value is not None
            assert value < RESIDUAL_MAX
        assert diagnostics["residual_max"] == RESIDUAL_MAX


class TestBlownResidual:
    """The failure the gate exists for: every order conserved, moments not reproduced."""

    @staticmethod
    def _blow(record):
        """Inflate one sector's residual without touching its conserved order."""
        errors = record["errors"]
        record["errors"] = dataclasses.replace(
            errors, relative_frobenius=tuple([1e4] * len(errors.relative_frobenius))
        )
        return record

    def test_gate_fails(self, mf, monkeypatch):
        _patch_record(monkeypatch, self._blow)

        diagnostics = _run(mf)

        assert diagnostics["gates"]["residual"] is False
        assert diagnostics["converged"] is False

    def test_realization_gate_still_passes(self, mf, monkeypatch):
        """The point of a separate gate: the order-based one cannot see this."""
        _patch_record(monkeypatch, self._blow)

        diagnostics = _run(mf)

        assert diagnostics["gates"]["realization"] is True
        assert diagnostics["gates"]["residual"] is False

    def test_one_blown_sector_is_enough(self, mf, monkeypatch):
        """A healthy hole must not carry a broken particle through the gate."""
        state = {"seen": 0}

        def only_second(record):
            state["seen"] += 1
            return self._blow(record) if state["seen"] == 2 else record

        _patch_record(monkeypatch, only_second)

        diagnostics = _run(mf)

        residuals = diagnostics["residual"]
        assert min(residuals.values()) < RESIDUAL_MAX < max(residuals.values())
        assert diagnostics["gates"]["residual"] is False


class TestUnmeasured:
    """Not measuring the residual is not the same as measuring it and finding it sound."""

    @staticmethod
    def _unmeasured(record):
        record["errors"] = None
        return record

    def test_gate_fails_when_unmeasured(self, mf, monkeypatch):
        _patch_record(monkeypatch, self._unmeasured)

        diagnostics = _run(mf)

        assert diagnostics["gates"]["residual"] is False
        assert diagnostics["converged"] is False

    def test_unmeasured_is_reported_as_none_not_zero(self, mf, monkeypatch):
        """`None` and a healthy `0.0` must not be confused in the readout."""
        _patch_record(monkeypatch, self._unmeasured)

        diagnostics = _run(mf)

        assert all(value is None for value in diagnostics["residual"].values())
