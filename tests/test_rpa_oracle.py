"""The dRPA construction against an external code and an explicit-pole oracle.

The scientific validation track asks for these comparisons to be *maintained*, so the fast
case is pinned here rather than only in `baseline/studies/rpa_oracle.py`. The study sweeps
systems and orders; this catches a regression in the suite.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from pyscf import dft, gto, tdscf

from baseline.studies.rpa_oracle import oracle_moment, rpa_spectrum
from momentGW.gw import GW
from momentGW.rpa import dRPA

HARTREE_TO_EV = 27.211386245988


@pytest.fixture(scope="module")
def case():
    """Build a tiny dRPA case: mean field, moments, energy differences and integrals."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = dft.RKS(mol, xc="hf").density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()

    gw = GW(mf, polarizability="drpa")
    with contextlib.redirect_stdout(io.StringIO()):
        integrals = gw.ao2mo()
        rpa = dRPA(gw, 7, integrals)
        rpa._build_d()
        moments = np.asarray(rpa.build_dd_moments())
        zeroth = np.asarray(rpa.build_zeroth_dd_moment())
    return mf, moments, zeroth, rpa.d, integrals.Lia


class TestAgainstPySCF:
    """The physics: is the matrix this code builds the direct RPA problem?"""

    def test_excitation_energies_match(self, case):
        """An independent implementation, with its own conventions, must agree."""
        mf, _, _, d, integrals = case
        omega, _ = rpa_spectrum(d, integrals)

        td = tdscf.dRPA(mf)
        td.nstates = omega.size
        with contextlib.redirect_stdout(io.StringIO()):
            td.kernel()
        theirs = np.sort(np.asarray(td.e).real)

        count = min(omega.size, theirs.size)
        assert count > 0
        assert np.max(np.abs(omega[:count] - theirs[:count])) * HARTREE_TO_EV < 1e-8

    def test_the_spectrum_is_not_trivially_small(self, case):
        """Guards the test above: agreeing on an empty spectrum would prove nothing."""
        _, _, _, d, integrals = case
        omega, _ = rpa_spectrum(d, integrals)

        assert omega.size >= 3
        assert np.all(omega > 0.0)


class TestAgainstExplicitPoles:
    """The recursion: does it reproduce a closed form in the explicit excitations?"""

    @pytest.mark.parametrize("order", [0, 1, 2, 3, 5, 7])
    def test_moment_matches_the_oracle(self, case, order):
        """Every order, against a sum over poles that shares no arithmetic with it."""
        _, moments, _, d, integrals = case
        omega, vectors = rpa_spectrum(d, integrals)

        reference = oracle_moment(order, d, integrals, omega, vectors)
        got = np.asarray(moments[order])

        assert np.linalg.norm(got - reference) / np.linalg.norm(reference) < 1e-10

    def test_first_moment_collapses_to_v_d(self, case):
        """`S` is orthogonal, so the oracle at n=1 must be `V D` exactly. Checks the oracle."""
        _, _, _, d, integrals = case
        omega, vectors = rpa_spectrum(d, integrals)

        reference = oracle_moment(1, d, integrals, omega, vectors)

        assert np.allclose(reference, integrals * d[None], atol=1e-12, rtol=0.0)

    def test_zeroth_moment_matches_the_certified_route(self, case):
        """The oracle at n=0 against the eta0 route Milestone 2 certified separately."""
        _, _, zeroth, d, integrals = case
        omega, vectors = rpa_spectrum(d, integrals)

        reference = oracle_moment(0, d, integrals, omega, vectors)

        assert np.linalg.norm(zeroth - reference) / np.linalg.norm(reference) < 1e-10
