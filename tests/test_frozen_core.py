"""Tests for the frozen-core path in `build_se_static`.

`self.active` masks orbitals. It was being applied to AO-basis matrices, which is a no-op
when nothing is frozen and `nao == nmo` and wrong otherwise - silently truncating them for
a Hartree-Fock reference, and raising for a DFT one where `vj` and `vk` then had different
shapes. These pin both halves: that freezing works at all, and that not freezing is
unchanged.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from pyscf import dft, gto

from momentGW import GW
from momentGW.gw import frontier_readout

HARTREE_TO_EV = 27.211386245988


@pytest.fixture(scope="module")
def mols():
    """Build one mean field per reference, on a molecule with a core to freeze."""
    mol = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="sto3g", verbose=0
    )
    fields = {}
    for xc in ("hf", "pbe"):
        mf = dft.RKS(mol, xc=xc).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        fields[xc] = mf
    return fields


def _run(mf, **kwargs):
    """Run a one-shot GW and hand back the solver and its frontier."""
    gw = GW(mf, polarizability="drpa", **kwargs)
    gw.verbose = 0
    with contextlib.redirect_stdout(io.StringIO()):
        _, gf, _, _ = gw.kernel(nmom_max=1)
    return gw, frontier_readout(gf)


class TestFrozenCoreRuns:
    """Freezing a core orbital must work for either reference."""

    @pytest.mark.parametrize("xc", ["hf", "pbe"])
    def test_runs_and_drops_an_orbital(self, mols, xc):
        """A DFT reference used to raise here on mismatched `vj` and `vk` shapes."""
        gw, frontier = _run(mols[xc], frozen=[0])

        assert gw.nmo == mols[xc].mo_occ.size - 1
        assert np.isfinite(frontier["homo"])
        assert np.isfinite(frontier["lumo"])

    @pytest.mark.parametrize("xc", ["hf", "pbe"])
    def test_static_self_energy_is_square_in_the_active_space(self, mols, xc):
        """The shape the rest of the solve depends on, and the one that was wrong."""
        gw = GW(mols[xc], polarizability="drpa", frozen=[0])
        gw.verbose = 0
        with contextlib.redirect_stdout(io.StringIO()):
            integrals = gw.ao2mo()
            se_static = gw.build_se_static(integrals)

        assert se_static.shape == (gw.nmo, gw.nmo)


class TestUnfrozenUnchanged:
    """The fix must be inert where nothing is frozen, which is every recorded case."""

    @pytest.mark.parametrize("xc", ["hf", "pbe"])
    def test_frontier_is_stable(self, mols, xc):
        """`frozen=None` and `frozen=[]` are the same calculation."""
        _, without = _run(mols[xc])
        _, empty = _run(mols[xc], frozen=[])

        assert abs(without["homo"] - empty["homo"]) * HARTREE_TO_EV < 1e-10
        assert abs(without["lumo"] - empty["lumo"]) * HARTREE_TO_EV < 1e-10


class TestFrozenCoreCosts:
    """Freezing is a physical approximation with a measurable price, not a free speedup."""

    def test_the_shift_is_large_enough_to_matter(self, mols):
        """Recorded so a change that quietly made it small would be noticed."""
        _, full = _run(mols["pbe"])
        _, frozen = _run(mols["pbe"], frozen=[0])

        shift = abs(frozen["homo"] - full["homo"]) * HARTREE_TO_EV * 1000
        assert shift > 1.0, "freezing the core should move the HOMO by more than a meV"
