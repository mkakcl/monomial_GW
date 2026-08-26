"""What auxiliary compression and a frozen core cost the frontier, in meV.

The scientific validation track asks for these two quantified in meV for the frontier
quasiparticle energies. They are not the same kind of approximation and the numbers say so:

**Compression is free at the tolerance this code uses.** At `compression_tol = 1e-10`, the
default, and at 1e-8, the frontier does not move at all - 0.000 meV on every system and
reference measured, while the auxiliary space shrinks. It only becomes visible at 1e-6
(under 0.5 meV) and 1e-4 (a few meV). This is a numerical knob behaving like one.

**A frozen core is not.** Freezing the core costs between 0.8 and 374 meV depending on the
system and the reference, which is four to five orders of magnitude above the baseline's own
reproducibility floor. It is a physical approximation with a physical price, and it belongs
with the starting-point uncertainty of the next item rather than with the numerical
tolerances of Milestone 3's error budget.

The reference for both is the same calculation with neither applied.

This study also found the bug that made half of it impossible: `build_se_static` applied
`self.active`, an orbital mask, to AO-basis matrices. With nothing frozen and `nao == nmo`
that is a no-op, which is why it survived; with a frozen core it silently truncated the AO
matrices for a Hartree-Fock reference and raised outright for a DFT one, where `vj` and `vk`
were then different shapes. Fixed alongside, and pinned in `tests/test_frozen_core.py`.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`.

Run from the repository root so the intended tree is imported::

    python -m baseline.studies.approximation_errors
    python -m baseline.studies.approximation_errors --systems water --orders 3
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np

import momentGW
from baseline import systems as systems_module
from baseline.studies.order_convergence import build_mean_field
from momentGW.gw import GW, frontier_readout

HARTREE_TO_MEV = 27211.386245988

#: Compression tolerances swept. The first is the default; the rest are loosened by two
#: orders at a time until the frontier moves, which is the point of the sweep.
TOLERANCES = (1e-10, 1e-8, 1e-6, 1e-4)

#: Systems with a core worth freezing. Hydrogen has none, so it is not in the table.
SYSTEMS = ("water", "lithium-hydride", "ozone")


def core_orbitals(mf):
    """Count the core orbitals: one per atom heavier than helium.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Converged mean field.

    Returns
    -------
    count : int
        Number of core orbitals.
    """
    return int(np.sum(mf.mol.atom_charges() > 2))


def frontier(mf, nmom_max, **kwargs):
    """Run a one-shot GW and read its frontier.

    Returns
    -------
    tuple
        The frontier readout and the retained auxiliary rank.
    """
    gw = GW(mf, polarizability="drpa", **kwargs)
    gw.verbose = 0
    with contextlib.redirect_stdout(io.StringIO()):
        integrals = gw.ao2mo()
        _, gf, _, _ = gw.kernel(nmom_max=nmom_max, integrals=integrals)
    return frontier_readout(gf), int(integrals.naux)


def main():
    """Sweep both approximations and print their tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--orders", nargs="+", type=int, default=[3])
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    for nmom_max in args.orders:
        print(f"\n=== nmom_max = {nmom_max}, shifts against no compression and no frozen core ===")
        print(
            f"{'system':>17} {'ref':>4} {'variant':>18} {'naux':>6} "
            f"{'dHOMO/meV':>11} {'dLUMO/meV':>11}"
        )
        for name in args.systems:
            system = systems_module.SYSTEMS[name]
            for xc in systems_module.STARTING_POINTS:
                with contextlib.redirect_stdout(io.StringIO()):
                    mf, _ = build_mean_field(system, xc)
                reference, naux_full = frontier(mf, nmom_max, compression="")
                print(
                    f"{name:>17} {xc:>4} {'reference':>18} {naux_full:>6} {0.0:11.3f} {0.0:11.3f}"
                )
                for tol in TOLERANCES:
                    got, naux = frontier(mf, nmom_max, compression="ia", compression_tol=tol)
                    print(
                        f"{'':>17} {'':>4} {'ia tol=' + format(tol, '.0e'):>18} {naux:>6} "
                        f"{(got['homo'] - reference['homo']) * HARTREE_TO_MEV:11.3f} "
                        f"{(got['lumo'] - reference['lumo']) * HARTREE_TO_MEV:11.3f}"
                    )
                ncore = core_orbitals(mf)
                got, naux = frontier(mf, nmom_max, compression="", frozen=list(range(ncore)))
                print(
                    f"{'':>17} {'':>4} {'frozen ' + str(ncore) + ' core':>18} {naux:>6} "
                    f"{(got['homo'] - reference['homo']) * HARTREE_TO_MEV:11.3f} "
                    f"{(got['lumo'] - reference['lumo']) * HARTREE_TO_MEV:11.3f}"
                )

    print(
        "\nCompression at the default tolerance is free to the resolution printed, while the\n"
        "frozen core is not: it is a physical approximation and belongs with the starting-point\n"
        "uncertainty rather than with the numerical tolerances of the error budget."
    )


if __name__ == "__main__":
    raise SystemExit(main())
