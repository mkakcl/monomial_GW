"""What the moment truncation costs, against an exact G0W0 spectrum.

Two roadmap items ask for this and they are the same measurement: Milestone 3.4's
"spectral-function comparison against an untruncated small-system oracle where satellites
or deep states are reported", and the validation track's "report spectral functions rather
than relying only on per-orbital pole assignment in dense satellite regions".

The oracle is the exact full-pole `G0W0` implementation in `mkakcl/cayley-gw`
(`ExactG0W0SelfEnergy`): every self-energy pole is constructed explicitly from the RPA
eigensystem, the upfolded Hamiltonian is formed and diagonalised, and nothing is truncated
at any moment order. It is a different code with different conventions, so this is an
external check rather than an internal one.

**The two codes are made to share their integrals**, which is the point of the exercise.
Left alone, `cayley-gw` uses exact four-index MO integrals and momentGW uses density
fitting, and the comparison then measures the RI difference rather than the truncation: the
spectral-function error stops falling at 5e-3 and the HOMO at 0.83 meV no matter how many
moments are added. Feeding cayley the same density-fitted integrals momentGW is using -
contracting `L_Ppq L_Prs` into the four-index tensor it wants - removes that floor and the
error falls to 3e-6. Both modes are available here, because the floor is itself worth
knowing: it is what density fitting costs on this system.

What the comparison shows is that **the frontier and the spectral function converge at very
different rates**, which is exactly the distinction 3.4 asks to be made rather than assumed.

`cayley-gw` is not a dependency of this repository. The study skips with an explanation
when it is not importable rather than failing.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`.

Run from the repository root, with `cayley-gw` on the path::

    PYTHONPATH=/path/to/cayley-gw/src python -m baseline.studies.spectral_oracle
    PYTHONPATH=... python -m baseline.studies.spectral_oracle --unmatched-integrals
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
from pyscf import gto, scf

import momentGW
from momentGW.gw import GW

HARTREE_TO_MEV = 27211.386245988

#: Lorentzian broadening for the spectral function, in Hartree. Wide enough that the
#: comparison is not a test of whether two delta functions land on the same grid point.
BROADENING = 0.01

#: Moment orders compared. Swept past the frontier's convergence so the gap between it and
#: the spectral function is visible rather than asserted.
ORDERS = (1, 3, 5, 7, 9, 11, 15)


def require_cayley():
    """Import the oracle, or explain why the study cannot run.

    Returns
    -------
    module or None
        The pieces needed from `cayleygw`, or `None` if it is not importable.
    """
    # Imported here rather than at module scope on purpose: `cayley-gw` is an optional
    # oracle and not a dependency of this repository, so importing it at the top would make
    # the module unimportable wherever it is absent - including for anything that only
    # wanted to read the helpers below.
    try:
        from cayleygw import ExactG0W0SelfEnergy, RestrictedPySCFAdapter  # noqa: PLC0415
        from cayleygw.pyscf_interface import FullMOIntegrals  # noqa: PLC0415
        from cayleygw.rpa import DirectRPAProblem  # noqa: PLC0415
    except ImportError:
        return None
    return ExactG0W0SelfEnergy, RestrictedPySCFAdapter, FullMOIntegrals, DirectRPAProblem


def quasiparticle_energies(energies, couplings, orbitals):
    """Energy of the pole carrying the most weight of each physical orbital.

    The same rule on both sides of the comparison, so that a difference is a difference in
    the calculation rather than in how the frontier was picked out of it. This is the
    largest-overlap assignment Milestone 3.4 warns about relying on alone, used here as a
    correspondence between two spectra rather than as a claim about either.
    """
    weights = np.abs(np.asarray(couplings)) ** 2
    return np.array([float(energies[int(np.argmax(weights[p]))]) for p in orbitals])


def spectral_function(energies, couplings, grid, broadening):
    """Lorentzian-broadened spectral function, traced over physical orbitals."""
    energies = np.asarray(energies)
    weights = np.einsum("pk,pk->k", np.asarray(couplings).conj(), np.asarray(couplings)).real
    denominator = (grid[:, None] - energies) ** 2 + broadening**2
    return np.sum(weights * broadening / (np.pi * denominator), axis=-1)


def build_exact(pieces, mf, matched):
    """Build the exact spectrum, optionally on momentGW's own density-fitted integrals."""
    exact_cls, adapter_cls, full_integrals_cls, rpa_problem_cls = pieces
    adapter = adapter_cls(mf)
    if matched:
        three_center = np.asarray(adapter.build_density_fitted_integrals().three_center)
        integrals = full_integrals_cls(
            np.einsum("Ppq,Prs->pqrs", three_center, three_center, optimize=True)
        )
    else:
        integrals = adapter.build_full_integrals()
    rpa_result = rpa_problem_cls.from_reference(
        adapter.reference, integrals, tolerances=adapter.tolerances
    ).solve()
    exact = exact_cls.from_rpa(
        adapter.reference,
        integrals,
        rpa_result,
        static_correction=adapter.build_static_self_energy_correction(),
    )
    return exact.upfolded_hamiltonian().diagonalize()


def main():
    """Run the comparison and print its table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", nargs="+", type=int, default=list(ORDERS))
    parser.add_argument(
        "--unmatched-integrals",
        action="store_true",
        help="let the oracle use exact four-index integrals, exposing the RI floor",
    )
    args = parser.parse_args()

    pieces = require_cayley()
    if pieces is None:
        print(
            "cayley-gw is not importable, so the oracle is unavailable and this study "
            "cannot run.\nIt is not a dependency of this repository; put its `src` on "
            "PYTHONPATH:\n\n    PYTHONPATH=/path/to/cayley-gw/src python -m "
            "baseline.studies.spectral_oracle\n"
        )
        return 0

    print(f"momentGW: {momentGW.__file__}")
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.595", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("the LiH/STO-3G reference did not converge")

    matched = not args.unmatched_integrals
    spectrum = build_exact(pieces, mf, matched)
    nocc = mol.nelectron // 2
    orbitals = list(range(min(nocc + 2, spectrum.nphysical)))
    exact_qp = quasiparticle_energies(spectrum.energies, spectrum.physical_amplitudes, orbitals)

    grid = np.linspace(
        float(spectrum.energies.min()) - 0.2, float(spectrum.energies.max()) + 0.2, 4000
    )
    reference = spectrum.spectral_function(grid, broadening=BROADENING)

    print(
        f"\nLiH/STO-3G, RHF. Oracle: {spectrum.energies.size} exact charged poles, "
        f"integrals {'shared with momentGW' if matched else 'exact four-index (RI floor)'}."
    )
    print(
        f"\n{'nmom_max':>9} {'poles':>6} {'int|dA|/intA':>14} {'max|dA|/maxA':>14} "
        f"{'dHOMO/meV':>11} {'worst occ':>11} {'worst shown':>12}"
    )
    for nmom_max in args.orders:
        gw = GW(mf, polarizability="drpa")
        gw.verbose = 0
        with contextlib.redirect_stdout(io.StringIO()):
            _, gf, _, _ = gw.kernel(nmom_max=nmom_max)
        energies = np.asarray(gf.energies)
        got = spectral_function(energies, gf.couplings, grid, BROADENING)
        area = float(np.trapezoid(np.abs(got - reference), grid) / np.trapezoid(reference, grid))
        peak = float(np.max(np.abs(got - reference)) / np.max(reference))
        shifts = (
            quasiparticle_energies(energies, gf.couplings, orbitals) - exact_qp
        ) * HARTREE_TO_MEV
        print(
            f"{nmom_max:>9} {energies.size:>6} {area:14.3e} {peak:14.3e} "
            f"{shifts[nocc - 1]:11.3f} {np.max(np.abs(shifts[:nocc])):11.3f} "
            f"{np.max(np.abs(shifts)):12.3f}"
        )

    print(
        "\n   The frontier converges long before the spectral function does: reporting a\n"
        "   settled HOMO says nothing about the satellites, which is what 3.4 asks to be\n"
        "   measured rather than assumed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
