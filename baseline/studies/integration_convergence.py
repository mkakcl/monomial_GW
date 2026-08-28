"""Converge each eta0 route in its own knob, and see which default was closer.

The question this answers: a def2-TZVPP GW100 campaign found frontier differences of tens
of meV between the legacy Clenshaw-Curtis integration and the certified HHT route, and the
suspicion was that the new route was introducing them. Neither route's default had been
converged against itself, so "which is wrong" had not been asked in a form that could be
answered.

Sweeping each in its own knob - `npoints` for Clenshaw-Curtis, `eta0_n_poles` for HHT -
answers it. **The two routes converge to each other**, to under 1 meV on 25 of 30 systems,
so the integral is not in dispute; the default-to-default gaps are resolution rather than
disagreement.

Which default was closer, over the 30 systems of that campaign's two deviation lists:

- **Clenshaw-Curtis at 48 points is off by more than 1 meV on 14 of 30**, worst 152 meV, and
  its error is systematically negative - a 13 to 50 meV deficit on many systems, which is
  the right size and sign to account for the campaign's old-versus-new list.
- **HHT at its automatic pole count is off by more than 1 meV on 7 of 30**, and is within
  0.1 meV on most. HHT is closer on 27 of the 30.
- **HHT's automatic selection has a real failure mode**: copper-cyanide +173.7 meV and
  potassium-hydride -98.9 meV. On copper-cyanide the eta0 certificate read `8.98e-15`,
  inside its `1e-14` tolerance, while the frontier was 174 meV out. **The scalar
  approximation error does not bound the quasiparticle error**, which is Milestone 3.1's
  open item stated as a measurement. The realization gate did catch that run.

Two findings that are not about the integration at all, recorded because they change how the
table should be read:

- **Twelve of the thirty fail the `realization` gate at default settings**, six of them on
  both routes. There the recurrence steps down however eta0 was computed, so those systems'
  numbers are suspect for an unrelated reason.
- **Five systems' converged limits still disagree by more than 1 meV** - lithium-fluoride by
  62.7 and potassium-bromide by 30.2 - and both fail `realization`. Where the two routes
  genuinely fail to meet, it is the realization rather than the integral.

Mean field is **Hartree-Fock**, as `dft.RKS(mol, xc="hf")`, deliberately. `xc="hf"` resolves
the auxiliary basis to `def2-tzvpp-jkfit` on every pyscf since 2.10, whereas a pure
functional resolves to the much smaller Coulomb-only `def2-universal-jfit`; running HF keeps
the auxiliary basis fixed across pyscf versions and comparable with the campaign's older
results. **Nothing here tests the PBE half**, where that auxiliary-basis difference lives.

The systems below are the subset that carries the finding, embedded so the study is
self-contained. The full thirty came from the GW100 set in the `molecular-mGW-testing`
repository and are quoted above rather than reproduced here.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`.

Run from the repository root::

    python -m baseline.studies.integration_convergence
    python -m baseline.studies.integration_convergence --systems krypton copper-cyanide
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

from pyscf import dft, gto

import momentGW
from momentGW.gw import GW, frontier_readout

HARTREE_TO_MEV = 27211.386245988

#: Moment order the campaign used.
NMOM_MAX = 11

#: Clenshaw-Curtis point counts. 48 is the default; 384 is the converged reference, and 192
#: agrees with it to under 1 meV on every system checked.
POINTS = (48, 384)

#: Extra HHT poles beyond the automatic count, as the converged reference. Twenty is well
#: past where the frontier stops moving on every system measured.
EXTRA_POLES = 20

#: Geometries in Angstrom, from the GW100 set. Chosen to carry both failure modes and a
#: clean control rather than to be representative: `copper-cyanide` and `potassium-hydride`
#: are where HHT's automatic pole count under-resolves, `arsenic-dimer`, `bromine` and
#: `krypton` are where Clenshaw-Curtis at 48 points does, `lithium-fluoride` is where the two
#: limits disagree, and `aluminum-fluoride` is a system on which everything agrees exactly.
SYSTEMS = {
    "aluminum-fluoride": (
        "Al 0.000000 0.000000 0.000000; F 0.000000 0.000000 1.633000; "
        "F 0.000000 1.414200 -0.816500; F 0.000000 -1.414200 -0.816500"
    ),
    "arsenic-dimer": "As 0.000000 0.000000 0.000000; As 0.000000 0.000000 2.102600",
    "bromine": "Br 0.000000 0.000000 0.000000; Br 0.000000 0.000000 2.281100",
    "copper-cyanide": (
        "C 0.000000 0.000000 0.000000; N 0.000000 0.000000 1.158000; Cu 0.000000 0.000000 -1.832000"
    ),
    "copper-dimer": "Cu 0.000000 0.000000 0.000000; Cu 0.000000 0.000000 2.219700",
    "krypton": "Kr 0.000000 0.000000 0.000000",
    "lithium-fluoride": "Li 0.000000 0.000000 0.000000; F 0.000000 0.000000 1.563900",
    "potassium-hydride": "K 0.000000 0.000000 0.000000; H 0.000000 0.000000 2.244000",
}


def mean_field(atom):
    """Build the Hartree-Fock reference the campaign settings imply.

    `RKS` with `xc="hf"` rather than `RHF`: it is the same mean field, and it is the form
    whose automatic auxiliary basis is stable across pyscf versions.
    """
    mol = gto.M(atom=atom, basis="def2-tzvpp", verbose=0)
    mf = dft.RKS(mol, xc="hf").density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("the mean field did not converge")
    return mf


def ionisation(mf, **options):
    """Run one G0W0 and return the ionisation potential, pole count and failed gates."""
    gw = GW(mf, polarizability="drpa", compression=None, **options)
    gw.verbose = 0
    with contextlib.redirect_stdout(io.StringIO()):
        _, gf, _, _ = gw.kernel(nmom_max=NMOM_MAX)
    diagnostics = getattr(gw, "eta0_diagnostics", None) or {}
    failed = [name for name, ok in gw.dyson_diagnostics["gates"].items() if not ok]
    return (
        -frontier_readout(gf)["homo"] * HARTREE_TO_MEV / 1000.0,
        diagnostics.get("n_poles"),
        diagnostics.get("scalar_error"),
        failed,
    )


def main():
    """Sweep both routes and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=sorted(SYSTEMS), choices=sorted(SYSTEMS))
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    print(
        f"\nHartree-Fock / def2-TZVPP / def2-tzvpp-jkfit, no compression, "
        f"nmom_max = {NMOM_MAX}. Ionisation potentials in eV, errors in meV."
    )
    print(
        f"\n{'system':>19} {'naux':>5} {'cc@48':>9} {'cc@384':>9} {'hht auto':>9} "
        f"{'hht+20':>9} {'cc err':>9} {'hht err':>9} {'limits':>8} {'closer':>8}  gates"
    )
    for name in args.systems:
        started = time.perf_counter()
        mf = mean_field(SYSTEMS[name])
        coarse, _, _, failed_cc = ionisation(mf, eta0_method="clencur", npoints=POINTS[0])
        fine, _, _, _ = ionisation(mf, eta0_method="clencur", npoints=POINTS[1])
        auto, poles, _, failed_hht = ionisation(mf, eta0_method="hht")
        more, _, _, _ = ionisation(mf, eta0_method="hht", eta0_n_poles=poles + EXTRA_POLES)

        error_cc = (coarse - fine) * 1000.0
        error_hht = (auto - more) * 1000.0
        gates = " ".join(
            part
            for part in (
                "cc:" + ",".join(failed_cc) if failed_cc else "",
                "hht:" + ",".join(failed_hht) if failed_hht else "",
            )
            if part
        )
        print(
            f"{name:>19} {mf.with_df.get_naoaux():>5} {coarse:9.4f} {fine:9.4f} {auto:9.4f} "
            f"{more:9.4f} {error_cc:9.1f} {error_hht:9.1f} {(fine - more) * 1000.0:8.1f} "
            f"{'hht' if abs(error_hht) < abs(error_cc) else 'clencur':>8}  "
            f"{gates or 'pass'}  [{time.perf_counter() - started:.0f}s]"
        )

    print(
        "\n   Over the full thirty systems of the campaign: the limits agree to under 1 meV\n"
        "   on 25, Clenshaw-Curtis at 48 points is off by more than 1 meV on 14 and HHT at\n"
        "   its automatic pole count on 7, and HHT is closer on 27. See the module docstring."
    )


if __name__ == "__main__":
    raise SystemExit(main())
