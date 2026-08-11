"""Document how the frontier converges with the moment order, and where it stops.

The Milestone 3 acceptance gate asks for order-convergence tables on H2O, LiH and the
selected small-gap system, and for evidence that increasing the supported order reduces the
stated frontier error until the floating-point or realization limit. This produces both.

Three things make the table cheap enough to sweep. The moments built at the cap contain
every lower order, so one construction serves the whole sweep. The frontier carries the
reference orbital dominating each pole, so a level crossing shows as a changed label rather
than as a large shift. And there are two independent truncation indicators - differencing
`m` against `m - 2`, and the spread between the Gauss and Gauss-Radau closures - which
matters because the first rests on a single difference of a sequence that is not monotonic.

**There is exactly one way this code steps down**, and it is worth stating plainly because
an earlier version of this study got it wrong. `MBLSE.kernel` sets `max_cycle_achieved` in
one place only, catching `NotPositiveSemiDefiniteError` from the next block's square root
(`dyson/solvers/static/_mbl.py:471`). So every shortfall is a **PSD failure** - the moments
cannot support a causal measure at that order - and there is no rank-versus-arithmetic
dichotomy to classify. Importing one from a code that gates on a Gram factorisation instead
produced two false readings: the reconstructed-moment residual is measured at the *achieved*
order, which the failure never touched, so it reports ~1e-15 for every step-down and can
never indicate the cause.

What the PSD gate costs is measurable, and is measured here. Loosening `neg_atol` and
`neg_rtol` lets the recurrence accept the offending direction and continue; the study
reports the order that buys and the residual it costs, so the trade is visible rather than
asserted. Note that it is `neg_atol`/`neg_rtol` that govern this, not `atol`/`rtol` - the
latter set the support mask and provably cannot move the step-down.

A study, not part of the recorded baseline set: re-run when the claims it supports are in
question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the printed
`momentGW.__file__` before believing a comparison::

    python -m baseline.studies.order_convergence
    python -m baseline.studies.order_convergence --systems water --cap 9
"""

import argparse
import contextlib
import io

import numpy as np

import momentGW
from baseline import frontier as frontier_lib
from baseline import systems as systems_module
from baseline.run import build_mean_field
from momentGW.gw import GW
from momentGW.rpa import dRPA

#: Below this quasiparticle weight the pole carrying the reference orbital is a satellite
#: rather than a quasiparticle, and the frontier energy is tracking the wrong thing. The
#: threshold follows `molecular-mGW-testing`, which flags LiH and MgO on exactly this.
QUASIPARTICLE_WEIGHT_MIN = 0.7

#: Systems the acceptance gate names: two ordinary closed shells and the small-gap case.
DEFAULT_SYSTEMS = ("water", "lithium-hydride", "ozone")


def _binding(realization):
    """Return the sector limiting the self-energy, and what it managed."""
    sector = min(realization, key=lambda s: realization[s]["nmom_conserved_achieved"])
    record = realization[sector]
    errors = record["errors"]
    residual = float(errors.max_relative_frobenius) if errors is not None else float("nan")
    return sector, record, residual


def _row(gw, order, nelectron):
    """Read one row of the table off a solved calculation."""
    diagnostics = gw.dyson_diagnostics
    sector, record, residual = _binding(diagnostics["realization"])
    requested = record["nmom_conserved_requested"]
    conserved = record["nmom_conserved_achieved"]
    shortfall = requested - conserved

    row = {
        "order": order,
        "built": record["moments_supplied"],
        "requested": requested,
        "conserved": conserved,
        "shortfall": shortfall,
        "binding": sector,
        "residual": residual,
        # There is one step-down path and it is a PSD failure; see the module docstring.
        "limit": "psd" if shortfall else "-",
        "nelec_error": float(diagnostics["nelec_error"]),
        "nelec_tol": float(diagnostics["nelec_tol"]),
        "gates": dict(diagnostics["gates"]),
        "converged": bool(diagnostics["converged"]),
        "differencing_ev": None,
        "closure_ev": None,
        "homo_ev": None,
        "homo_z": None,
        "homo_mo": None,
    }

    # The crossing-safe readout: Aufbau counting over the correlated multiplets, which does
    # not depend on the reference ordering. `frontier_readout` in the solver is index-based
    # and is what the two indicator columns use, so a level crossing is the one case where
    # this column and those disagree - which is the point of reading it this way here.
    try:
        readout = frontier_lib.readouts(gf_energies(gw), gf_couplings(gw), nelectron)
    except frontier_lib.SpectrumError as error:
        row["frontier_error"] = str(error)
    else:
        homo = readout["frontier"]["homo"]
        row["homo_ev"] = readout["homo_ha"] * frontier_lib.EV
        row["homo_z"] = homo["weight_per_state"]
        row["homo_mo"] = homo["dominant_mo_index"]
        row["lumo_ev"] = readout["lumo_ha"] * frontier_lib.EV

    frontier = diagnostics["moment_order_convergence"]
    closure = diagnostics["closure_spread"]
    if frontier is not None and "homo_shift" in frontier:
        row["differencing_ev"] = abs(frontier["homo_shift"]) * frontier_lib.EV
        row["crossed"] = bool(frontier.get("homo_orbital_changed"))
    if closure is not None and "homo_spread" in closure:
        row["closure_ev"] = abs(closure["homo_spread"]) * frontier_lib.EV
        row["closure_crossed"] = bool(closure.get("homo_orbital_changed"))
    return row


def gf_energies(gw):
    """Pole energies of the correlated Green's function."""
    return np.asarray(gw.gf.energies)


def gf_couplings(gw):
    """Couplings of the correlated Green's function."""
    return np.asarray(gw.gf.couplings)


def _psd_probe(mf, moments, integrals, order, loosen=1e4):
    """Measure what the PSD gate is costing at a stepped-down order.

    Loosening `neg_atol`/`neg_rtol` lets the recurrence accept the direction it refused,
    so the order it then reaches, and the residual that order carries, say what the gate
    bought. These are the tolerances that govern the step-down; `atol`/`rtol` set the
    support mask and cannot move it.

    Returns
    -------
    conserved : int
        Order conserved with the gate loosened.
    residual : float
        Reconstructed-moment residual at that order.
    """
    gw = GW(mf, polarizability="drpa")
    gw.dyson_opts = dict(
        gw.dyson_opts,
        neg_atol=gw.dyson_opts["neg_atol"] * loosen,
        neg_rtol=gw.dyson_opts["neg_rtol"] * loosen,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        gw.kernel(nmom_max=order, moments=moments, integrals=integrals)
    _, record, residual = _binding(gw.dyson_diagnostics["realization"])
    return record["nmom_conserved_achieved"], residual


def run(name, cap, xc="pbe"):
    """Sweep one system and print its order-convergence table."""
    system = systems_module.SYSTEMS[name]
    with contextlib.redirect_stdout(io.StringIO()):
        mf, _ = build_mean_field(system, xc)
        reference = GW(mf, polarizability="drpa")
        integrals = reference.ao2mo()
        moments = dRPA(
            reference,
            cap,
            integrals,
            mo_energy=dict(g=reference.mo_energy, w=reference.mo_energy),
        ).kernel()

    print(f"\n### {name} / {system.basis} / {xc}   nmo={mf.mo_occ.size}   cap nmom_max={cap}")
    nelectron = mf.mol.nelectron
    print(
        f"{'K':>3s} {'built':>5s} {'req':>4s} {'cons':>5s} {'short':>6s} {'residual':>10s} "
        f"{'limit':>6s} {'HOMO/eV':>10s} {'Z':>5s} {'MO':>3s} {'dHOMO/meV':>10s} "
        f"{'diff/meV':>9s} {'closure/meV':>12s} {'dN':>9s} {'gates':>7s}"
    )
    rows, previous = [], None
    for order in range(1, cap + 1, 2):
        gw = GW(
            mf,
            polarizability="drpa",
            moment_order_convergence=True,
            closure_spread=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            gw.kernel(
                nmom_max=order,
                moments=tuple(m[: order + 1] for m in moments),
                integrals=integrals,
            )
        row = _row(gw, order, nelectron)
        rows.append(row)
        if row["homo_ev"] is None:
            print(
                f"{row['order']:3d} {row['built']:5d} {row['requested']:4d} "
                f"{row['conserved']:5d} {row['shortfall']:6d} {row['residual']:10.2e} "
                f"{row['limit']:>6s}   no frontier: {row.get('frontier_error', '')}"
            )
            continue
        step = "" if previous is None else f"{(row['homo_ev'] - previous) * 1000:10.2f}"
        previous = row["homo_ev"]
        diff = "-" if row["differencing_ev"] is None else f"{row['differencing_ev'] * 1000:9.2f}"
        clos = "-" if row["closure_ev"] is None else f"{row['closure_ev'] * 1000:12.2f}"
        failed = [k for k, v in row["gates"].items() if not v]
        gates = "ok" if not failed else ",".join(failed)
        print(
            f"{row['order']:3d} {row['built']:5d} {row['requested']:4d} {row['conserved']:5d} "
            f"{row['shortfall']:6d} {row['residual']:10.2e} {row['limit']:>6s} "
            f"{row['homo_ev']:10.5f} {row['homo_z']:5.3f} {row['homo_mo']:3d} {step:>10s} "
            f"{diff:>9s} {clos:>12s} {row['nelec_error']:9.2e} {gates:>7s}"
        )

    # Probe every distinct stepped-down order, not just the first: the shortfall grows
    # with the order and a verdict read off the lowest one does not cover the rest.
    seen = set()
    for row in (r for r in rows if r["shortfall"]):
        if row["conserved"] in seen:
            continue
        seen.add(row["conserved"])
        reached, residual = _psd_probe(
            mf, tuple(m[: row["order"] + 1] for m in moments), integrals, row["order"]
        )
        if reached > row["conserved"]:
            verdict = f"the PSD gate is binding; it costs {reached - row['conserved']} orders"
        elif reached < row["conserved"]:
            verdict = "loosening the gate made it worse"
        else:
            verdict = "not the PSD tolerance: unchanged when it is loosened"
        print(
            f"  K={row['order']:2d} conserved {row['conserved']} at residual "
            f"{row['residual']:.2e}; with neg_atol/neg_rtol x1e4 it conserves {reached} at "
            f"{residual:.2e} -> {verdict}"
        )
    if any(r.get("crossed") for r in rows):
        print("  NOTE: the dominant frontier orbital changed between consecutive orders")
    if any(r.get("closure_crossed") for r in rows):
        print(
            "  NOTE: the two closures disagreed on the frontier state at least once, so "
            "the spread there is between different states"
        )
    weights = [r["homo_z"] for r in rows if r.get("homo_z") is not None]
    if not weights:
        return rows
    worst_z = min(weights)
    if worst_z < QUASIPARTICLE_WEIGHT_MIN:
        print(
            f"  WARNING: quasiparticle weight falls to {worst_z:.3f}, below "
            f"{QUASIPARTICLE_WEIGHT_MIN}: the frontier energy is tracking a satellite, so "
            "its convergence says nothing about a quasiparticle"
        )
    return rows


def main():
    """Run the sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--cap", type=int, default=15)
    parser.add_argument("--xc", default="pbe")
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    for name in args.systems:
        run(name, args.cap, xc=args.xc)


if __name__ == "__main__":
    main()
