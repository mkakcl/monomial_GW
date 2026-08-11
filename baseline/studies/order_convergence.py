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

A shortfall in the conserved order has two causes that mean opposite things, and the
`limit` column separates them by the reconstructed-moment residual:

``rank``
    The sector's support is exhausted. The rule reproduces its own moments to ~1e-15 and
    there is nothing left for higher orders to conserve. Not a failure.
``arith``
    float64 has run out of digits. The residual degrades by decades.

Where a shortfall is called ``rank`` this also re-runs it with the scale-aware support
policy tightened, because the two are distinguishable: a genuine rank limit does not move
when the policy changes, while a policy-imposed one does. The distinction was open after
the benzene sweep of 2026-08-10.

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
from baseline import systems as systems_module
from baseline.run import build_mean_field
from momentGW.gw import GW
from momentGW.rpa import dRPA

HARTREE2EV = 27.211386245988

#: Residual below which a shortfall is the sector's support running out rather than
#: float64 running out of digits. The two sit decades apart, so the exact value is not
#: delicate; see the module docstring.
RANK_RESIDUAL_MAX = 1e-13

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


def _row(gw, order):
    """Read one row of the table off a solved calculation."""
    diagnostics = gw.dyson_diagnostics
    sector, record, residual = _binding(diagnostics["realization"])
    requested = record["nmom_conserved_requested"]
    conserved = record["nmom_conserved_achieved"]
    shortfall = requested - conserved

    limit = "-"
    if shortfall:
        limit = "rank" if residual < RANK_RESIDUAL_MAX else "arith"

    frontier = diagnostics["moment_order_convergence"]
    closure = diagnostics["closure_spread"]
    gf = gw.gf.physical(weight=0.1)
    row = {
        "order": order,
        "built": record["moments_supplied"],
        "requested": requested,
        "conserved": conserved,
        "shortfall": shortfall,
        "binding": sector,
        "residual": residual,
        "limit": limit,
        "nelec_error": float(diagnostics["nelec_error"]),
        "differencing_ev": None,
        "closure_ev": None,
    }
    for name, sub, index in (("homo", gf.occupied(), -1), ("lumo", gf.virtual(), 0)):
        if not sub.naux:
            continue
        couplings = sub.couplings[..., index]
        row[f"{name}_ev"] = float(np.real(sub.energies[index])) * HARTREE2EV
        row[f"{name}_z"] = float(np.dot(couplings, couplings).real)
        row[f"{name}_mo"] = int(np.argmax(np.abs(couplings)))
    if frontier is not None and "homo_shift" in frontier:
        row["differencing_ev"] = abs(frontier["homo_shift"]) * HARTREE2EV
        row["crossed"] = bool(frontier.get("homo_orbital_changed"))
    if closure is not None and "homo_spread" in closure:
        row["closure_ev"] = abs(closure["homo_spread"]) * HARTREE2EV
    return row


def _policy_probe(mf, moments, integrals, order, tighten=1e-4):
    """Re-run a rank-limited order with the support policy tightened.

    A genuine rank limit is the sector's support running out and does not move when the
    scale-aware `matrix_power` policy changes. A policy-imposed one does.
    """
    gw = GW(mf, polarizability="drpa")
    gw.dyson_opts = dict(
        gw.dyson_opts,
        atol=gw.dyson_opts["atol"] * tighten,
        rtol=gw.dyson_opts["rtol"] * tighten,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        gw.kernel(nmom_max=order, moments=moments, integrals=integrals)
    _, record, _ = _binding(gw.dyson_diagnostics["realization"])
    return record["nmom_conserved_achieved"]


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
    print(
        f"{'K':>3s} {'built':>5s} {'req':>4s} {'cons':>5s} {'short':>6s} {'residual':>10s} "
        f"{'limit':>6s} {'HOMO/eV':>10s} {'Z':>5s} {'MO':>3s} {'dHOMO/meV':>10s} "
        f"{'diff/meV':>9s} {'closure/meV':>12s} {'dN':>9s}"
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
        row = _row(gw, order)
        step = "" if previous is None else f"{(row['homo_ev'] - previous) * 1000:10.2f}"
        previous = row["homo_ev"]
        rows.append(row)
        diff = "-" if row["differencing_ev"] is None else f"{row['differencing_ev'] * 1000:9.2f}"
        clos = "-" if row["closure_ev"] is None else f"{row['closure_ev'] * 1000:12.2f}"
        print(
            f"{row['order']:3d} {row['built']:5d} {row['requested']:4d} {row['conserved']:5d} "
            f"{row['shortfall']:6d} {row['residual']:10.2e} {row['limit']:>6s} "
            f"{row['homo_ev']:10.5f} {row['homo_z']:5.3f} {row['homo_mo']:3d} {step:>10s} "
            f"{diff:>9s} {clos:>12s} {row['nelec_error']:9.2e}"
        )

    limited = [r for r in rows if r["limit"] == "rank"]
    if limited:
        first = limited[0]
        tightened = _policy_probe(
            mf, tuple(m[: first["order"] + 1] for m in moments), integrals, first["order"]
        )
        verdict = (
            "policy, not support: tightening the support policy recovers order"
            if tightened > first["conserved"]
            else "genuine: unchanged when the support policy is tightened"
        )
        print(
            f"  rank limit at K={first['order']} conserved {first['conserved']}; "
            f"with atol/rtol x1e-4 it conserves {tightened} -> {verdict}"
        )
    if any(r.get("crossed") for r in rows):
        print("  NOTE: the dominant frontier orbital changed at least once in this sweep")
    worst_z = min(r["homo_z"] for r in rows if "homo_z" in r)
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
