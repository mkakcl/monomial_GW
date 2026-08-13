"""Find every MBLSE stall in the baseline set, and price it on the frontier.

This is the remaining question in [`HANKEL_PENCIL.md`](../../HANKEL_PENCIL.md)
§6.B.  The head-to-head there left the pencil with one niche -- a fallback for
sectors where the recursion has stalled -- and one condition on taking it:
*"find a case where a stall costs more than a µeV, or close the pencil out"*.

Two things count as the recursion failing to deliver what it was asked for, and
both are swept.  The first is a **stall**: `MBLSE` conserving fewer moments than
it was given.  ROADMAP 3.3 names the mechanism -- `MBLSE.kernel` steps down in
exactly one place, a PSD failure on the next block's square root, and loosening
`neg_atol`/`neg_rtol` by 1e4 buys lithium-hydride two more orders at 60x the
residual -- so a stall is a gate decision rather than a breakdown.  The second
is **ungated**: conserving everything requested while not reproducing it, which
3.3 records for lithium-hydride's particle sector at `nmom_max = 19`, 20 of 20
conserved with a residual eighteen orders above the healthy band.  Counting only
the first would miss the second entirely, and the second is the one no gate
catches.

The cost is measured the way §6.B measured it: realize the same moments by the
deflated block Hankel pencil, which has no PSD gate, put both self-energies
through the same `FockLoop` and Dyson solve, and difference the frontier.  The
pencil is the *comparison*, not a proposal -- it stands in for "what the
recursion would have said had it not failed".

**That stand-in only works where the pencil is itself trustworthy**, and section
4 predicts it degrading with order.  It does: above `nmom_max = 11` this sweep
sees pencil reconstruction errors reaching 1e+11.  So a row is priced only when
the pencil clears the same `RESIDUAL_MAX` the order-convergence study uses, six
orders above the healthy band and twelve below the failures.  Scoring against a
pencil in that state measures the pencil rather than the recursion, and counting
those is what made an earlier version of this study report 9.1 meV.

Three limits on the sweep.  It covers the Milestone 0 systems in cc-pVDZ and
6-31G, which is where the baseline lives, not the larger cases in
`benchmark/`; a failure that only appears at benzene/cc-pVTZ would be missed.
Compression is off, so one induced by auxiliary compression is out of scope --
`--compression` puts it back.  And the orders run past 19, where 3.3 finds all
three acceptance-gate systems pin their realization: an earlier version stopped
at 15, the same cap that hid a realization failure from the order-convergence
tables until it was lifted.

A study, not part of the recorded baseline set: it is re-run when the claim it
supports is in question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the
printed `momentGW.__file__` before believing a comparison::

    python -m baseline.studies.stall_sweep
    python -m baseline.studies.stall_sweep --systems water ozone --orders 15 19 21
"""

import argparse
import contextlib
import io
import time

import numpy as np
from dyson import MBLSE

import momentGW
from baseline import run as baseline_run
from baseline import systems as baseline_systems
from baseline.studies.order_convergence import RESIDUAL_MAX
from baseline.studies.pencil_vs_mblse import (
    frontier_from_self_energy,
    pencil_self_energy,
    reconstruction_error,
)
from momentGW.gw import GW, achieved_iteration
from momentGW.rpa import dRPA

HARTREE2EV = 27.211386245988

#: Below this, a stall is not worth a fallback: ROADMAP 3.3 measures truncation moving the
#: frontier by tens of meV, four to five orders of magnitude above it.
MICROELECTRONVOLT = 1e-6

#: Swept past `nmom_max = 19`, where ROADMAP 3.3 finds all three acceptance-gate systems pin
#: their realization.  An earlier version of this study stopped at 15, which is the same cap
#: that hid a realization failure from the order-convergence tables until #31 lifted it.
DEFAULT_ORDERS = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
DEFAULT_SYSTEMS = ("hydrogen", "hydrogen-631g", "lithium-hydride", "water", "ozone")
DEFAULT_STARTING_POINTS = ("hf", "pbe")


def build_case(mf, nmom_max, compression):
    """Build the self-energy moments and static part for one case.

    Parameters
    ----------
    mf : pyscf.dft.RKS
        Converged mean field.
    nmom_max : int
        Maximum moment order.
    compression : str
        Auxiliary compression sectors, or `""` for none.

    Returns
    -------
    tuple
        The solver, the static self-energy, and the hole and particle moments.
    """
    gw = GW(mf)
    gw.compression = compression
    with contextlib.redirect_stdout(io.StringIO()):
        integrals = gw.ao2mo()
        se_static = gw.build_se_static(integrals)
        rpa = dRPA(gw, nmom_max, integrals)
        hole, particle = rpa.build_se_moments(rpa.build_dd_moments())
    return gw, se_static, np.asarray(hole), np.asarray(particle)


def realize(gw, se_static, sectors, route):
    """Realize both sectors by one route and combine them.

    Parameters
    ----------
    gw : momentGW.gw.GW
        The solver.
    se_static : numpy.ndarray
        Static part of the self-energy.
    sectors : sequence of numpy.ndarray
        The hole and particle moments.
    route : str
        Either `"mblse"` or `"pencil"`.

    Returns
    -------
    tuple
        The combined self-energy, the per-sector conserved counts (empty for the
        pencil, which has no such notion), and the per-sector reconstruction errors.
    """
    options = dict(gw.dyson_opts, calculate_errors=False)
    parts, conserved, errors = [], [], []
    for moments in sectors:
        if route == "mblse":
            with contextlib.redirect_stdout(io.StringIO()):
                solver = MBLSE(se_static, np.array(moments), **options)
                solver.kernel()
                part = solver.solve().get_self_energy()
            conserved.append(
                (
                    solver.nmom_conserved(achieved_iteration(solver)),
                    solver.nmom_conserved(solver.max_cycle),
                )
            )
        else:
            part, _ = pencil_self_energy(moments)
        parts.append(part)
        errors.append(float(np.max(reconstruction_error(part, moments))))

    combined = parts[0].copy()
    for part in parts[1:]:
        combined = combined.concatenate(part)
    return combined, conserved, errors


def frontier_difference(first, second):
    """Largest frontier difference between two readouts, in eV.

    Parameters
    ----------
    first, second : dict or None
        Frontier readouts.

    Returns
    -------
    float or None
        The largest absolute difference over the keys both carry, or `None` if
        either readout is missing.
    """
    if not first or not second:
        return None
    keys = [k for k in ("homo", "lumo") if k in first and k in second]
    if not keys:
        return None
    return max(abs(first[k] - second[k]) * HARTREE2EV for k in keys)


def run_case(name, xc, nmom_max, compression):
    """Sweep one case, reporting a stall and its frontier cost if there is one.

    Parameters
    ----------
    name : str
        System name.
    xc : str
        Exchange-correlation functional, or `"hf"`.
    nmom_max : int
        Maximum moment order.
    compression : str
        Auxiliary compression sectors, or `""` for none.

    Returns
    -------
    dict or None
        The case record, or `None` if the case could not be built.
    """
    system = baseline_systems.SYSTEMS[name]
    with contextlib.redirect_stdout(io.StringIO()):
        mf, _ = baseline_run.build_mean_field(system, xc)
    gw, se_static, hole, particle = build_case(mf, nmom_max, compression)

    started = time.perf_counter()
    combined, conserved, errors = realize(gw, se_static, (hole, particle), "mblse")
    stalled = [i for i, (got, want) in enumerate(conserved) if got < want]
    # A step-down is not the only way the recursion fails to deliver the moments it was
    # given.  ROADMAP 3.3 records the other: lithium-hydride's particle sector conserves
    # 20 of 20 at `nmom_max = 19` with a residual eighteen orders above the healthy band,
    # so its gate passes while its realization does not reproduce its own moments.  Both
    # are failures to price here, and looking only at the conserved count would miss the
    # second entirely.
    ungated = [i for i, error in enumerate(errors) if error > RESIDUAL_MAX]
    record = {
        "case": f"{name}_{xc}",
        "nmom_max": nmom_max,
        "conserved": conserved,
        "stalled": stalled,
        "ungated": ungated,
        "mblse_error": errors,
        "seconds": time.perf_counter() - started,
    }
    if not stalled and not ungated:
        return record

    # Only pay for the pencil and the second Dyson solve where there is a stall to price.
    pencil, _, pencil_errors = realize(gw, se_static, (hole, particle), "pencil")
    record["pencil_error"] = pencil_errors
    record["frontier_ev"] = frontier_difference(
        frontier_from_self_energy(gw, se_static, combined),
        frontier_from_self_energy(gw, se_static, pencil),
    )
    # The pencil stands in for "what the recursion would have said had it not failed", which
    # requires it to be trustworthy in absolute terms and not merely better than a broken
    # recursion.  It is held to the same `RESIDUAL_MAX` the order-convergence study uses --
    # six orders above the healthy band, twelve below the failures -- because section 4
    # predicts the pencil degrading with order and above `nmom_max = 11` it frequently does,
    # with reconstruction errors reaching 1e+11 in this sweep.  Scoring a frontier difference
    # against a pencil in that state measures the pencil's failure rather than the
    # recursion's, and counting those is what made an earlier version of this study report
    # 9.1 meV.
    record["valid"] = max(pencil_errors) < RESIDUAL_MAX and max(pencil_errors) < max(errors)
    return record


def main():
    """Run the sweep and print the verdict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        choices=sorted(baseline_systems.SYSTEMS),
        help="systems to sweep",
    )
    parser.add_argument(
        "--starting-points",
        nargs="+",
        default=list(DEFAULT_STARTING_POINTS),
        help="mean-field starting points",
    )
    parser.add_argument(
        "--orders", nargs="+", type=int, default=list(DEFAULT_ORDERS), help="moment orders"
    )
    parser.add_argument(
        "--compression", default="", help="auxiliary compression sectors; default is off"
    )
    args = parser.parse_args()

    print(f"momentGW imported from: {momentGW.__file__}")
    print(f"compression = {args.compression or 'off'}\n")
    print(
        f"{'case':26s} {'K':>3s} {'conserved h/p':>16s} {'mblse recon':>12s} "
        f"{'pencil recon':>13s} {'frontier (eV)':>14s}"
    )
    print("-" * 92)

    stalls = []
    for name in args.systems:
        for xc in args.starting_points:
            for nmom_max in args.orders:
                try:
                    record = run_case(name, xc, nmom_max, args.compression)
                except Exception as error:  # noqa: BLE001 - a failure is a result here
                    print(f"{name + '_' + xc:26s} {nmom_max:3d}   FAILED ({type(error).__name__})")
                    continue
                if not record["stalled"] and not record["ungated"]:
                    continue
                stalls.append(record)
                got = "/".join(f"{g}of{w}" for g, w in record["conserved"])
                if record["ungated"] and not record["stalled"]:
                    got += " !"
                cost = record["frontier_ev"]
                mark = "" if record["valid"] else "  (pencil worse - not a price)"
                print(
                    f"{record['case']:26s} {nmom_max:3d} {got:>16s} "
                    f"{max(record['mblse_error']):12.2e} {max(record['pencil_error']):13.2e} "
                    f"{'n/a' if cost is None else f'{cost:.3e}':>14s}{mark}"
                )

    print()
    if not stalls:
        print("No stall found anywhere in the swept set.")
        return
    priced = [r for r in stalls if r["frontier_ev"] is not None and r["valid"]]
    unusable = [r for r in stalls if r["frontier_ev"] is not None and not r["valid"]]
    print(f"{len(stalls)} stalled cases.")
    print(f"  {len(priced)} priced: the pencil reconstructs better, so it can stand in.")
    print(f"  {len(unusable)} not priced: the pencil is worse than the stalled recursion.")
    if not priced:
        print("\nNo stall could be priced. The pencil is not a usable fallback anywhere here.")
        return
    worst = max(priced, key=lambda r: r["frontier_ev"])
    print(
        f"\nLargest priced stall cost: {worst['frontier_ev']:.3e} eV "
        f"({worst['case']}, K = {worst['nmom_max']})"
    )
    verdict = "ABOVE" if worst["frontier_ev"] > MICROELECTRONVOLT else "below"
    print(f"That is {verdict} the 1e-6 eV bar HANKEL_PENCIL.md section 6.B set.")
    above = [r for r in priced if r["frontier_ev"] > MICROELECTRONVOLT]
    if above:
        print(f"\n{len(above)} priced stalls exceed the bar:")
        for record in sorted(above, key=lambda r: -r["frontier_ev"]):
            print(
                f"  {record['case']:26s} K = {record['nmom_max']:2d}  "
                f"{record['frontier_ev']:.3e} eV"
            )


if __name__ == "__main__":
    main()
