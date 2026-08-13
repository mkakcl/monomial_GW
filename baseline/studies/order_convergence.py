"""Document how the frontier converges with the moment order, and where it stops.

The Milestone 3 acceptance gate asks for order-convergence tables on H2O, LiH and the
selected small-gap system, and for evidence that increasing the supported order reduces the
stated frontier error until the floating-point or realization limit. This produces both.

Three things make the table cheap enough to sweep. The moments built at the cap contain
every lower order, so one construction serves the whole sweep. The frontier carries the
reference orbital dominating each pole, so a level crossing shows as a changed label rather
than as a large shift. And the truncation error has an indicator that does
not rest on differencing a sequence that is not monotonic: the spread between the Gauss and
Gauss-Radau closures, which is read within a single order.

The `diff` column is *not* a second independent indicator, and an earlier version of this
docstring said it was. It is the library's own `m` against `m - 2` estimate, which on an odd
sweep is the same pair of orders `dHOMO` differences, so the two agree in magnitude to every
printed digit in all 37 rows that print both (K=1 in each table has neither). Only in
magnitude: `differencing_ev` is stored as an absolute value, so wherever the frontier falls
`dHOMO` is negative and `diff` is its modulus. It is kept because it
exercises the estimate the acceptance gate relies on against a number computed here
independently - a cross-check, not a second measurement - and
it costs one extra realization and Dyson solve per row.

**There is exactly one way this code steps down**, and it is worth stating plainly because
an earlier version of this study got it wrong. `MBLSE.kernel` sets `max_cycle_achieved` in
one place only, catching `NotPositiveSemiDefiniteError` from the next block's square root
(`dyson/solvers/static/_mbl.py:471`). So every shortfall is a **PSD failure** - the moments
cannot support a causal measure at that order - and there is no rank-versus-arithmetic
dichotomy to classify. Importing one from a code that gates on a Gram factorisation instead
produced two false readings: the reconstructed-moment residual is measured at the *achieved*
order, which the failure never touched, so it cannot indicate the cause.

What the residual can and cannot tell you `[corrected 2026-08-12]`. It is measured at the
*achieved* order, which the PSD failure never touched, so it cannot say why the recurrence
stopped where it did - that much of the original claim stands. It was also asserted to sit
at ~1e-15 at every step-down, and that was a generalisation from a sweep capped at 15:
lithium-hydride's particle sector carries 3.30e+04 from nmom_max = 19 onward, having
stepped down at 21 and 23. So the residual is not a diagnosis of the step-down, but it is
not uninformative either - it says whether the realization reproduces the moments it was
given, which is a separate question from how many orders it claims, and the two can
disagree by twenty orders of magnitude.

What the PSD gate costs is measurable, and is measured here. Loosening `neg_atol` and
`neg_rtol` lets the recurrence accept the offending direction and continue; the study
reports the order that buys and the residual it costs, so the trade is visible rather than
asserted. Note that it is `neg_atol`/`neg_rtol` that govern this, not `atol`/`rtol` - the
latter set the support mask and provably cannot move the step-down.

A study, not part of the recorded baseline set: re-run when the claims it supports are in
question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the printed
`momentGW.__file__` before believing a comparison. With no `--cap` each system is swept to
the cap its own limit needs, so the bare command reproduces every claim the acceptance gate
records, ozone's included::

    python -m baseline.studies.order_convergence                          # all three, ~35 s
    python -m baseline.studies.order_convergence --systems ozone          # cap 31, 22 s
    python -m baseline.studies.order_convergence --systems water --cap 9
"""

import argparse
import contextlib
import io
import math
import traceback

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

#: Factor applied to `neg_atol`/`neg_rtol` when probing what the PSD gate costs. Four orders
#: is enough to admit the direction the recurrence refused wherever the gate is the binding
#: constraint, and the residual reported beside it says what admitting it cost.
LOOSEN = 1e4

#: How far each system is swept, **measured at the default PBE reference**. A limit moves
#: with the reference - hydrogen-631g pins its particle at 8 under PBE and at 6 under HF -
#: so these caps are not transferable, and `main` says so when `--xc` is changed without an
#: explicit `--cap`.
#: All three reach their final realization at nmom_max = 19
#: and are frozen from there, so a cap of 23 shows the limit and two orders past it. Ozone
#: goes further because it is the acceptance gate's small-gap case and the one that was
#: previously recorded as having no limit at all: sweeping it to 31 confirms the frozen
#: region persists for seven orders rather than two. The old single cap of 15 was below all
#: three limits, not just ozone's, and stopped one order short of lithium-hydride's
#: residual blow-up at 19.
DEFAULT_CAPS = {
    "water": 23,
    "lithium-hydride": 23,
    "ozone": 31,
}

#: Systems with no realization limit to measure. Both hydrogens pin their particle sector
#: early - 8 under PBE, 6 under HF - while the hole gains indefinitely, so no order freezes
#: the whole realization and any cap is a reading budget. Listed rather than left out of
#: `DEFAULT_CAPS`, so the note below can say which of the two situations a reader is in.
NO_MEASURED_LIMIT = ("hydrogen", "hydrogen-631g")

#: For a system with no entry above. 23 rather than the old 15, which is below the limit of
#: every system measured here - a new system given 15 would get a table that looks converged
#: for exactly the reason this default was raised. Not a substitute for measuring its own
#: limit and adding it to `DEFAULT_CAPS`.
DEFAULT_CAP = 23

#: Reference the caps above were measured at.
DEFAULT_XC = "pbe"


#: Above this reconstructed-moment residual the realization does not reproduce the moments
#: it was given, whatever its conserved order says. Healthy values across these systems are
#: 2.4e-15 to 2e-14, and lithium-hydride's particle sector reaches ~3e+04 at nmom_max = 19
#: while conserving 20 of 20 - so the gate beside it passes. Six orders above the observed
#: band and twelve below that failure: nothing legitimate is near it, which is the point of
#: putting it so far from both rather than tuning it.
#:
#: The magnitude of that failure is not reproducible and should not be quoted as though it
#: were. The moments are built once at the cap and sliced, and the slicing is equal only to
#: roundoff - but a realization that has stopped reproducing its moments amplifies roundoff
#: without bound, so the same K=19 particle reads 7.7e+03 swept to 21, 3.3e+04 to 23 and
#: 3.5e+04 to 19. What is stable is that it is twelve orders above this threshold and
#: eighteen above a healthy residual. Quote the gap, not the number.
RESIDUAL_MAX = 1e-8


class ProbeError(Exception):
    """Some PSD probes raised, so this sweep has no verdict on what the gate costs."""


class LayoutError(Exception):
    """The diagnostics do not have the shape this table is laid out for.

    Its own type so the probe loop, which catches failures as results about the PSD gate,
    re-raises it instead of printing a structural bug as physics.
    """


#: Sectors this table is laid out for, in column order. The restricted molecular G0W0 path
#: resolves the self-energy into exactly these two, and the fixed-width columns and the
#: `h/p` headings both assume it. `_by_sector` reads whatever the diagnostics carry, so a
#: third sector is caught here rather than quietly overflowing a column and leaving the
#: headings pointing at the wrong numbers.
SECTORS = ("hole", "particle")


def _by_sector(realization):
    """Read the conserved order, requested order and residual for every sector.

    The one place that knows the shape of `dyson_diagnostics["realization"]`; everything
    below reads sectors through this.
    """
    # Before any field is read: a differently shaped dict would otherwise raise KeyError
    # from the loop below, and the probe loop would report that structural bug as a result
    # about the PSD gate - which is what the typed error exists to prevent.
    if tuple(realization) != SECTORS:
        raise LayoutError(
            f"table is laid out for sectors {SECTORS}, got {tuple(realization)}: widen the "
            "columns and the headings before sweeping this path"
        )
    # The field names too, not just the sector names: a rename in `realization_record` would
    # otherwise reach the loop below as a KeyError, which the probe loop catches and reports
    # as a result about the PSD gate.
    needed = {"nmom_conserved_achieved", "nmom_conserved_requested", "moments_supplied", "errors"}
    for name, record in realization.items():
        if missing := needed - set(record):
            raise LayoutError(f"the {name} record is missing {sorted(missing)}")

    by_sector = {}
    for name, record in realization.items():
        conserved = int(record["nmom_conserved_achieved"])
        requested = int(record["nmom_conserved_requested"])
        by_sector[name] = {
            "conserved": conserved,
            "requested": requested,
            "supplied": int(record["moments_supplied"]),
            "shortfall": requested - conserved,
            "residual": (
                float(record["errors"].max_relative_frobenius)
                if record["errors"] is not None
                else float("nan")
            ),
        }
    return by_sector


def _pin_key(by_sector):
    """Reduce a realization to what decides whether the next order can differ from it.

    The conserved order per sector, and nothing else. Two orders that conserve the same in
    every sector produce the same recurrence blocks from the same moment sequence, so their
    answers are identical and any shift between them is zero by construction. The residual
    is deliberately excluded: it is `nan` whenever errors are not being calculated, and
    `nan != nan` would make every comparison false and silently switch the marking off
    entirely - and being a float, it is exposed to the BLAS summation order that CLAUDE.md
    warns changes with the thread count.
    """
    return tuple((name, entry["conserved"]) for name, entry in by_sector.items())


def _sectors(by_sector, field, fmt="{:d}"):
    """Join one per-sector field for the table, in the order the diagnostics give them."""
    return "/".join(fmt.format(v[field]) for v in by_sector.values())


def _row(gw, order, nelectron):
    """Read one row of the table off a solved calculation."""
    diagnostics = gw.dyson_diagnostics
    # Per sector throughout, never the binding minimum: the minimum cannot tell a sector
    # that stepped down from one that never had further to go. On ozone at nmom_max=19 the
    # hole stops at 18 while the particle reaches 20, so the row moves even though its
    # binding order is the same as nmom_max=17's.
    by_sector = _by_sector(diagnostics["realization"])
    # Both sectors are asked for the same order from the same moment build, so these are
    # one number rather than a column each - but read them off the sectors, so a path
    # where that stops holding is caught instead of silently printing one sector's.
    requested = {v["requested"] for v in by_sector.values()}
    if len(requested) != 1:
        # A property of this calculation, not of the table, so not a LayoutError: `main`
        # abandons every remaining system for those, and the other systems are still fine.
        raise ValueError(f"sectors disagree on what was asked of them: {by_sector}")
    # Not asserted equal to `requested` - at an even order dyson conserves one fewer than it
    # is handed, and says so - but recorded, so a build that hands the recurrence a
    # different count than it is asked to conserve is visible rather than invisible.

    shortfall = max(v["shortfall"] for v in by_sector.values())

    # One column, `asked`, holding the order the recurrence was asked to conserve. On this
    # sweep `moments_supplied` happens to equal it - both are `order + 1` in all 40 rows of
    # the three tables, because every order swept is odd - but that is a property of the
    # sweep, not an invariant: at an even order dyson conserves one fewer than it is handed
    # and says so. Only the requested order is printed, so the two are not asserted equal.
    row = {
        "order": order,
        "asked": next(iter(requested)),
        # Per sector, so a build that hands one sector a different count than it is asked to
        # conserve shows up. Reported in the footer rather than in the column: at an even
        # order dyson conserves one fewer than it is handed, and a widened or overflowing
        # `asked` field would push every column right of it off its heading.
        "surplus": {k: v["supplied"] - v["requested"] for k, v in by_sector.items()},
        "by_sector": by_sector,
        # The largest over the sectors, used only to decide whether this order stepped down
        # at all; the table prints the shortfall per sector.
        "shortfall": shortfall,
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


def _runs(orders, step):
    """Split ascending orders into maximal runs that step by exactly `step`.

    Both the probe verdicts and the residual warning collapse repeats into `K=a to b`, and
    both are wrong if they bridge an order the sweep skipped or that behaved differently.
    Written once, and given the stride rather than assuming the sweep is over odd orders.
    """
    if not orders:
        return []
    runs = [[orders[0]]]
    for order in orders[1:]:
        if order == runs[-1][-1] + step:
            runs[-1].append(order)
        else:
            runs.append([order])
    return runs


def _format_runs(runs):
    """Format already-split runs as `K=a`, `K=a to b`, comma-joined."""
    return ", ".join(f"K={r[0]}" if len(r) == 1 else f"K={r[0]} to {r[-1]}" for r in runs)


def _span(orders, step):
    """Format ascending orders as the runs they fall into."""
    return _format_runs(_runs(orders, step))


def _residual_report(rows, stride):
    """Report realizations whose reconstruction residual is unusable, or unmeasured.

    Grouped by the blown sector's own conserved order rather than by the row: once that
    sector pins, every higher order repeats its realization, and keying on the row would
    count a fresh failure each time the *other* sector gained. Grouped again into contiguous
    runs within that, because a sector can recover and blow up again at the same conserved
    count, and those are two failed realizations rather than one long one.

    `nan` is reported separately rather than filtered out. It is what `_by_sector` stores
    when errors are not being calculated, and `nan > RESIDUAL_MAX` is False, so a sweep that
    measured nothing about realization fidelity would otherwise be indistinguishable from
    one where every residual was healthy.

    Returns
    -------
    lines : list of str
        Warnings to print, empty if every residual was measured and healthy.
    """
    blown, unmeasured, surplus = [], {}, {}
    for row in rows:
        for sector, entry in row["by_sector"].items():
            if math.isnan(entry["residual"]):
                unmeasured.setdefault(sector, []).append(row["order"])
            elif entry["residual"] > RESIDUAL_MAX:
                blown.append((row["order"], sector, entry["residual"], entry["conserved"]))
        for sector, extra in row.get("surplus", {}).items():
            if extra:
                surplus.setdefault((sector, extra), []).append(row["order"])

    lines = []
    if surplus:
        where = "; ".join(
            f"{sector} handed {extra:+d} at {_span(orders, stride)}"
            for (sector, extra), orders in sorted(surplus.items())
        )
        lines.append(
            f"  NOTE: the build did not hand the recurrence what it was asked to conserve "
            f"({where}), so `asked` describes the request and not the supply"
        )
    if unmeasured:
        # Scoped to the orders it applies to: saying "this sweep" would retract evidence
        # from every order that was measured and healthy.
        where = "; ".join(
            f"{sector} at {_span(orders, stride)}" for sector, orders in sorted(unmeasured.items())
        )
        lines.append(
            f"  WARNING: no residual was calculated for {where}, so nothing there says "
            "whether those realizations reproduce their moments - a healthy-looking row is "
            "an absence of evidence, not evidence"
        )
    if not blown:
        return lines

    by_realization = {}
    for order, sector, _, conserved in blown:
        by_realization.setdefault((sector, conserved), []).append(order)

    parts, distinct = [], 0
    for (sector, conserved), orders in sorted(by_realization.items()):
        # One split, so the count and the printed span cannot drift apart.
        runs = _runs(orders, stride)
        distinct += len(runs)
        parts.append(f"{sector} at {conserved} conserved over {_format_runs(runs)}")

    worst = max(blown, key=lambda b: b[2])
    lines.append(
        f"  WARNING: reconstructed-moment residual reaches {worst[2]:.2e} in the "
        f"{worst[1]} sector at K={worst[0]}, above {RESIDUAL_MAX:g} - {distinct} distinct "
        f"realization(s): {'; '.join(parts)}. That realization does not reproduce the "
        "moments it was given, whatever its conserved order says, and no gate covers it"
    )
    return lines


def _classify_residuals(before, reached):
    """Split the loosened run's sectors by what their residual says, and who did it.

    Four outcomes, because collapsing them loses the fact that matters. A sector loosening
    pushed past `RESIDUAL_MAX` is a cost of the knob; one that was already past it is the
    ungated failure this study reports and not the knob's doing, and reading it as a cost
    would void orders another sector genuinely bought. A `nan` residual - what `_by_sector`
    stores when errors are not being calculated - is neither: it means the check did not
    run, and every comparison against it is False, so without this it would silently land
    in "already" and assert the opposite of the truth.

    Returns
    -------
    caused, already, repaired : dict
        Sector to residual, for blow-ups the loosening created, ones that predate it, and
        ones it fixed. The last is easy to omit and produces the worst reading: with the
        conserved counts unchanged the verdict would say "no sector moves when it is
        loosened" for a run where the loosened realization is the only one reproducing its
        own moments.
    unknown : set
        Sectors whose residual is `nan` at either end, so nothing can be said.
    """
    caused, already, repaired, unknown = {}, {}, {}, set()
    for sector, entry in reached.items():
        after, prior = entry["residual"], before[sector]["residual"]
        if math.isnan(after) or math.isnan(prior):
            unknown.add(sector)
        elif after > RESIDUAL_MAX:
            (already if prior > RESIDUAL_MAX else caused)[sector] = after
        elif prior > RESIDUAL_MAX:
            repaired[sector] = after
    return caused, already, repaired, unknown


def _flush_probe(pending, stride):
    """Print a held probe verdict, as a single order or as the range it held over."""
    if pending is None:
        return
    _, body, orders = pending
    print(f"  {_span(orders, stride)} {body}")


def _psd_probe(gw, moments, integrals, order):
    """Measure what the PSD gate is costing at a stepped-down order.

    Runs the solver it is handed - `_loosened` is what relaxes `neg_atol`/`neg_rtol`, and
    those are the tolerances that govern the step-down; `atol`/`rtol` set the support mask
    and cannot move it. **Hand it `_loosened(mf, LOOSEN)`, never the sweep's own solver**: a
    probe on an unrelaxed solver measures nothing and reports "no sector moves when it is
    loosened" at every order, a false negative on the gate's central claim. `dyson_opts`
    rejects keys it does not know, so there is nowhere to stash a marker to check for.

    Returns
    -------
    by_sector : dict
        Conserved order and residual per sector with the gate loosened. Per sector rather
        than the binding minimum: a lift in one sector alone would otherwise be invisible
        and the verdict would read "unchanged" for a gate that had moved, and pairing a
        per-sector order with a binding-sector residual compares two different sectors
        whenever loosening moves which one binds.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        gw.kernel(nmom_max=order, moments=moments, integrals=integrals)
    return _by_sector(gw.dyson_diagnostics["realization"])


def _loosened(mf, loosen):
    """Build a solver with the PSD tolerances relaxed.

    Separate from `_psd_probe` so the caller can construct it outside the handler that
    catches probe failures: a bug here is a bug in this study, and printing it as a finding
    about the PSD gate is the misreporting the typed errors exist to prevent.
    """
    gw = GW(mf, polarizability="drpa")
    gw.dyson_opts = dict(
        gw.dyson_opts,
        neg_atol=gw.dyson_opts["neg_atol"] * loosen,
        neg_rtol=gw.dyson_opts["neg_rtol"] * loosen,
    )
    return gw


def run(name, cap, xc=DEFAULT_XC):
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
        f"{'K':>3s} {'asked':>5s} {'cons h/p':>9s} {'short h/p':>10s} "
        f"{'resid h/p':>17s} {'limit':>6s} {'HOMO/eV':>10s} {'Z':>5s} {'MO':>3s} "
        # The two shift fields carry a trailing mark, so their headings sit over the number
        # and the mark column is part of the separator. Otherwise every value in them reads
        # one column left of its own heading.
        f"{'dHOMO/meV':>10s}  {'diff/meV':>9s}  {'closure/meV':>11s}  {'dN':>9s}  gates"
    )
    rows, previous, previous_sector, marked = [], None, None, False
    # Named `stride`, not `step`: the row loop below uses `step` for the dHOMO shift.
    stride = 2
    for order in range(1, cap + 1, stride):
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
        cons = _sectors(row["by_sector"], "conserved")
        short = _sectors(row["by_sector"], "shortfall")
        # Three significant figures, the same as the probe lines below print for the same
        # quantity: one figure cannot tell 1.0e-14 from 1.4e-14, which is the resolution the
        # acceptance gate quotes its band at, nor a 3e+04 blow-up from a 3e-04 entry.
        resid = _sectors(row["by_sector"], "residual", "{:.2e}")
        failed_gates = [k for k, v in row["gates"].items() if not v]
        gates = "ok" if not failed_gates else ",".join(failed_gates)
        if row["homo_ev"] is None:
            print(
                f"{row['order']:3d} {row['asked']:5d} "
                f"{cons:>9s} {short:>10s} {resid:>17s} "
                f"{row['limit']:>6s}  dN {row['nelec_error']:.2e}  gates {gates}"
                f"  no frontier: {row.get('frontier_error', '')}"
            )
            # Clear the energy predecessor rather than skipping past it: `dHOMO` would
            # otherwise silently span two orders under a header that means one. The sector
            # state is kept, because the realization here succeeded - it is only the
            # frontier readout that failed - and `diff` is the solver's own m-against-m-2
            # estimate, which does not depend on this row having been printable.
            previous = None
            previous_sector = _pin_key(row["by_sector"])
            continue
        # Both shift columns difference this row against the one before it - `dHOMO`
        # directly, `diff` as the m-against-m-2 estimate, which on an odd sweep is the same
        # pair. If the realization did not change between them the two answers are identical
        # by construction, so the shift is zero because the realization is pinned, not
        # because the frontier has settled. Marked rather than blanked, so the frozen region
        # stays visible as a run of zeros.
        pinned = previous_sector is not None and _pin_key(row["by_sector"]) == previous_sector
        mark = "*" if pinned else " "
        # Each of the three columns drops its mark when it has no value to carry it, so the
        # legend follows the marks that reach the page, not the rows that qualified for one.
        marked = marked or (
            pinned
            and (
                previous is not None
                or row["differencing_ev"] is not None
                or row["closure_ev"] is not None
            )
        )
        step = "" if previous is None else f"{(row['homo_ev'] - previous) * 1000:10.2f}{mark}"
        previous = row["homo_ev"]
        previous_sector = _pin_key(row["by_sector"])
        # The placeholder has to carry the mark column too, or it lands one to the right of
        # its own heading while the numbers beside it are correct.
        diff = (
            f"{'-':>9s} "
            if row["differencing_ev"] is None
            else f"{row['differencing_ev'] * 1000:9.2f}{mark}"
        )
        # The closure spread is read off the same two realizations, so it freezes with them
        # and needs the same mark: an unflagged 98.48 beside a flagged 0.00* reads as live
        # evidence at an order where it is the earlier number repeated by construction.
        clos = (
            f"{'-':>11s} "
            if row["closure_ev"] is None
            else f"{row['closure_ev'] * 1000:11.2f}{mark}"
        )
        print(
            f"{row['order']:3d} {row['asked']:5d} "
            f"{cons:>9s} {short:>10s} {resid:>17s} {row['limit']:>6s} "
            f"{row['homo_ev']:10.5f} {row['homo_z']:5.3f} {row['homo_mo']:3d} {step:>11s} "
            f"{diff:>10s} {clos:>12s} {row['nelec_error']:9.2e}  {gates}"
        )

    if marked:
        print(
            "  NOTE: * marks dHOMO, diff and closure at an order that realized identically "
            "to the order before it, so the value is fixed by that rather than measured "
            "afresh - for the two shifts it makes them zero, for the spread it repeats a "
            "still-live measurement. cons, HOMO, Z, MO, dN and gates freeze with it and carry "
            "no mark, being readings of that realization rather than comparisons of two. "
            "resid is not marked either but is not guaranteed identical: a blown "
            "realization amplifies roundoff, so it can move between two marked rows. short "
            "keeps growing, being the gap to a request that keeps rising"
        )

    # Probe every stepped-down order, not just the first and not one per distinct achieved
    # state. Each probe supplies a different number of moments, so no two ask the same
    # question, and deduplicating on the achieved state skips the whole frozen region -
    # exactly where "loosening the gate recovers nothing" has to hold to mean anything.
    # Consecutive orders that give the identical verdict are printed as a range. The probes
    # all still run - each supplies different moments, so the answer is measured rather than
    # assumed - but a reader scanning seven byte-identical lines learns what one line and a
    # range say.
    pending = None
    probes_failed = 0
    for row in (r for r in rows if r["shortfall"]):
        before = row["by_sector"]
        # Built outside the `try`: a failure constructing the loosened solver is a bug in
        # this study, not a result about the PSD gate, and must not be printed as one.
        probe_moments = tuple(m[: row["order"] + 1] for m in moments)
        probe_gw = _loosened(mf, LOOSEN)
        try:
            reached = _psd_probe(probe_gw, probe_moments, integrals, row["order"])
        except LayoutError:
            # Fatal, but do not swallow verdicts already measured and held for grouping.
            _flush_probe(pending, stride)
            raise
        except Exception as error:
            # A probe deliberately asks the recurrence to accept a direction it refused, so
            # a failure downstream is a result about the gate, not a reason to abandon the
            # remaining probes or discard the footer for a table already printed.
            _flush_probe(pending, stride)
            pending = None
            probes_failed += 1
            print(
                f"  K={row['order']:2d} conserved {_sectors(before, 'conserved')}; the "
                f"loosened run failed: {type(error).__name__}: {error}"
            )
            continue
        caused, already, repaired, unknown = _classify_residuals(before, reached)
        gained = {k: reached[k]["conserved"] - before[k]["conserved"] for k in before}
        # `abs` because the sign belongs to the verb: "loses hole -2" reads as a gain.
        up = ", ".join(f"{k} +{v}" for k, v in gained.items() if v > 0)
        down = ", ".join(f"{k} -{abs(v)}" for k, v in gained.items() if v < 0)
        if up and down:
            # Reporting only the gain here would call a gate "binding" while loosening it
            # was destroying the other sector.
            verdict = f"mixed: loosening it gains {up} but loses {down}"
        elif up:
            verdict = f"the PSD gate is binding; loosening it gains {up}"
        elif down:
            verdict = f"loosening the gate made it worse: {down}"
        else:
            verdict = "not the PSD tolerance: no sector moves when it is loosened"
        # Appended rather than replacing the verdict: whether orders were bought and whether
        # the result reproduces its moments are two separate facts about two possibly
        # different sectors.
        if caused:
            cost = ", ".join(f"{k} at {v:.2e}" for k, v in caused.items())
            verdict += (
                f" - but loosening pushes {cost} past {RESIDUAL_MAX:g}, so whatever those "
                "sectors conserve does not reproduce their moments"
            )
        if already:
            note = ", ".join(f"{k} at {v:.2e}" for k, v in already.items())
            verdict += (
                f" (note {note}, above {RESIDUAL_MAX:g} both before and after, so that is "
                "not the loosening's doing)"
            )
        if repaired:
            fixed = ", ".join(f"{k} to {v:.2e}" for k, v in repaired.items())
            verdict += (
                f" - and it brings {fixed} back under {RESIDUAL_MAX:g}, so the loosened "
                "realization is the one that reproduces its moments"
            )
        if unknown:
            verdict += (
                f" (residual unavailable for {', '.join(sorted(unknown))}, so nothing here "
                "says whether the realization reproduces its moments)"
            )
        body = (
            f"conserved {_sectors(before, 'conserved')} at residual "
            f"{_sectors(before, 'residual', '{:.2e}')}; with neg_atol/neg_rtol x{LOOSEN:g} it "
            f"conserves {_sectors(reached, 'conserved')} at "
            f"{_sectors(reached, 'residual', '{:.2e}')} -> {verdict}"
        )
        # The whole of `body` is the key, residuals included. A collapsed `K=a to b` prints
        # one order's numbers for the whole range, so anything the line states has to be
        # identical across it or the line asserts a measurement that was never taken - and
        # the residuals are what the acceptance gate quotes as bands. The cost is that a
        # third-digit move splits a run into separate lines; a verbose report beats a
        # confidently wrong one.
        key = body
        # Only group orders that are genuinely adjacent in the sweep. A row with no
        # shortfall between two stepped-down rows is skipped by the filter above, and
        # extending across it would print a verdict at an order where no probe ran.
        if pending is not None and pending[0] == key and pending[2][-1] + stride == row["order"]:
            pending[2].append(row["order"])
        else:
            _flush_probe(pending, stride)
            pending = (key, body, [row["order"]])
    _flush_probe(pending, stride)
    if any(r.get("crossed") for r in rows):
        print("  NOTE: the dominant frontier orbital changed between consecutive orders")
    if any(r.get("closure_crossed") for r in rows):
        print(
            "  NOTE: the two closures disagreed on the frontier state at least once, so "
            "the spread there is between different states"
        )
    # A conserved order says how many moments the recurrence claims; the residual says
    # whether the realization reproduces them. They can disagree, and nothing in the gate
    # set catches it: lithium-hydride's particle sector conserves 20 of 20 at nmom_max=19,
    # so its realization gate passes, while missing its own input moments by four orders of
    # magnitude. Until a gate covers it, the study says so.
    for line in _residual_report(rows, stride):
        print(line)

    weights = [r["homo_z"] for r in rows if r.get("homo_z") is not None]
    worst_z = min(weights) if weights else None
    if worst_z is not None and worst_z < QUASIPARTICLE_WEIGHT_MIN:
        print(
            f"  WARNING: quasiparticle weight falls to {worst_z:.3f}, below "
            f"{QUASIPARTICLE_WEIGHT_MIN}: the frontier energy is tracking a satellite, so "
            "its convergence says nothing about a quasiparticle"
        )

    # Last, after every footer. The table and its warnings are about rows that printed and
    # stand on their own; what is missing is the gate verdict, and the acceptance gate quotes
    # that, so a sweep short of one must not be quotable as though it had measured it.
    if probes_failed:
        raise ProbeError(
            f"{probes_failed} PSD probe(s) failed; the table above is complete but its gate "
            "verdicts are not"
        )
    return rows


def main():
    """Run the sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="override the per-system cap; by default each system uses the cap its limit "
        "needs (see DEFAULT_CAPS)",
    )
    parser.add_argument("--xc", default=DEFAULT_XC)
    args = parser.parse_args()
    caps = {**DEFAULT_CAPS, "<default>": DEFAULT_CAP}
    bad = {n: c for n, c in caps.items() if c < 1 or c % 2 == 0}
    if bad:
        # The same rule the flag is held to: a table entry bypasses the check below entirely.
        raise SystemExit(f"DEFAULT_CAPS entries must be positive odd orders, got {bad}")
    if args.cap is not None and (args.cap < 1 or args.cap % 2 == 0):
        # An even cap sweeps one order short of what the heading claims, and a cap below 1
        # prints a heading and a column header with no rows at all - a table that looks fine
        # and says nothing, which is the failure this whole study exists to catch.
        raise SystemExit(f"--cap must be a positive odd order, got {args.cap}")

    print(f"momentGW: {momentGW.__file__}")
    if args.xc != DEFAULT_XC and args.cap is None:
        # The caps below were measured at PBE; at another reference the sweep can stop short
        # of the limit, which is the failure that made the old fixed cap of 15 wrong.
        print(
            f"  NOTE: --xc {args.xc} with per-system caps measured at {DEFAULT_XC}. They may "
            "not reach this reference's limits; pass --cap to sweep further."
        )
    unknown = [n for n in args.systems if n not in systems_module.SYSTEMS]
    if unknown:
        # A typo is one mistake, not one numerical failure per system requested.
        raise SystemExit(
            f"unknown system(s): {', '.join(unknown)}. Known: {', '.join(systems_module.SYSTEMS)}"
        )

    failed = []
    for name in args.systems:
        measured = DEFAULT_CAPS.get(name)
        cap = args.cap if args.cap is not None else (measured or DEFAULT_CAP)
        if measured is not None and cap < measured:
            # An explicit `--cap` below the measured limit is the original failure: a table
            # that ends before the realization pins and reads as though it had converged.
            print(
                f"\n  NOTE: {name} is swept to {measured} by default because that is what "
                f"its limit needs; --cap {cap} stops short of it, so the table below does "
                "not reach a limit."
            )
        # Both notes describe the system, not how the cap was chosen, so neither is
        # conditional on `--cap` being absent: an explicit cap does not give a system a
        # limit it does not have.
        if name in NO_MEASURED_LIMIT:
            print(
                f"\n  NOTE: {name} has no realization limit to reach - its particle sector "
                "pins early while the hole gains indefinitely - so this cap is a reading "
                "budget and the table below does not end at a limit."
            )
        elif measured is None:
            # Silence here would read as "this cap was measured", which is the reading that
            # made the old fixed cap of 15 wrong.
            print(
                f"\n  NOTE: {name} has no measured cap; using the {DEFAULT_CAP} default. "
                "Check its realization pins inside that before quoting the table."
            )
        try:
            run(name, cap, xc=args.xc)
        except ProbeError as error:
            # Distinct from the abort below: `run` raises this at the very end, so the table
            # is complete and worth keeping - only the gate verdicts are short.
            print(f"\n  INCOMPLETE: {name} - {error}")
            failed.append(name)
        except LayoutError:
            # Structural, not numerical: every system on this path will hit it, so failing
            # once is the honest outcome rather than three identical "mid-sweep" aborts.
            # Anything already known to be incomplete still has to reach stdout first.
            print(
                f"\n  ABORTED: {name} - the diagnostics are not the shape this table "
                "reads; every system on this path will hit the same fault"
            )
            failed.append(name)
            if failed:
                print(f"\nincomplete: {', '.join(failed)}")
            raise
        except Exception:
            # One system failing must not cost the others their tables: the acceptance gate
            # reads all three, and the sweep is long enough that re-running to recover a
            # system that would have worked is a real cost. The marker goes to stdout as
            # well as the traceback to stderr, because stdout is what gets pasted into the
            # roadmap and a table that stops mid-sweep otherwise looks merely short.
            # `run` raises before printing its heading for anything that fails during
            # setup, so "the table above" would point at the previous system's finished
            # one. Name the system that failed and say nothing about position.
            print(f"\n  ABORTED: {name} raised; it has no complete table in this run")
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f"\nincomplete: {', '.join(failed)}")
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
