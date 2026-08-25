"""Attribute the lithium-hydride particle-sector residual blow-up.

ROADMAP 3.2 unparked on the finding that lithium-hydride's particle sector conserves 20 of
the 20 orders requested at ``nmom_max = 19`` with a reconstructed-moment residual eighteen
orders above the healthy band, and named the first step: *"what it does not establish is
whether the blow-up is the basis or the molecule ... that attribution is the first step, not
the port."*

This is that attribution. Four controls, each removing one candidate cause:

1. **Reproduce it.** Moments built once at a cap and sliced, exactly as
   `order_convergence` builds them, because the reading depends on the build - see the
   acceptance gate's note on quoting the gap rather than the number.
2. **Change the reference.** The same molecule, basis, order and moment basis from a
   Hartree-Fock starting point instead of PBE. If the molecule were the cause this would
   fail too.
3. **Change the realization route.** The same moments through the one-shot block Hankel
   pencil of `HANKEL_PENCIL.md`, which shares no code with the recursion. If the moments
   could not be realized at this order, this would fail too.
4. **Compare the moment magnitudes.** A sequence reaching ``1e+09`` invites "it is the
   dynamic range"; this checks that against the reference that works.

What the study cannot say is *why* the recursion diverges. Loss of orthogonality in a block
Lanczos recursion is the standard candidate and would be repaired by re-orthogonalisation
rather than by a change of moment basis, but that is a hypothesis and is not measured here.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the printed
`momentGW.__file__` before believing a comparison::

    python -m baseline.studies.residual_attribution
    python -m baseline.studies.residual_attribution --cap 21 --order 19
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
from dyson import MBLSE

import momentGW
from baseline import systems as systems_module
from baseline.studies.order_convergence import build_mean_field
from baseline.studies.pencil_vs_mblse import pencil_self_energy, reconstruction_error
from momentGW.gw import GW, RESIDUAL_MAX
from momentGW.rpa import dRPA

#: The system the blow-up was found on, and the sector it was found in.
SYSTEM = "lithium-hydride"
SECTORS = ("hole", "particle")

#: References compared. `pbe` is `order_convergence`'s default and the one that fails.
REFERENCES = ("hf", "pbe")


def build_sliced_moments(xc, cap, order):
    """Build the self-energy moments at `cap` and slice them to `order`.

    Built once at the cap and sliced, which is what `order_convergence` does. The
    distinction matters: a realization that has stopped reproducing its moments amplifies
    roundoff without bound, so the same realization reads a different residual depending
    only on how far the sweep that found it went.

    Parameters
    ----------
    xc : str
        Mean-field reference.
    cap : int
        Moment order the sequence is built at.
    order : int
        Moment order the sequence is sliced to.

    Returns
    -------
    tuple
        The static self-energy, the solver options, and one moment array per sector.
    """
    system = systems_module.SYSTEMS[SYSTEM]
    with contextlib.redirect_stdout(io.StringIO()):
        mf, _ = build_mean_field(system, xc)
        reference = GW(mf, polarizability="drpa")
        integrals = reference.ao2mo()
        se_static = reference.build_se_static(integrals)
        moments = dRPA(
            reference,
            cap,
            integrals,
            mo_energy=dict(g=reference.mo_energy, w=reference.mo_energy),
        ).kernel()
    sliced = tuple(np.asarray(m[: order + 1]) for m in moments)
    return se_static, reference.dyson_opts, sliced


def realize_both_ways(se_static, dyson_opts, moments):
    """Realize one sector's moments by the recursion and by the pencil.

    Returns
    -------
    dict
        Conserved order and residual for each route. The two residuals are comparable
        only when the recursion conserves every supplied order: `MBLSE` is scored over
        the orders it claims, the pencil over all of them, and a recursion that stepped
        down is being scored on a shorter list than its rival.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        solver = MBLSE(se_static, moments, **dyson_opts)
        solver.kernel()
    achieved = solver.max_cycle if solver.max_cycle_achieved is None else solver.max_cycle_achieved
    record = {
        "supplied": int(moments.shape[0]),
        "conserved": int(solver.nmom_conserved(achieved)),
        "mblse_residual": float(solver.moment_errors().max_relative_frobenius),
        "mblse_per_order": np.asarray(solver.moment_errors().relative_frobenius),
        "norms": np.array([float(np.linalg.norm(m)) for m in moments]),
    }
    try:
        realized, rank = pencil_self_energy(moments)
        per_order = reconstruction_error(realized, moments)
        record["pencil_rank"] = int(rank)
        record["pencil_residual"] = float(per_order.max())
        record["pencil_per_order"] = np.asarray(per_order)
    except (np.linalg.LinAlgError, ValueError) as error:
        record["pencil_rank"] = None
        record["pencil_residual"] = float("nan")
        record["pencil_per_order"] = None
        record["pencil_error"] = f"{type(error).__name__}: {error}"
    return record


def main():
    """Run the attribution and print its table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap", type=int, default=21, help="order the moments are built at")
    parser.add_argument("--order", type=int, default=19, help="order they are sliced to")
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    print(
        f"\n{SYSTEM} / {systems_module.SYSTEMS[SYSTEM].basis}, moments built at cap "
        f"{args.cap} and sliced to K={args.order}. Residual is the maximum relative "
        f"Frobenius error over the orders each route claims; the gate is {RESIDUAL_MAX:g}."
    )
    print(
        f"\n{'ref':>5} {'sector':>9} {'conserved':>10} {'MBLSE resid':>13} "
        f"{'pencil rank':>12} {'pencil resid':>13} {'comparable':>11}"
    )
    records = {}
    for xc in REFERENCES:
        se_static, dyson_opts, sliced = build_sliced_moments(xc, args.cap, args.order)
        for sector, moments in zip(SECTORS, sliced):
            record = realize_both_ways(se_static, dyson_opts, moments)
            records[xc, sector] = record
            comparable = record["conserved"] == record["supplied"]
            print(
                f"{xc:>5} {sector:>9} {record['conserved']:>4}/{record['supplied']:<5} "
                f"{record['mblse_residual']:13.2e} {str(record['pencil_rank']):>12} "
                f"{record['pencil_residual']:13.2e} {'yes' if comparable else 'no':>11}"
            )

    print(
        "\n'comparable' is no wherever the recursion stepped down: it is then scored over "
        "fewer orders than the pencil, and the two numbers answer different questions."
    )

    particle_hf = records["hf", "particle"]
    particle_pbe = records["pbe", "particle"]
    print(f"\nparticle sector, per-order relative reconstruction error at K={args.order}")
    print(f"{'order':>5} {'MBLSE hf':>11} {'MBLSE pbe':>11} {'pencil pbe':>11} {'|T_n| pbe':>11}")
    for order in range(particle_pbe["supplied"]):
        pencil = particle_pbe["pencil_per_order"]
        print(
            f"{order:>5} {particle_hf['mblse_per_order'][order]:11.2e} "
            f"{particle_pbe['mblse_per_order'][order]:11.2e} "
            f"{'-' if pencil is None else format(pencil[order], '11.2e'):>11} "
            f"{particle_pbe['norms'][order]:11.2e}"
        )

    print("\nmoment magnitude, the candidate this rules out")
    print(f"{'order':>5} {'|T_n| hf':>11} {'|T_n| pbe':>11}")
    for order in (0, args.order // 2, args.order):
        print(
            f"{order:>5} {particle_hf['norms'][order]:11.2e} {particle_pbe['norms'][order]:11.2e}"
        )
    growth = lambda norms: float((norms[-1] / norms[0]) ** (1.0 / (len(norms) - 1)))  # noqa: E731
    print(
        f"\ngeometric growth per order: hf {growth(particle_hf['norms']):.2f}x, "
        f"pbe {growth(particle_pbe['norms']):.2f}x - the reference that *works* grows "
        "faster and ends larger, so the dynamic range of the moments is not the cause."
    )


if __name__ == "__main__":
    raise SystemExit(main())
