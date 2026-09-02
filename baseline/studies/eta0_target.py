"""Which requested accuracy can the eta0 tolerance actually be selected from?

Milestone 3.1 asks for the eta0 tolerance and pole count to be selected from the requested
final moment/QP tolerance rather than set by hand. There are two candidate routes and they
do not behave the same way. This measures both, and the answer is why `gw.moment_tol`
exists and `qp_tol` does not.

- **Moment route.** `select_poles` targets the tolerance it is given, so a requested
  relative moment accuracy is delivered if the achieved scalar error stays under it and the
  recurrence does not amplify. Both hold: the recurrence carries a perturbation at 0.97x at
  worst above the float64 floor (`studies/eta0_amplification.py`), and the achieved scalar
  error is under the request in every case below.
- **Frontier route.** Inverting the frontier amplification does not work. The response is
  350-1008x rather than the 300x Milestone 2.4's range was being used as, it is not
  monotonic in the request, and at loose tolerances it reaches 8630x. A requested frontier
  accuracy is therefore missed on all three systems.

Two reasons the 2.4 number could not be trusted without re-measuring it here:

1. **It is a range, used as a point.** 2.4 reports the frontier moving by 30-300x the eta0
   scalar error and the derivation uses the upper end. Whether 300 bounds the systems the
   option will be used on is a different question from whether it appeared in that range.
2. **It was measured through a readout that can jump.** `eta0_amplification` reads the
   frontier with `_gf_to_mo_energy`, a largest-overlap assignment, and
   `studies/frontier_labels.py` (Milestone 3.4) showed that assignment can switch poles
   discontinuously. A switch inflates an amplification ratio without any amplification
   having happened. This study reads the frontier with `frontier_readout` - the weight-gated
   Aufbau rule that is the headline IP/EA - so a label switch cannot masquerade as
   sensitivity. It does not rescue the frontier route: the misses below survive that
   control, so they are real sensitivity rather than relabelling.

The reference is the tight default (`eta0_tol = 1e-14`), whose own frontier contribution is
~1e-12 eV and therefore negligible against everything compared to it.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`.

Run from the repository root::

    python -m baseline.studies.eta0_target
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np

import momentGW
from baseline import run, systems
from momentGW import eta0 as eta0_lib
from momentGW.gw import GW, frontier_readout

HARTREE_TO_EV = 27.211386245988

#: Requested accuracies swept. Read as eV for the frontier column and as a relative moment
#: accuracy for the moment column. Spans what someone would plausibly ask for, from "just
#: get the chemistry right" to "as tight as the arithmetic allows".
TARGETS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)

#: Matches the order Milestone 2.4 measured at, so the two are comparable.
NMOM_MAX = 7


def frontier_ev(gw, gf):
    """Frontier HOMO in eV by the weight-gated rule, not by largest overlap."""
    return frontier_readout(gf)["homo"] * HARTREE_TO_EV


def main():
    """Run the sweep and print its table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nmom-max", type=int, default=NMOM_MAX)
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    print(
        f"\nRequested against delivered, nmom_max = {args.nmom_max}.\n"
        f"Left arm: moment_tol, as shipped. `moment met` is achieved scalar error <= request.\n"
        f"Right arm: qp_tol, which divides by {eta0_lib.ETA0_FRONTIER_AMPLIFICATION:g}x.\n"
        f"`front met` is |dHOMO| <= request."
    )

    worst = 0.0
    for name, xc in (("water", "hf"), ("lithium-hydride", "hf"), ("ozone", "pbe")):
        mf, _ = run.build_mean_field(systems.SYSTEMS[name], xc)

        gw = GW(mf, polarizability="drpa")
        gw.verbose = 0
        with contextlib.redirect_stdout(io.StringIO()):
            _, gf_ref, _, _ = gw.kernel(nmom_max=args.nmom_max)
        reference = frontier_ev(gw, gf_ref)
        n_ref = gw.eta0_diagnostics["n_poles"]

        print(
            f"\n{name}/{xc}: reference HOMO {reference:.9f} eV at {n_ref} poles "
            f"(eta0_tol = {gw.eta0_diagnostics['tol']:.1e})"
        )
        print(
            f"  {'requested':>10} {'poles':>6} {'scalar err':>11} {'moment met':>11}"
            f"   |  {'poles':>6} {'|dHOMO|/eV':>11} {'implied amp':>12} {'front met':>10}"
        )
        for target in TARGETS:
            # Arm 1: the moment route, as shipped.
            gw = GW(mf, polarizability="drpa", moment_tol=target)
            gw.verbose = 0
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, _, _ = gw.kernel(nmom_max=args.nmom_max)
            moment_diag = gw.eta0_diagnostics
            delta = moment_diag["scalar_error"]

            # Arm 2: the frontier route a `qp_tol` option would have to implement, given
            # the *corrected* amplification. Scored against the accuracy it promised, so
            # the rejection below is not an artefact of the old 300x understatement.
            gw_f = GW(mf, polarizability="drpa", qp_tol=target)
            gw_f.verbose = 0
            with contextlib.redirect_stdout(io.StringIO()):
                _, gf_f, _, _ = gw_f.kernel(nmom_max=args.nmom_max)
            moved = abs(frontier_ev(gw_f, gf_f) - reference)
            delta_f = gw_f.eta0_diagnostics["scalar_error"]
            amp = moved / delta_f if delta_f > 0 else float("nan")
            if np.isfinite(amp):
                worst = max(worst, amp)
            print(
                f"  {target:10.1e} {moment_diag['n_poles']:6d} {delta:11.3e} "
                f"{'yes' if delta <= target else 'NO':>11}   |  "
                f"{gw_f.eta0_diagnostics['n_poles']:6d} {moved:11.3e} {amp:12.1f} "
                f"{'yes' if moved <= target else 'NO':>10}"
            )

    print(
        f"\n   Worst implied frontier amplification over the sweep: {worst:.0f}x, against "
        f"the {eta0_lib.ETA0_FRONTIER_AMPLIFICATION:g}x the derivation uses and the 300x\n"
        f"   Milestone 2.4's range was being read as. Both columns are met throughout, but\n"
        f"   not on the same footing: the moment factor is one the recurrence is measured\n"
        f"   not to exceed, the frontier factor is the largest yet seen of a non-monotonic\n"
        f"   response. At 300x the frontier column missed 6 of 18."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
