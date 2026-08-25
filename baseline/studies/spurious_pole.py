"""What in the recursion produces the lithium-hydride residual blow-up.

`residual_attribution.py` established that the blow-up is neither the molecule nor the raw
monomial moments but the recursion, and named loss of orthogonality as the standard
candidate while recording that it was not measured. This measures it, and it is **not** the
cause. The cause is a single spurious pole.

Three candidates, tested in order:

1. **Loss of orthogonality.** Refuted. `force_orthogonality` changes the residual by nothing
   at all, and with it off - so the recursion's own order-zero coefficients are kept - the
   *healthy* Hartree-Fock run drifts from orthonormality **further** than the failing PBE one
   at every iteration through 9 (1.0e-09 against 7.4e-11).
2. **Conditioning of the inverted block.** Real but far too small on its own: the final
   block's condition number is 24 in the failing run against 4 in the healthy one, which
   cannot produce an eighteen-order error.
3. **A spurious pole.** This is it. The failing realization contains a pole at **301 Ha**,
   58x beyond a spectrum otherwise clustered below 5.14 - which is exactly where the
   independent pencil realization puts its own edge. Reconstructing moment `n` weights every
   pole by `e**n`, so an outlier that is invisible at `n = 0` dominates at `n = 19`: the
   reconstructed moment comes out at 1.9e+13 against an input of 2.5e+09, and that ratio is
   the residual.

The onset table ties the three together. The spurious pole, the collapse of the final
block's smallest singular value, and the residual all appear at the same order in the same
run, and never in the other.

What this does not establish is why that block goes near-singular at that order in this
reference and not the other. The correlation is exact across every order measured, and a
near-singular block is the natural source of a spurious eigenvalue once its inverse square
root is applied, but the study measures the coincidence rather than the mechanism.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`.

Run from the repository root so the intended tree is imported::

    python -m baseline.studies.spurious_pole
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
from dyson import MBLSE

import momentGW
from baseline.studies.residual_attribution import REFERENCES, SYSTEM, build_sliced_moments
from momentGW.gw import RESIDUAL_MAX

#: Orders the onset is looked for over. The failure appears at 19 and the sweep needs orders
#: on both sides of it to show that it is an onset rather than a level.
ORDERS = (13, 15, 17, 19, 21)

#: The sector the blow-up is in. The hole sector steps down long before this and is a
#: different question.
SECTOR = 1


def realize(se_static, dyson_opts, moments, **overrides):
    """Run the recursion and read back what it produced.

    Returns
    -------
    dict
        Pole spectrum, final-block conditioning and residual for one realization.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        solver = MBLSE(se_static, moments, **dict(dyson_opts, **overrides))
        solver.kernel()
        self_energy = solver.solve().get_self_energy()
    energies = np.sort(np.abs(np.asarray(self_energy.energies)))[::-1]
    last = max(solver.off_diagonal)
    singular = np.linalg.svd(np.asarray(solver.off_diagonal[last]), compute_uv=False)
    return {
        "solver": solver,
        "energies": energies,
        "smallest_singular": float(singular[-1]),
        "condition": float(singular[0] / singular[-1]),
        "residual": float(solver.moment_errors().max_relative_frobenius),
    }


def orthogonality_drift(solver, iteration):
    """Deviation of an order-zero recursion coefficient from orthonormality.

    Only meaningful with `force_orthogonality` off, which is what keeps the recursion's own
    value instead of overwriting it with the identity.
    """
    try:
        diagonal = np.asarray(solver.coefficients[iteration, iteration, 0])
    except KeyError:
        return None, None
    identity = np.eye(diagonal.shape[0])
    drift = float(np.linalg.norm(diagonal - identity) / np.linalg.norm(identity))
    off = None
    if iteration > 1:
        with contextlib.suppress(KeyError):
            off = float(
                np.linalg.norm(np.asarray(solver.coefficients[iteration, iteration - 1, 0]))
            )
    return drift, off


def main():
    """Run the three candidate tests and print their tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap", type=int, default=21, help="order the moments are built at")
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    built = {}
    for xc in REFERENCES:
        se_static, dyson_opts, sliced = build_sliced_moments(xc, args.cap, args.cap)
        built[xc] = (se_static, dyson_opts, np.asarray(sliced[SECTOR]))

    print(f"\n1. loss of orthogonality - refuted. {SYSTEM}, particle sector, K=19")
    print(f"{'ref':>5} {'force_orth':>11} {'residual':>12}")
    solvers = {}
    for xc in REFERENCES:
        se_static, dyson_opts, moments = built[xc]
        for force in (True, False):
            record = realize(se_static, dyson_opts, moments[:20], force_orthogonality=force)
            solvers[xc, force] = record["solver"]
            print(f"{xc:>5} {str(force):>11} {record['residual']:12.2e}")
    print("   the knob changes nothing, so it is not holding the recursion together")

    print("\n   drift of the order-zero coefficients, force_orthogonality off")
    print(
        f"{'iter':>5} {'hf diag-I':>12} {'hf offdiag':>12} {'pbe diag-I':>12} {'pbe offdiag':>12}"
    )
    for iteration in range(1, 12):
        cells = []
        for xc in REFERENCES:
            drift, off = orthogonality_drift(solvers[xc, False], iteration)
            cells += [drift, off]
        if all(value is None for value in cells):
            break
        print(f"{iteration:>5} " + " ".join(f"{0.0 if v is None else v:12.2e}" for v in cells))
    print("   the healthy reference drifts further for most of the recursion, so drift")
    print("   does not separate the two")

    print("\n2 and 3. onset: pole spectrum, final-block conditioning and residual together")
    print(
        f"{'ref':>5} {'K':>4} {'max|e|':>10} {'2nd |e|':>9} {'min sv':>10} "
        f"{'cond':>9} {'residual':>11} {'gate':>6}"
    )
    for xc in REFERENCES:
        se_static, dyson_opts, moments = built[xc]
        for order in ORDERS:
            if order + 1 > moments.shape[0]:
                continue
            record = realize(se_static, dyson_opts, moments[: order + 1])
            energies = record["energies"]
            gate = "pass" if record["residual"] <= RESIDUAL_MAX else "FAIL"
            print(
                f"{xc:>5} {order:>4} {energies[0]:10.2e} {energies[1]:9.2e} "
                f"{record['smallest_singular']:10.2e} {record['condition']:9.2e} "
                f"{record['residual']:11.2e} {gate:>6}"
            )
    print(
        "\n   the spurious pole, the collapse of the smallest singular value and the residual\n"
        "   appear at the same order in the same run, and never in the other"
    )


if __name__ == "__main__":
    raise SystemExit(main())
