"""Test whether affine renormalisation changes what the MBLSE recursion does.

[`HANKEL_PENCIL.md`](../../HANKEL_PENCIL.md) §6.A proposes putting an exact
affine renormalisation in front of the recursion, and ROADMAP 3.2 carries the
same two items ("affinely center and scale the hole and particle spectral
sectors separately", "transform raw monomial moments to the scaled basis before
realization and transform poles back afterward").  Both rest on an assumption
that has never been tested.

The assumption is doubtful on its face, and §3.2 says why: **`MBLSE` runs a
block Lanczos recurrence and never forms a Hankel matrix.**  §3 of
`HANKEL_PENCIL.md` measures `cond(H0)` improving by 13.7x to 2.3e6x under the
renormalisation, but `H0` is a matrix the production path does not build.  What
the recursion actually does is take `matrix_power(off_diagonal_squared, -0.5)`
at every cycle (`mblse.py:300-315`), and whether *that* is helped by moving the
support is a separate question with a separate answer.

So this measures the only thing that decides §6.A: does the renormalisation
move the recursion's stall?  The sharpest test case is the lithium-hydride hole
sector, which `pencil_vs_mblse.py` found stalling at 6 conserved moments from
`nmom_max = 7` onward, drifting to 7.0e-6 reconstruction error while the pencil
holds all 12 to 1.4e-11.

Three centre/scale choices are compared against no transform at all:

- `trace`, from the first three moments only -- the estimator that would be
  available inside a solver, where the support is exactly what is unknown;
- `support`, from the true pole range -- unavailable in practice, included as
  the best the transform could possibly do;
- `centre-only`, the binomial shift with no scaling -- what `shift_moments`
  already implements in dyson, to separate the shift's contribution from the
  scale's.

Poles are mapped back exactly.  The transform induces `J -> s J + mu I` on the
block-tridiagonal, a similarity, which leaves eigenvectors and therefore the
couplings untouched; only the energies move.  `_self_energy` reproduces the
standard `solve().get_self_energy()` route when `centre = 0` and `scale = 1`,
and the `identity` row of the report is that check rather than a claim.

Reconstruction error is always measured against the **original, untransformed**
moments, so the four routes are scored on the same target.

A study, not part of the recorded baseline set: it is re-run when the claim it
supports is in question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the
printed `momentGW.__file__` before believing a comparison::

    python -m baseline.studies.affine_recursion
    python -m baseline.studies.affine_recursion --systems lih_hf --orders 7 11
"""

import argparse
import contextlib
import io

import numpy as np
from dyson import MBLSE, Lehmann
from dyson import util as dyson_util

import momentGW
from baseline.studies.hankel_pencil import affine_moments, moment_support_estimate
from baseline.studies.moment_noise import SYSTEMS
from baseline.studies.pencil_vs_mblse import (
    build_moments,
    pencil_self_energy,
    reconstruction_error,
)

#: Orders to test.  The lithium-hydride stall appears at 7 and persists.
DEFAULT_ORDERS = (5, 7, 9, 11)


def _self_energy(solver, centre, scale):
    """Build the self-energy from a solved recurrence, undoing an affine transform.

    Mirrors the route `gw._radau_self_energy` uses minus the Radau pin: assemble the
    block tridiagonal from the recurrence coefficients, drop the physical block, and
    read energies and couplings off it.  Undoing the transform is `J -> s J + mu I`,
    a similarity, so it is applied to the energies alone -- the eigenvectors, and
    therefore the couplings, are unchanged by it.

    Parameters
    ----------
    solver : dyson.MBLSE
        Solver, after `kernel` has been called.
    centre : float
        The shift the moments were transformed by.
    scale : float
        The scale the moments were transformed by.

    Returns
    -------
    dyson.Lehmann
        The self-energy in original coordinates.
    """
    iteration = solver.max_cycle if solver.max_cycle_achieved is None else solver.max_cycle_achieved
    nphys = solver.nphys
    jacobi = dyson_util.build_block_tridiagonal(
        [solver.on_diagonal[i] for i in range(iteration + 2)],
        [solver.off_diagonal[i] for i in range(iteration + 1)],
        None,
    )[nphys:, nphys:]
    energies, rotated = dyson_util.eig_lr(jacobi, hermitian=True)
    couplings = np.atleast_2d(solver.off_diagonal[0]) @ rotated[0][:nphys]
    return Lehmann(scale * energies + centre, couplings)


def run_recursion(se_static, moments, options, centre=0.0, scale=1.0):
    """Run the recursion on transformed moments and report what it achieved.

    Parameters
    ----------
    se_static : numpy.ndarray
        Static part of the self-energy.
    moments : numpy.ndarray
        The original, untransformed moments.
    options : dict
        Options for `dyson.MBLSE`.
    centre : float, optional
        Shift to transform the moments by.  Default is no shift.
    scale : float, optional
        Scale to transform the moments by.  Default is no scaling.

    Returns
    -------
    dict
        The conserved order, requested order, pole count, and the reconstruction
        error against the *original* moments.
    """
    transformed = moments
    if centre != 0.0 or scale != 1.0:
        transformed = affine_moments(moments, centre, scale)

    with contextlib.redirect_stdout(io.StringIO()):
        solver = MBLSE(se_static, np.array(transformed), **options)
        solver.kernel()
        self_energy = _self_energy(solver, centre, scale)

    achieved = solver.max_cycle if solver.max_cycle_achieved is None else solver.max_cycle_achieved
    return {
        "conserved": solver.nmom_conserved(achieved),
        "requested": solver.nmom_conserved(solver.max_cycle),
        "cycles": (achieved, solver.max_cycle),
        "poles": self_energy.naux,
        "error": np.max(reconstruction_error(self_energy, moments)),
    }


def variants(moments, energies=None):
    """Build the centre/scale choices to compare.

    Parameters
    ----------
    moments : numpy.ndarray
        The moments, for the trace-based estimator.
    energies : numpy.ndarray, optional
        True pole energies, for the ideal transform.  Omitted if unavailable.

    Returns
    -------
    dict
        Variant name mapped to its centre and scale.
    """
    trace_centre, trace_scale = moment_support_estimate(moments)
    choices = {
        "none": (0.0, 1.0),
        "centre-only": (trace_centre, 1.0),
        "trace": (trace_centre, trace_scale),
    }
    if energies is not None and energies.size:
        low, high = float(energies.min()), float(energies.max())
        choices["support"] = (0.5 * (low + high), max(0.5 * (high - low), 1e-30))
    return choices


def run_system(name, orders):
    """Compare the transforms for one system across a set of moment orders.

    Parameters
    ----------
    name : str
        System name.
    orders : sequence of int
        Moment orders to test.
    """
    print(f"\n== {name} ==")
    for nmom_max in orders:
        gw, _, se_static, hole, particle = build_moments(name, nmom_max)
        options = dict(gw.dyson_opts, calculate_errors=False)

        print(f"\n  nmom_max = {nmom_max}")
        for label, moments in (("hole", hole), ("particle", particle)):
            print(f"    {label}")
            # The `support` variant needs the pole range, which is only known after a
            # realization.  Taking it from the pencil rather than from the recursion
            # matters here: on a stalled sector the recursion's own poles span a
            # truncated support, so using them would hand the ideal transform a range
            # narrowed by the very failure it is being asked to fix.
            try:
                energies = pencil_self_energy(moments)[0].energies
            except np.linalg.LinAlgError:
                energies = None
            for variant, (centre, scale) in variants(moments, energies).items():
                try:
                    record = run_recursion(se_static, moments, options, centre, scale)
                except Exception as error:  # noqa: BLE001 - the failure mode is the result
                    print(f"      {variant:12s} FAILED ({type(error).__name__})")
                    continue
                stalled = "" if record["conserved"] >= record["requested"] else "  STALLED"
                print(
                    f"      {variant:12s} mu={centre:+9.3f} s={scale:8.3f}  "
                    f"conserved {record['conserved']:2d}/{record['requested']:2d}  "
                    f"cycles {record['cycles'][0]}/{record['cycles'][1]}  "
                    f"poles {record['poles']:3d}  recon {record['error']:.2e}{stalled}"
                )


def main():
    """Run the requested comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--systems",
        nargs="+",
        default=sorted(SYSTEMS),
        choices=sorted(SYSTEMS),
        help="systems to test",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=list(DEFAULT_ORDERS),
        help="moment orders to test",
    )
    args = parser.parse_args()

    print(f"momentGW imported from: {momentGW.__file__}")
    for name in args.systems:
        run_system(name, args.orders)


if __name__ == "__main__":
    main()
