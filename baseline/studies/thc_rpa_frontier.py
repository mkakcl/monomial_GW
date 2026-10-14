"""Measure the tensor-hypercontracted dRPA against the density-fitted one.

`momentGW.thc_rpa.dRPA` reaches the same density-density moments as `momentGW.rpa.dRPA`
by a route that never forms the particle-hole index.  Two errors separate the two, and the
whole point of measuring rather than asserting is that they are separable:

1. **The factorisation.** The hypercontracted vertex is not the density-fitted one, and at
   a finite THC rank it never will be.  This error falls with the rank, and `THC.md`
   section 4.3 measures what rank buys what.
2. **The separable resolvent.** `momentGW.laplace` represents ``D/(D^2+s)`` on a fitted
   exponential grid, at a residual it measures.  This error falls with
   `thc_laplace_tol` and has nothing to do with the rank.

If they were entangled -- if the solver had a floor of its own -- sweeping the rank would
stop improving at some point that moving the tolerance could not shift.  The two sweeps
here are run against the same reference so that the question is answered rather than
assumed: the rank sweep at fixed tolerance, and the tolerance sweep at fixed rank.

Factorisations are not built here, for the reason `baseline/studies/thc_rank.py` gives:
PyTHC needs its own Python 3.12 environment (`THC.md` section 4.2).  This reads the same
directory and the same filenames that study writes, so one build serves both::

    python -m baseline.studies.thc_rank --emit-commands --dir /path/to/factorisations
    python -m baseline.studies.thc_rpa_frontier --dir /path/to/factorisations

A study, not part of the recorded baseline set: it is re-run when the claims it supports
are in question, not by `baseline.check`.  Run from the repository root so the intended
tree is imported, and read the printed `momentGW.__file__` before believing a comparison.
"""

import argparse
import os
import time

import h5py

import momentGW
from baseline.studies.thc_rank import SWEEP, SYSTEMS, build_mol, mean_field, run_frontier, tag

#: Separable-fit tolerances swept at fixed rank.  The top of the range is deliberately
#: past the point where the fit is worse than any factorisation on offer, so that the
#: sweep shows where the solver error starts to matter rather than only that it does not.
LAPLACE_TOLS = (1e-11, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)

#: Threshold whose rank the tolerance sweep is run at -- the operating point of
#: `THC.md` section 4.3, around 10 x nao.
FIXED_THRESHOLD = 1e-7


def reference(mf, nmom_max):
    """Run both density-fitted references once.

    Parameters
    ----------
    mf : pyscf.dft.RKS
        Converged mean field.
    nmom_max : int
        Maximum moment order.

    Returns
    -------
    references : dict
        Frontier energies keyed by the hypercontracted polarizability they are the
        reference for.
    """

    return {
        "thc-drpa": run_frontier(mf, nmom_max, "drpa"),
        "thc-dtda": run_frontier(mf, nmom_max, "dtda"),
    }


def run(mf, nmom_max, polarizability, path, **opts):
    """Run one hypercontracted calculation and time it.

    Parameters
    ----------
    mf : pyscf.dft.RKS
        Converged mean field.
    nmom_max : int
        Maximum moment order.
    polarizability : str
        Either ``"thc-drpa"`` or ``"thc-dtda"``.
    path : str
        Factorisation file.
    **opts
        Extra solver options, e.g. `thc_laplace_tol`.

    Returns
    -------
    ip : float
        Ionisation potential, eV, or `None` if the run raised.
    ea : float
        Electron affinity, eV, or `None`.
    seconds : float
        Wall clock.
    error : str or None
        The exception name, if there was one.
    """

    start = time.perf_counter()
    try:
        ip, ea = run_frontier(mf, nmom_max, polarizability, thc_opts=dict(file_path=path), **opts)
        return ip, ea, time.perf_counter() - start, None
    except Exception as error:  # pragma: no cover - reported, not raised
        return None, None, time.perf_counter() - start, type(error).__name__


def main():
    """Run the study."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=".", help="directory holding the factorisations")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--nmom-max", type=int, default=3)
    parser.add_argument("--skip-tolerance-sweep", action="store_true")
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    print(f"nmom_max {args.nmom_max}, cholesky factorisations, thresholds {SWEEP['cholesky']}")

    for system in args.systems or sorted(SYSTEMS):
        mol = build_mol(system)
        nao = mol.nao_nr()
        mf = mean_field(system)
        refs = reference(mf, args.nmom_max)

        print(f"\n=== {system}, nao {nao} ===")
        for pol, (ip, ea) in refs.items():
            print(f"   reference for {pol}: IP {ip:.6f} eV, EA {ea:.6f} eV")

        print(
            "\n   threshold     rank  rank/nao |   dRPA dIP    dRPA dEA     s |"
            "   dTDA dIP    dTDA dEA"
        )
        for threshold in SWEEP["cholesky"]:
            path = os.path.join(args.dir, tag(system, "cholesky", threshold) + ".h5")
            if not os.path.exists(path):
                print(f"   {threshold:9g}   (missing {os.path.basename(path)})")
                continue

            cells = []
            rank = None
            for pol in ("thc-drpa", "thc-dtda"):
                ip, ea, seconds, error = run(mf, args.nmom_max, pol, path)
                if error is not None:
                    cells.append(f"  failed: {error:14s}")
                    continue
                ref_ip, ref_ea = refs[pol]
                cell = f"{1000 * (ip - ref_ip):10.2f}  {1000 * (ea - ref_ea):10.2f}"
                cells.append(f"{cell}  {seconds:4.1f}" if pol == "thc-drpa" else cell)

            with h5py.File(path) as handle:
                rank = handle["coulomb_matrix"].shape[1]

            print(f"   {threshold:9g} {rank:8d} {rank / nao:9.2f} | {cells[0]} | {cells[1]}")
        print("   (meV against each polarizability's own density-fitted reference)")

        if args.skip_tolerance_sweep:
            continue

        path = os.path.join(args.dir, tag(system, "cholesky", FIXED_THRESHOLD) + ".h5")
        if not os.path.exists(path):
            continue

        print(f"\n   separable tolerance at threshold {FIXED_THRESHOLD:g}:")
        print("   thc_laplace_tol     dIP (meV)    dEA (meV)        s")
        ref_ip, ref_ea = refs["thc-drpa"]
        for tol in LAPLACE_TOLS:
            ip, ea, seconds, error = run(mf, args.nmom_max, "thc-drpa", path, thc_laplace_tol=tol)
            if error is not None:
                print(f"   {tol:15g}     raised {error}")
                continue
            print(
                f"   {tol:15g}   {1000 * (ip - ref_ip):10.3f}   "
                f"{1000 * (ea - ref_ea):10.3f}   {seconds:6.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
