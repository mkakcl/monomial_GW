"""Measure the separable assembly of the dRPA resolvent moments on real factorisations.

`momentGW.thc_rpa` claims three things, and this measures each on a THC factorisation of a
real molecule rather than on a synthetic vertex:

1. **The assembly is an identity.**  Against the *same* fitted scalar function of the
   gaps, the separable route must reproduce an explicit contraction over the
   particle-hole pairs to roundoff, because the binomial expansion of ``D^m`` is exact.
   Anything above that is a defect in the expansion or the blocking, not an approximation.
2. **One residual certifies every order.**  Against the true ``D^(m+1)/(D^2+s)``, the
   error must be the residual `momentGW.laplace` measured, flat in the moment order.  This
   is the property that replaces the pair-space partition every earlier attempt needed
   (`THC.md` sections 2 and 3), so it is the number that decides whether the design holds.
3. **The stream costs what it says.**  Assembling every ``Q^(m+1)(s_j)`` before the first
   pole solve is formally quadratic but carries a constant near a hundred; blocking over
   the second auxiliary index is what makes the footprint a budget rather than a shape.
   The two are compared here by peak resident memory in separate processes, since a
   high-water mark inside one process only ever rises.

The reference is an explicit ``sum_ia W_{P,ia} f(D_ia) W_{Q,ia}`` with
``W_{P,ia} = X_iP X_aP`` -- the quartic contraction the module exists not to do.  It costs
``naux^2 * nov`` per shift and order, which is minutes at benzene, so `--shifts` subsamples
the pole set; what it drops is printed rather than left implicit.

Factorisations are not built here, for the reason `baseline/studies/thc_rank.py` gives:
PyTHC needs its own Python 3.12 environment (`THC.md` section 4.2).  Build them there and
point this at the directory::

    python -m baseline.studies.thc_rank --emit-commands
    python -m baseline.studies.thc_rpa_assembly --dir /path/to/factorisations

A study, not part of the recorded baseline set: it is re-run when the claims it supports
are in question, not by `baseline.check`.  Run from the repository root so the intended
tree is imported, and read the printed `momentGW.__file__` before believing a comparison.
"""

import argparse
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

import momentGW
from baseline.studies.thc_rank import SYSTEMS, mean_field
from momentGW import eta0, laplace, thc_build, thc_rpa

#: Tolerance for the certified pole set, matching the production default.
ETA0_TOL = 1e-14

#: Tolerance for the separable fit.  It cannot be `ETA0_TOL`: `laplace.TOL_FLOOR` is 1e-12,
#: so the THC path needs its own and saying so is part of `THC.md` M3.
LAPLACE_TOL = 1e-10

#: Highest moment order, matching the largest the roadmap targets.
NMOM_MAX = 9

#: Fraction of physical memory above which the unblocked assembly is predicted rather
#: than run.  Measuring a footprint that pages tells you about the swap device, not about
#: the algorithm, and the prediction is the claim under test in any case.
MEMORY_FRACTION = 0.4


def physical_memory():
    """Get the machine's physical memory in bytes.

    Returns
    -------
    total : int
        Physical memory, or zero if the platform does not report it.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


class Case:
    """A system reduced to the objects the assembly needs.

    Parameters
    ----------
    system : str
        Key of `baseline.studies.thc_rank.SYSTEMS`.
    path : str
        Factorisation file, as written by `momentGW.thc_build`.
    laplace_tol : float, optional
        Requested separable residual.  Default value is `LAPLACE_TOL`.
    """

    def __init__(self, system, path, laplace_tol=LAPLACE_TOL):
        mf = mean_field(system)
        coll, self.cou = thc_build.read(path)

        occupied = mf.mo_occ > 0
        # The same rotation `thc.Integrals.transform` applies
        self.coll_occ = coll @ mf.mo_coeff[:, occupied]
        self.coll_vir = coll @ mf.mo_coeff[:, ~occupied]
        self.e_occ = mf.mo_energy[occupied]
        self.e_vir = mf.mo_energy[~occupied]

        self.system = system
        self.nao = mf.mol.nao_nr()
        self.naux = coll.shape[0]
        self.nocc, self.nvir = self.e_occ.size, self.e_vir.size
        self.d = (self.e_vir[None, :] - self.e_occ[:, None]).ravel()

        # The pole set a calculation actually sees.  The coupling bound is
        # 4 sum_ia d_ia ||V_(:,ia)||^2 with V = L^T W the vertex including the Cholesky
        # factor of Z, and sum_ia d_ia W_(P,ia) W_(Q,ia) is exactly the auxiliary-space
        # image of the gaps -- so the whole thing is sum_PQ Z_PQ [D_X^(1)]_PQ, which
        # costs naux^2 nmo rather than naux^2 nov and is exact rather than a bound on
        # a bound.
        mu = 0.5 * (self.e_occ.max() + self.e_vir.min())
        occ_0 = self.coll_occ @ self.coll_occ.T
        occ_1 = (self.coll_occ * (mu - self.e_occ)) @ self.coll_occ.T
        vir_0 = self.coll_vir @ self.coll_vir.T
        vir_1 = (self.coll_vir * (self.e_vir - mu)) @ self.coll_vir.T
        coupling = 4.0 * float(np.sum(self.cou * (occ_1 * vir_0 + occ_0 * vir_1)))
        lmin, lmax = eta0.certified_interval(self.d, coupling)
        self.shifts, self.weights, *_ = eta0.select_poles(lmin, lmax, ETA0_TOL)

        self.t, self.coeffs, self.diagnostics = laplace.build(self.d, self.shifts, laplace_tol)

    def vertex(self):
        """Form the pair-index vertex the reference contracts over.

        Returns
        -------
        w : numpy.ndarray
            ``W_{P,ia} = X_iP X_aP``, shape ``(naux, nov)``.  Quadratic in memory but
            with a pair index, which is exactly what the separable route avoids holding.
        """
        return (self.coll_occ[:, :, None] * self.coll_vir[:, None, :]).reshape(self.naux, -1)


def reference(w, f):
    """Contract a scalar function of the gaps against the pair-index vertex.

    Parameters
    ----------
    w : numpy.ndarray
        ``(naux, nov)`` vertex.
    f : numpy.ndarray
        ``(n_order, nov)`` values on the gaps.

    Returns
    -------
    q : numpy.ndarray
        ``(n_order, naux, naux)``.

    Notes
    -----
    Operations ``O(n_order * naux^2 * nov)``, memory ``O(n_order * naux * nov)``.  Quartic
    and deliberately so: it is the definition, kept here to check the cubic route against.
    """
    naux, nov = w.shape
    stack = (w[None] * f[:, None, :]).reshape(-1, nov)
    return (stack @ w.T).reshape(f.shape[0], naux, naux)


def accuracy(case, orders, n_shifts=None):
    """Measure the assembly against the quartic contraction, per moment order.

    Parameters
    ----------
    case : Case
        System.
    orders : sequence of int
        Moment orders.
    n_shifts : int, optional
        Number of poles to check, spread evenly through the set and always including both
        ends.  `None` checks every pole.

    Returns
    -------
    identity : numpy.ndarray
        Relative spectral error against the same fitted scalar, per order.
    resolvent : numpy.ndarray
        Relative spectral error against the true resolvent, per order.
    checked : numpy.ndarray
        Indices of the poles checked.
    """

    orders = np.atleast_1d(np.asarray(orders, dtype=int))
    total = case.shifts.size
    checked = (
        np.arange(total)
        if n_shifts is None or n_shifts >= total
        else np.unique(np.linspace(0, total - 1, n_shifts).round().astype(int))
    )

    q = thc_rpa.build_q(
        case.coll_occ, case.coll_vir, case.e_occ, case.e_vir, case.t, case.coeffs, orders
    )
    w = case.vertex()
    basis = np.exp(-np.outer(case.d, case.t))

    identity = np.zeros(orders.size)
    resolvent = np.zeros(orders.size)

    for j in checked:
        shift = case.shifts[j]
        fitted = basis @ case.coeffs[j]
        exact = case.d / (case.d**2 + shift)

        powers = np.stack([case.d**m for m in orders])
        for target, out in ((powers * fitted, identity), (powers * exact, resolvent)):
            ref = reference(w, target)
            for mi in range(orders.size):
                norm = np.linalg.norm(ref[mi], 2)
                out[mi] = max(out[mi], np.linalg.norm(q[j, mi] - ref[mi], 2) / norm)
            del ref

    return identity, resolvent, checked


def footprint(case, orders, blocked):
    """Assemble once and report the wall clock and the peak resident memory.

    Parameters
    ----------
    case : Case
        System.
    orders : sequence of int
        Moment orders.
    blocked : bool
        Whether to stream over blocks of the second auxiliary index, or to hold the whole
        set at once.

    Returns
    -------
    seconds : float
        Wall clock of the assembly.
    added_mb : float
        Rise in the process high-water mark of resident memory across the assembly, in
        MB.  A high-water mark only ever rises, so this is a *lower* bound on what the
        assembly held: zero would mean it stayed under whatever the mean field peaked at,
        not that it allocated nothing.  `predicted_mb` says what to expect.
    block : int
        Block width used.
    predicted_mb : float
        Analytic working set of the choice made, in MB.
    norm : float
        Frobenius norm of the pole-weighted sum, so the two paths can be compared on the
        object a solver would actually carry forward.
    """

    orders = np.atleast_1d(np.asarray(orders, dtype=int))
    args = (case.coll_occ, case.coll_vir, case.e_occ, case.e_vir, case.t, case.coeffs, orders)
    block = (
        thc_rpa.block_size(
            case.naux, case.t.size, case.shifts.size, orders.size, int(orders.max()) + 1
        )
        if blocked
        else case.naux
    )

    scale = 1024**2 if platform.system() == "Darwin" else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

    start = time.perf_counter()
    if blocked:
        # What a solver does: consume each block and keep only what it accumulates
        total = np.zeros((orders.size, case.naux, case.naux))
        for cols, chunk in thc_rpa.q_blocks(*args, block=block):
            total[:, :, cols] = np.tensordot(case.weights, chunk, axes=(0, 0))
    else:
        whole = thc_rpa.build_q(*args, block=case.naux)
        total = np.tensordot(case.weights, whole, axes=(0, 0))
    seconds = time.perf_counter() - start

    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    # The same expression for both: only the block width differs, which is the whole
    # claim.  The unblocked path is this at ``block = naux``, so it is larger by exactly
    # the number of blocks the stream would have taken.
    stream = (case.t.size + case.shifts.size) * orders.size + 2 * (int(orders.max()) + 1)
    predicted = 8 * case.naux * block * stream
    return seconds, after - before, block, predicted / 1024**2, float(np.linalg.norm(total))


def main():
    """Run the study."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=".", help="directory holding the factorisations")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="system keys; each needs <key>.h5 in --dir",
    )
    parser.add_argument("--nmom-max", type=int, default=NMOM_MAX)
    parser.add_argument("--laplace-tol", type=float, default=LAPLACE_TOL)
    parser.add_argument(
        "--shifts",
        type=int,
        default=None,
        help="poles to check against the quartic reference (default: every one)",
    )
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-footprint", action="store_true")
    parser.add_argument(
        "--footprint-only",
        choices=("blocked", "whole"),
        default=None,
        help="internal: run one assembly and print its peak memory, for the subprocess "
        "comparison, since a high-water mark inside one process only ever rises",
    )
    args = parser.parse_args()

    orders = list(range(0, args.nmom_max, 2))
    systems = args.systems or sorted(SYSTEMS)

    if args.footprint_only is not None:
        case = Case(systems[0], os.path.join(args.dir, f"{systems[0]}.h5"), args.laplace_tol)
        measured = footprint(case, orders, args.footprint_only == "blocked")
        seconds, added, block, predicted, norm = measured
        print(f"{seconds:.6f} {added:.3f} {block:d} {predicted:.3f} {norm:.12e}")
        return 0

    print(f"momentGW: {momentGW.__file__}")
    print(f"orders {orders}, eta0 tolerance {ETA0_TOL:g}, laplace tolerance {args.laplace_tol:g}")

    for system in systems:
        path = os.path.join(args.dir, f"{system}.h5")
        if not os.path.exists(path):
            print(f"\n=== {system}: missing {path} ===")
            continue

        case = Case(system, path, args.laplace_tol)
        print(f"\n=== {system} ===")
        print(
            f"nao {case.nao}, rank {case.naux} ({case.naux / case.nao:.2f} x nao), "
            f"nocc {case.nocc}, nvir {case.nvir}, nov {case.nocc * case.nvir}"
        )
        print(
            f"{case.shifts.size} poles over [{case.shifts.min():.3e}, {case.shifts.max():.3e}]; "
            f"{case.diagnostics['n_points']} exponents, residual "
            f"{case.diagnostics['residual']:.3e}, amplification "
            f"{case.diagnostics['amplification']:.2e}"
        )

        if not args.skip_accuracy:
            identity, resolvent, checked = accuracy(case, orders, args.shifts)
            if checked.size < case.shifts.size:
                print(
                    f"  reference: {checked.size} of {case.shifts.size} poles checked "
                    f"(indices {', '.join(str(int(i)) for i in checked)}); "
                    "the rest were not evaluated"
                )
            print("     m   vs same fit   vs resolvent")
            for mi, m in enumerate(orders):
                print(f"  {m:4d}   {identity[mi]:11.3e}   {resolvent[mi]:12.3e}")

        if not args.skip_footprint:
            print(
                "   assembly        time (s)   RSS rise (MB)   predicted (MB)   block   |sum w Q|"
            )
            results = {}
            stream = (case.t.size + case.shifts.size) * len(orders) + 2 * (max(orders) + 1)
            unblocked = 8 * case.naux**2 * stream
            total_memory = physical_memory()

            width = thc_rpa.block_size(
                case.naux, case.t.size, case.shifts.size, len(orders), max(orders) + 1
            )
            if width >= case.naux:
                print(
                    f"   the budget is not binding at rank {case.naux}: the whole "
                    f"assembly is {unblocked / 1024**2:.0f} MB, so the two rows below "
                    "run the same code and differ only in allocation order"
                )

            for mode in ("blocked", "whole"):
                if mode == "whole" and total_memory and unblocked > MEMORY_FRACTION * total_memory:
                    print(
                        f"   {mode:12s}   not run: its working set is "
                        f"{unblocked / 1024**3:.1f} GB against {total_memory / 1024**3:.0f} GB "
                        f"of memory, so the measurement would be of the swap device"
                    )
                    continue
                out = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "baseline.studies.thc_rpa_assembly",
                        "--dir",
                        args.dir,
                        "--systems",
                        system,
                        "--nmom-max",
                        str(args.nmom_max),
                        "--laplace-tol",
                        str(args.laplace_tol),
                        "--footprint-only",
                        mode,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if out.returncode != 0:
                    print(f"   {mode:12s}   failed: {out.stderr.strip().splitlines()[-1:]}")
                    continue
                seconds, added, block, predicted, norm = out.stdout.split()
                results[mode] = float(norm)
                print(
                    f"   {mode:12s}   {float(seconds):8.3f}   {float(added):13.1f}   "
                    f"{float(predicted):14.1f}   {int(block):5d}   {float(norm):.12e}"
                )
            if len(results) == 2:
                gap = abs(results["blocked"] - results["whole"]) / abs(results["whole"])
                print(f"   the two agree to {gap:.2e} relative")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
