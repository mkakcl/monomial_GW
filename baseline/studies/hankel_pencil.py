"""Measure the one-shot block Hankel pencil as an alternative to the MBLSE recursion.

The Cayley project (`mkakcl/cayley-gw`) carries two realization backends.  Its
`taylor` backend is an operator-Schur recursion that repeatedly applies inverse
defect factors; its `toeplitz-qr` backend replaces the whole recursion with one
block Toeplitz Gram matrix, one Hermitian eigendecomposition and one orthogonal
Procrustes step, and survives to roughly three times the moment order as a
result.

The trait `toeplitz-qr` leverages is a Gram identity.  For a *unitary* shift the
Krylov Gram block depends only on the index difference, `<U^i B*, U^j B*> =
C_{j-i}`, so the Gram is block Toeplitz and is written down directly from the
moments.  momentGW has the same identity with a different index law: the
self-energy moments are those of a *self-adjoint* operator, so `<H^i B*, H^j B*>
= T_{i+j}` and the Gram is block **Hankel**.  The one-shot analogue is therefore
the symmetric-definite pencil

    H0 = [T_{i+j}],  H1 = [T_{i+j+1}],   i, j = 0 ... m

whose eigenvalues are the poles and whose eigenvectors give the couplings --
no recursion and no repeated inverse square roots, against
`MBLSE._recurrence_iteration_hermitian` (`mblse.py:300-315`), which applies
`matrix_power(off_diagonal_squared, -0.5)` on both sides at every cycle.

This study measures four things, in this order:

1. `conditioning` -- `cond(H0)` over the recorded baseline set, raw against an
   exact affine renormalisation of the moment sequence.
2. `deflation` -- whether `H0` rank-deflates at the pole count, which is what
   makes the singular Gram usable rather than fatal, and whether the deflated
   pencil recovers the nodes.
3. `necessity` -- whether the affine renormalisation is required for the rank
   cut to be identifiable at all.
4. `noise` -- how the deflated pencil degrades under perturbed moments, which
   is what decides whether it can carry production work.

Findings, measured 2026-08-12 on this machine, are written up in
[`HANKEL_PENCIL.md`](../../HANKEL_PENCIL.md).  In short: the route is exact and
deflates cleanly, the affine renormalisation is not optional but is free, and
the noise margin is about seven decades narrower than the circle version --
so this is a diagnostic and a second opinion, not a replacement.

Sections 2 to 4 synthesise moments from a recorded Lehmann representation, so
their moments are *exact* by construction.  That is deliberate -- it isolates
the algorithm from quadrature error -- but it means the `noise` section is a
proxy for real moment error, not a measurement of it.

A study, not part of the recorded baseline set: it is re-run when the claim it
supports is in question, not by `baseline.check`.

`baseline/arrays` is gitignored and reproducible, so a fresh worktree has none;
produce them with `python -m baseline.run` or point `--arrays` at a tree that
already has them.

Run from the repository root::

    python -m baseline.studies.hankel_pencil
    python -m baseline.studies.hankel_pencil --section deflation
    python -m baseline.studies.hankel_pencil --arrays ../../../baseline/arrays
"""

import argparse
import glob
import math
import os

import numpy as np

HARTREE2EV = 27.211386245988

# The case whose Lehmann representation drives sections 2-4.  Any recorded case
# works; this one is small enough to reach order 41 in seconds and large enough
# that the pole count sits well inside the Gram dimension.
LEHMANN_CASE = "water_hf_nmom7_ia.npz"

ARRAYS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arrays")


def _arrays(directory, pattern):
    """Resolve a glob under the arrays directory, failing with the way to produce it.

    Parameters
    ----------
    directory : str
        Directory of baseline `.npz` archives.
    pattern : str
        Glob relative to that directory.

    Returns
    -------
    list of str
        Matching paths, sorted.
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise SystemExit(
            f"no baseline arrays matching {pattern!r} under {directory}.\n"
            "They are gitignored and reproducible: run `python -m baseline.run`, "
            "or pass --arrays pointing at a tree that already has them."
        )
    return paths


def block_hankel(moments, order, offset=0):
    """Build the block Hankel matrix ``[T_{i+j+offset}]`` for ``i, j = 0 ... order``.

    Parameters
    ----------
    moments : numpy.ndarray
        Moments indexed by order along the leading axis.
    order : int
        The block order `m`; the result is `(order + 1) * nphys` square.
    offset : int, optional
        Index shift, so `offset=1` gives the shifted pencil partner `H1`.

    Returns
    -------
    numpy.ndarray
        The block Hankel matrix.
    """
    nphys = moments.shape[1]
    size = (order + 1) * nphys
    result = np.empty((size, size), dtype=moments.dtype)
    for i in range(order + 1):
        rows = slice(i * nphys, (i + 1) * nphys)
        for j in range(order + 1):
            result[rows, j * nphys : (j + 1) * nphys] = moments[i + j + offset]
    return result


def affine_moments(moments, centre, scale):
    r"""Map a monomial moment sequence onto shifted, scaled coordinates.

    Applies the exact transform induced by :math:`x \to (x - \mu) / s`,

    .. math::
        \tilde{T}^{n} = s^{-n} \sum_{k=0}^{n} \binom{n}{k} (-\mu)^{n-k} T^{k},

    which needs only the moments themselves.  The shift is the binomial
    transform already implemented as `dyson.solvers.static._mbl.shift_moments`;
    the scale is the part that is missing there.  Both are undone exactly on the
    recovered block-tridiagonal matrix as `J -> s J + mu I`, a similarity, so
    nothing is approximated by using them.

    Parameters
    ----------
    moments : numpy.ndarray
        Moments indexed by order along the leading axis.
    centre : float
        The shift :math:`\mu`.
    scale : float
        The scale :math:`s`.

    Returns
    -------
    numpy.ndarray
        The transformed moments.
    """
    result = np.zeros_like(moments)
    for n in range(moments.shape[0]):
        for k in range(n + 1):
            result[n] += math.comb(n, k) * (-centre) ** (n - k) * moments[k]
        result[n] /= scale**n
    return result


def moment_support_estimate(moments):
    """Estimate a centre and scale for a moment sequence from its first three moments.

    Uses the trace Rayleigh quotients, which give the mean and standard deviation
    of the scalar measure `tr dSigma`.  This needs no knowledge of the poles and
    so is available in the solver, where the support is exactly what is unknown.

    Parameters
    ----------
    moments : numpy.ndarray
        Moments indexed by order along the leading axis; at least three.

    Returns
    -------
    tuple of float
        The centre and scale.
    """
    zeroth = np.trace(moments[0])
    mean = np.trace(moments[1]) / zeroth
    variance = np.trace(moments[2]) / zeroth - mean**2
    return mean, math.sqrt(max(variance, 1e-30))


def load_sectors(path):
    """Split a recorded Lehmann self-energy into hole and particle sectors.

    Parameters
    ----------
    path : str
        Path to a `baseline/arrays` archive.

    Returns
    -------
    dict
        Sector name mapped to its pole energies and couplings.
    """
    archive = np.load(path)
    energies = archive["se_energies"]
    couplings = archive["se_couplings"]
    gf_energies = archive["gf_energies"]
    chempot = 0.5 * (gf_energies[gf_energies < 0].max() + gf_energies[gf_energies >= 0].min())
    return {
        "hole": (energies[energies < chempot], couplings[:, energies < chempot]),
        "particle": (energies[energies >= chempot], couplings[:, energies >= chempot]),
    }


def moments_from_poles(energies, couplings, count):
    """Build exact moments from a Lehmann representation.

    Parameters
    ----------
    energies : numpy.ndarray
        Pole energies.
    couplings : numpy.ndarray
        Couplings, physical index first.
    count : int
        Number of moments to build.

    Returns
    -------
    numpy.ndarray
        Moments indexed by order along the leading axis.
    """
    return np.array([(couplings * energies**n) @ couplings.T for n in range(count)])


def deflated_pencil(moments, order, tol=1e-13):
    """Solve the deflated symmetric-definite pencil ``(H1, H0)``.

    `H0` is positive semidefinite by construction and becomes singular once the
    Gram dimension passes the number of poles.  That is not a failure: it is the
    measure reporting its own finite support, and the null space is discarded
    rather than regularised, exactly as `toeplitz-qr` deflates its Toeplitz Gram.

    Parameters
    ----------
    moments : numpy.ndarray
        Moments indexed by order along the leading axis.
    order : int
        The block order `m`.
    tol : float, optional
        Relative eigenvalue floor for the retained support.

    Returns
    -------
    tuple
        The sorted nodes and the retained rank.
    """
    gram = block_hankel(moments, order)
    shifted = block_hankel(moments, order, offset=1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > eigenvalues.max() * tol
    root = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    projected = root.T @ shifted @ root
    nodes = np.linalg.eigvalsh(0.5 * (projected + projected.T))
    return np.sort(nodes), int(keep.sum())


def section_conditioning(arrays_dir):
    """Report `cond(H0)` across the baseline set, raw against affine coordinates.

    Parameters
    ----------
    arrays_dir : str
        Directory of baseline `.npz` archives.
    """
    print("\n=== 1. Conditioning of the block Hankel Gram, recorded baseline set ===")
    print("Affine centre and scale from the first three moments only -- no pole knowledge.\n")
    print(
        f"{'case':34s} {'sector':9s} {'K':>3s} {'cond raw':>11s} {'cond affine':>12s} {'gain':>11s}"
    )
    print("-" * 86)
    gains = []
    for path in _arrays(arrays_dir, "*nmom7*"):
        name = os.path.basename(path).replace("_nmom7", "").replace(".npz", "")
        archive = np.load(path)
        for sector in ("hole", "particle"):
            moments = archive[f"se_moments_{sector}"]
            order = (moments.shape[0] - 1) // 2
            centre, scale = moment_support_estimate(moments)
            raw = np.linalg.cond(block_hankel(moments, order))
            affine = np.linalg.cond(block_hankel(affine_moments(moments, centre, scale), order))
            gains.append(raw / affine)
            print(
                f"{name:34s} {sector:9s} {2 * order + 1:3d} "
                f"{raw:11.2e} {affine:12.2e} {raw / affine:10.1f}x"
            )
    print(
        f"\nGain over {len(gains)} sectors: min {min(gains):.1f}x, median "
        f"{np.median(gains):.1f}x, max {max(gains):.1f}x"
    )


def section_deflation(arrays_dir):
    """Report rank deflation and node recovery for the deflated pencil.

    Parameters
    ----------
    arrays_dir : str
        Directory of baseline `.npz` archives.
    """
    print("\n=== 2. Rank deflation and node recovery ===")
    print(f"Moments synthesised exactly from {LEHMANN_CASE}.")
    print("Affine map onto the true support, which is the best the real line allows.\n")
    for sector, (energies, couplings) in load_sectors(_arrays(arrays_dir, LEHMANN_CASE)[0]).items():
        low, high = energies.min(), energies.max()
        centre, scale = 0.5 * (low + high), 0.5 * (high - low)
        reduced = (energies - centre) / scale
        moments = moments_from_poles(reduced, couplings, 84)
        npoles = energies.size
        print(f"-- {sector}: {npoles} poles, support [{low:.3f}, {high:.3f}] Ha")
        print(
            f"   {'K':>3s} {'dim':>5s} {'rank':>5s} {'cond':>10s} "
            f"{'sv[n-1]/sv[0]':>14s} {'sv[n]/sv[0]':>12s} {'node err (Ha)':>14s}"
        )
        for order in (1, 3, 6, 10, 16, 20):
            gram = block_hankel(moments, order)
            values = np.linalg.svd(gram, compute_uv=False)
            rank = int((values > values[0] * 1e-14).sum())
            last = values[npoles - 1] / values[0] if values.size >= npoles else float("nan")
            first_out = values[npoles] / values[0] if values.size > npoles else float("nan")
            nodes, kept = deflated_pencil(moments, order)
            error = scale * max(np.min(np.abs(np.sort(reduced) - node)) for node in nodes)
            # Below rank saturation the rule has fewer nodes than the measure has
            # poles, so it is an under-resolved quadrature rule by construction and
            # the distance to the nearest pole is not a measure of its accuracy.
            note = "" if kept >= npoles else "  (under-resolved)"
            print(
                f"   {2 * order + 1:3d} {gram.shape[0]:5d} {rank:5d} "
                f"{values[0] / values[-1]:10.1e} {last:14.2e} {first_out:12.2e} "
                f"{error:14.2e}{note}"
            )
        print()


def section_necessity(arrays_dir):
    """Report whether the affine renormalisation is needed for an identifiable rank cut.

    Parameters
    ----------
    arrays_dir : str
        Directory of baseline `.npz` archives.
    """
    print("\n=== 3. Is the affine renormalisation essential? ===")
    print("The rank cut is what makes the singular Gram usable.  If it is not")
    print("identifiable, the whole route is unavailable.\n")
    print(
        f"{'sector':9s} {'K':>3s} {'coordinates':>20s} {'sv[n-1]/sv[0]':>14s} "
        f"{'sv[n]/sv[0]':>12s} {'gap (decades)':>14s}"
    )
    print("-" * 78)
    for sector, (energies, couplings) in load_sectors(_arrays(arrays_dir, LEHMANN_CASE)[0]).items():
        npoles = energies.size
        low, high = energies.min(), energies.max()
        centre, scale = 0.5 * (low + high), 0.5 * (high - low)
        variants = (
            ("raw monomial (Ha)", energies),
            ("affine to [-1, 1]", (energies - centre) / scale),
        )
        for label, reduced in variants:
            moments = moments_from_poles(reduced, couplings, 40)
            values = np.linalg.svd(block_hankel(moments, 6), compute_uv=False)
            last, first_out = values[npoles - 1] / values[0], values[npoles] / values[0]
            print(
                f"{sector:9s} {13:3d} {label:>20s} {last:14.2e} {first_out:12.2e} "
                f"{np.log10(last / first_out):14.1f}"
            )
    print()


def section_noise(arrays_dir, repeats=5, seed=0):
    """Report how the deflated pencil degrades under perturbed moments.

    Parameters
    ----------
    arrays_dir : str
        Directory of baseline `.npz` archives.
    repeats : int, optional
        Perturbed samples per cell; the median is reported.
    seed : int, optional
        Seed for the perturbations.
    """
    rng = np.random.default_rng(seed)
    print("\n=== 4. Noise robustness of the deflated pencil ===")
    print("Relative noise added entrywise to every moment, then re-symmetrised.")
    print("Median max node error in Ha over", repeats, "samples.\n")
    for sector, (energies, couplings) in load_sectors(_arrays(arrays_dir, LEHMANN_CASE)[0]).items():
        low, high = energies.min(), energies.max()
        centre, scale = 0.5 * (low + high), 0.5 * (high - low)
        reduced = (energies - centre) / scale
        exact = np.sort(reduced)
        clean = moments_from_poles(reduced, couplings, 80)
        orders = (3, 6, 10, 16)
        print(f"-- {sector}")
        print(f"   {'noise':>8s} " + " ".join(f"{'K=' + str(2 * m + 1):>12s}" for m in orders))
        for epsilon in (0.0, 1e-14, 1e-12, 1e-10, 1e-8):
            cells = []
            for order in orders:
                errors = []
                for _ in range(repeats if epsilon else 1):
                    moments = clean
                    if epsilon:
                        perturbation = rng.standard_normal(clean.shape)
                        moments = clean + epsilon * np.abs(clean).max() * perturbation
                        moments = 0.5 * (moments + moments.transpose(0, 2, 1))
                    try:
                        nodes, _ = deflated_pencil(moments, order)
                        errors.append(scale * max(np.min(np.abs(exact - node)) for node in nodes))
                    except np.linalg.LinAlgError:
                        errors.append(float("nan"))
                cells.append(f"{np.median(errors):12.2e}")
            print(f"   {epsilon:8.0e} " + " ".join(cells))
        print()


SECTIONS = {
    "conditioning": section_conditioning,
    "deflation": section_deflation,
    "necessity": section_necessity,
    "noise": section_noise,
}


def main():
    """Run the requested sections."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arrays",
        default=ARRAYS_DIR,
        help="directory of baseline .npz archives (default: baseline/arrays)",
    )
    parser.add_argument(
        "--section",
        choices=sorted(SECTIONS),
        action="append",
        help="Run only this section; repeatable.  Default is all four.",
    )
    args = parser.parse_args()
    arrays_dir = os.path.abspath(args.arrays)
    for name in args.section or ["conditioning", "deflation", "necessity", "noise"]:
        SECTIONS[name](arrays_dir)


if __name__ == "__main__":
    main()
