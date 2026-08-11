"""Compare the one-shot Hankel pencil against the MBLSE recursion on real moments.

[`HANKEL_PENCIL.md`](../../HANKEL_PENCIL.md) §6.B makes this the gate on the
pencil: *"head-to-head against `MBLSE` at realistic noise.  If the pencil loses
everywhere, stop here -- record the negative result and close it out."*  §5
lists it as the study's largest omission, because everything measured there
compares the pencil against exact poles rather than against the route momentGW
actually uses.

Both routes are given **the same moments** -- the real ones, built by the
production dRPA path, carrying the 1.4e-15 to 8.9e-15 error measured in §4.1.
`MBLSE` at `max_cycle = m` and the pencil at block order `m` both consume
`T_0 ... T_{2m+1}`, so neither is handed information the other lacks.

Two comparisons are reported, because they can disagree and the second is the
one that matters:

1. **Moment reconstruction.**  Each route yields a Lehmann self-energy; its
   moments are compared per order against the input.  This is the question
   "which route represents the data it was given more faithfully", and it is
   the only one with an unambiguous reference.
2. **Frontier quasiparticle energies.**  Both self-energies go through the same
   `FockLoop` and the same Dyson solve, so the difference is attributable to
   the realization alone.  A route can reconstruct moments well and still move
   the frontier, which is why this is not folded into the first.

The pencil is run in affine coordinates throughout (§3): without the
renormalisation its rank cut is not identifiable, so "the pencil" always means
the affine-renormalised pencil.  The centre and scale come from the first three
moments alone, not from the true support, so this is the estimator that would
actually be available in a solver rather than the idealised one §2 uses.

What a win here would and would not mean.  Reconstructing moments better is not
by itself a reason to switch: `MBLSE` produces a block-tridiagonal form the rest
of the code consumes, and the pencil does not.  §6.B's step 2 is a diagnostic,
not a backend, and this study is evidence for that step only.

A study, not part of the recorded baseline set: it is re-run when the claim it
supports is in question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the
printed `momentGW.__file__` before believing a comparison::

    python -m baseline.studies.pencil_vs_mblse
    python -m baseline.studies.pencil_vs_mblse --systems water_hf --orders 3 5 7
"""

import argparse
import contextlib
import io

import numpy as np
from dyson import MBLSE, Lehmann
from pyscf import dft, gto, scf

import momentGW
from baseline.studies.hankel_pencil import affine_moments, block_hankel, moment_support_estimate
from baseline.studies.moment_noise import SYSTEMS
from momentGW.fock import FockLoop
from momentGW.gw import GW, achieved_iteration, frontier_readout
from momentGW.rpa import dRPA

HARTREE2EV = 27.211386245988

#: Moment orders to compare.  `nmom_max` is odd throughout momentGW; `MBLSE` consumes
#: `2 * max_cycle + 2` moments, so order `n` and pencil block order `(n - 1) // 2` see
#: the same data.
DEFAULT_ORDERS = (3, 5, 7, 9, 11)


def pencil_self_energy(moments, tol=1e-13):
    """Realize a self-energy from its moments by the deflated block Hankel pencil.

    The realization is the one described in `HANKEL_PENCIL.md` §1 and §3: map the
    moment sequence onto affine coordinates, build the Gram `H0` and its shift
    partner `H1`, deflate `H0` onto its numerical support, and solve the resulting
    symmetric eigenproblem.  Nodes map back exactly under `e = s * lambda + mu`;
    couplings are unaffected by the affine map, since it moves the support and not
    the measure.

    Parameters
    ----------
    moments : numpy.ndarray
        Self-energy moments indexed by order, shape ``(2m + 2, nphys, nphys)``.
    tol : float, optional
        Relative eigenvalue floor for the retained support of `H0`.

    Returns
    -------
    tuple
        The self-energy as a `dyson.Lehmann`, and the retained rank.
    """
    nphys = moments.shape[1]
    order = (moments.shape[0] - 2) // 2

    centre, scale = moment_support_estimate(moments)
    reduced = affine_moments(moments, centre, scale)

    gram = block_hankel(reduced, order)
    shifted = block_hankel(reduced, order, offset=1)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > eigenvalues.max() * tol
    rank = int(keep.sum())

    # `root` is the positive square root of the Gram on its retained support, so
    # `root.T @ root = H0`; its leading `nphys` columns are the selector block.
    root = np.sqrt(eigenvalues[keep])[:, None] * eigenvectors[:, keep].T
    inverse = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])

    projected = inverse.T @ shifted @ inverse
    nodes, rotation = np.linalg.eigh(0.5 * (projected + projected.T))

    couplings = (rotation.T @ root[:, :nphys]).T
    return Lehmann(scale * nodes + centre, couplings), rank


def reconstruction_error(self_energy, reference):
    """Relative Frobenius error of a realization's moments against the input.

    Parameters
    ----------
    self_energy : dyson.Lehmann
        The realized self-energy.
    reference : numpy.ndarray
        The moments it was built from, indexed by order.

    Returns
    -------
    numpy.ndarray
        One relative error per order.
    """
    predicted = np.asarray(self_energy.moments(range(reference.shape[0])))
    errors = []
    for order in range(reference.shape[0]):
        difference = float(np.linalg.norm(np.ravel(predicted[order] - reference[order])))
        norm = float(np.linalg.norm(np.ravel(reference[order])))
        errors.append(difference / norm if norm > 0.0 else (0.0 if difference == 0.0 else np.inf))
    return np.array(errors)


def build_moments(name, nmom_max):
    """Build the real self-energy moments for one system at one order.

    Parameters
    ----------
    name : str
        System name in `moment_noise.SYSTEMS`.
    nmom_max : int
        Maximum moment order.

    Returns
    -------
    tuple
        The solver, the static self-energy, and the hole and particle moments.
    """
    atom, basis, xc = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    if xc == "hf":
        mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    else:
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    gw = GW(mf)
    gw.compression = ""
    with contextlib.redirect_stdout(io.StringIO()):
        integrals = gw.ao2mo()
        se_static = gw.build_se_static(integrals)
        rpa = dRPA(gw, nmom_max, integrals)
        hole, particle = rpa.build_se_moments(rpa.build_dd_moments())
    return gw, integrals, se_static, np.asarray(hole), np.asarray(particle)


def frontier_from_self_energy(gw, se_static, self_energy):
    """Read the frontier through the same Dyson solve both routes share.

    Parameters
    ----------
    gw : momentGW.gw.GW
        The solver.
    se_static : numpy.ndarray
        Static part of the self-energy.
    self_energy : dyson.Lehmann
        The realized self-energy, both sectors concatenated.

    Returns
    -------
    dict or None
        The frontier readout, or `None` if the Dyson solve failed.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fock_loop = FockLoop(gw, se=self_energy, **gw.fock_opts)
            gf, _ = fock_loop.solve_dyson(se_static, se=self_energy)
            return frontier_readout(gf)
    except (np.linalg.LinAlgError, ValueError):
        return None


def run_system(name, orders):
    """Compare both routes for one system across a set of moment orders.

    Parameters
    ----------
    name : str
        System name.
    orders : sequence of int
        Moment orders to compare.
    """
    print(f"\n== {name} ==")
    for nmom_max in orders:
        gw, _, se_static, hole, particle = build_moments(name, nmom_max)

        results = {}
        for label in ("mblse", "pencil"):
            sectors, ranks, errors, conserved = [], [], [], []
            failed = None
            for moments in (hole, particle):
                try:
                    if label == "mblse":
                        options = dict(gw.dyson_opts, calculate_errors=False)
                        with contextlib.redirect_stdout(io.StringIO()):
                            solver = MBLSE(se_static, np.array(moments), **options)
                            solver.kernel()
                            sector = solver.solve().get_self_energy()
                        ranks.append(sector.naux)
                        # The recursion reports its own stall, so a reconstruction error
                        # can be read against what the solver already admits rather than
                        # being presented as an undetected failure. `max_cycle_achieved`
                        # below `max_cycle` means it stopped early, and `nmom_conserved`
                        # is how many moments it actually kept.
                        conserved.append(solver.nmom_conserved(achieved_iteration(solver)))
                    else:
                        sector, rank = pencil_self_energy(moments)
                        ranks.append(rank)
                    sectors.append(sector)
                    errors.append(reconstruction_error(sector, moments))
                except Exception as error:  # noqa: BLE001 - the failure mode is the result
                    failed = type(error).__name__
                    break

            if failed is not None:
                results[label] = {"failed": failed}
                continue

            combined = sectors[0].copy()
            for sector in sectors[1:]:
                combined = combined.concatenate(sector)
            results[label] = {
                "hole": errors[0],
                "particle": errors[1],
                "ranks": ranks,
                "conserved": conserved,
                "requested": int(hole.shape[0]),
                "frontier": frontier_from_self_energy(gw, se_static, combined),
            }

        _report(nmom_max, results)


def _report(nmom_max, results):
    """Print one order's comparison.

    Parameters
    ----------
    nmom_max : int
        The moment order.
    results : dict
        Per-route results.
    """
    print(f"\n  nmom_max = {nmom_max}")
    for label in ("mblse", "pencil"):
        record = results[label]
        if "failed" in record:
            print(f"    {label:7s} FAILED ({record['failed']})")
            continue
        worst_hole = np.max(record["hole"])
        worst_particle = np.max(record["particle"])
        line = (
            f"    {label:7s} recon max: hole {worst_hole:.2e}  "
            f"particle {worst_particle:.2e}   poles {record['ranks']}"
        )
        if record["conserved"]:
            line += f"   conserved {record['conserved']}/{record['requested']}"
        print(line)

    usable = {k: v for k, v in results.items() if "failed" not in v and v["frontier"]}
    if len(usable) == 2:
        a, b = usable["mblse"]["frontier"], usable["pencil"]["frontier"]
        for key in ("homo", "lumo"):
            if key in a and key in b:
                delta = abs(a[key] - b[key]) * HARTREE2EV
                print(
                    f"    {key.upper():7s} mblse {a[key] * HARTREE2EV:+.6f} eV   "
                    f"pencil {b[key] * HARTREE2EV:+.6f} eV   diff {delta:.3e} eV"
                )
    else:
        for label in ("mblse", "pencil"):
            record = results[label]
            if "failed" not in record and not record["frontier"]:
                print(f"    {label:7s} frontier unavailable (Dyson solve failed)")


def main():
    """Run the requested comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--systems",
        nargs="+",
        default=sorted(SYSTEMS),
        choices=sorted(SYSTEMS),
        help="systems to compare",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=list(DEFAULT_ORDERS),
        help="moment orders to compare",
    )
    args = parser.parse_args()

    print(f"momentGW imported from: {momentGW.__file__}")
    for name in args.systems:
        run_system(name, args.orders)


if __name__ == "__main__":
    main()
