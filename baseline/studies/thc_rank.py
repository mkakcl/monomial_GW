"""Measure what a PyTHC factorisation costs and what it is worth.

Two questions decide whether a THC dRPA is worth building, and both are properties of the
factorisation rather than of the solver:

1. **What rank does each construction need to reach a given integral accuracy?**  The
   separable dRPA cost goes as the *square* of the rank
   (`baseline/studies/thc_separable_q.py`), and the in-tree factoriser the old work used
   reached 10.2 x `nmo` at C16 with the ratio still growing.  A method that holds 5 x `nmo`
   at the same accuracy moves the predicted crossover by a factor of four.
2. **What does the resulting `G0W0` frontier do?**  A rank is only worth quoting next to
   the quasiparticle error it leaves.

This script answers both for the factorisations it is given.  It does *not* build them:
PyTHC needs a Python 3.12 environment with `numba`, `cotengra`, `pytblis` and a
git-sourced `acccholesky`, which is deliberately not `mgw-monomial` (see `THC.md` §4.2).
Run `--emit-commands` to print the build lines, run those under the PyTHC environment, then
run this against the directory they wrote to::

    python -m baseline.studies.thc_rank --emit-commands
    python -m baseline.studies.thc_rank --dir /path/to/factorisations

If PyTHC *is* importable -- which it will be after `THC.md` M0.3 -- the factorisations are
built in process and `--dir` is only a cache.

`cholesky`, `snri` and `nnls` select their points by *randomly*-pivoted Cholesky, so their
rank and error move by a few per cent between builds of the same system at the same
threshold.  Read the ranks as a band, not as a value, and do not chase a difference of
that size between two of them.

A study, not part of the recorded baseline set: it is re-run when the claims it supports are
in question, not by `baseline.check`.  Run from the repository root so the intended tree is
imported, and read the printed `momentGW.__file__` before believing a comparison.
"""

import argparse
import os

import numpy as np
from pyscf import dft, gto

import momentGW
from baseline import frontier
from momentGW import logging, thc_build
from momentGW.gw import GW

#: Systems, as ``(atom, basis, auxbasis, xc)``.  The first four are the baseline set, so a
#: rank quoted here sits next to numbers `baseline.check` already carries; benzene is added
#: because the auxiliary and particle-hole dimensions only separate at that size.
SYSTEMS = {
    "water": (
        "O 0.000000 0.000000 0.000000; H 0.757100 0.000000 0.586100; H -0.757100 0.000000 0.586100",
        "ccpvdz",
        "weigend",
        "pbe",
    ),
    "lithium-hydride": (
        "Li 0.000000 0.000000 0.000000; H 0.000000 0.000000 1.594900",
        "ccpvdz",
        "weigend",
        "pbe",
    ),
    "ozone": (
        "O 0.000000 0.000000 0.000000; O 1.086900 0.000000 0.660000; O -1.086900 0.000000 0.660000",
        "ccpvdz",
        "weigend",
        "pbe",
    ),
    "benzene": (
        "C 0.000000 1.396000 0.000000; C 1.209000 0.698000 0.000000; "
        "C 1.209000 -0.698000 0.000000; C 0.000000 -1.396000 0.000000; "
        "C -1.209000 -0.698000 0.000000; C -1.209000 0.698000 0.000000; "
        "H 0.000000 2.479000 0.000000; H 2.147000 1.240000 0.000000; "
        "H 2.147000 -1.240000 0.000000; H 0.000000 -2.479000 0.000000; "
        "H -2.147000 -1.240000 0.000000; H -2.147000 1.240000 0.000000",
        "ccpvdz",
        "weigend",
        "pbe",
    ),
}

#: Methods swept, with the rank or accuracy settings tried for each.  `becke`, `nnls` and
#: the unpruned `aux-becke` are left out: the first two have no rank control, and the third
#: keeps the whole parent Becke grid, which measured at 61 to 69 x nao and still 2e-2 to
#: 9e-2 relative -- its pruned subclass `aux-becke-cd` is the usable one.
SWEEP = {
    "cholesky": (1e-3, 1e-5, 1e-6, 1e-7, 1e-8),
    "kmeans": (1.5, 2.5, 4.0, 6.0),
    "qrcp": (1e-4, 1e-5, 1e-6),
    "snri": (1e-3, 1e-5, 1e-6, 1e-7),
    "aux-becke-cd": (1e-8,),
}

#: Largest system for which the dense reference integral tensor is formed.  `nao**4` in
#: float64 is 1.9 GB at 148 orbitals, so this is a memory limit, not a policy.
MAX_NAO_FOR_DENSE_ERI = 90

#: The THC construction fits the *density-fitted* tensor, and momentGW's own RI path uses
#: the mean field's auxiliary basis -- so the same basis is used for both, and the error
#: quoted is what the factorisation adds on top of density fitting rather than the
#: auxiliary basis's own error, which is larger and would hide it.


def tag(system, method, rank):
    """Get the filename stem for one factorisation.

    Parameters
    ----------
    system : str
        Key of `SYSTEMS`.
    method : str
        Key of `thc_build.METHODS`.
    rank : float
        Rank or accuracy control.

    Returns
    -------
    stem : str
        Filename stem, without an extension.
    """
    return f"{system}_{method}_{rank:g}".replace(".", "p").replace("-", "m")


def build_mol(system):
    """Build the molecule for a system key.

    Parameters
    ----------
    system : str
        Key of `SYSTEMS`.

    Returns
    -------
    mol : pyscf.gto.Mole
        The molecule.
    """
    atom, basis, _, _ = SYSTEMS[system]
    return gto.M(atom=atom, basis=basis, verbose=0)


def mean_field(system):
    """Run the density-fitted reference mean field for a system.

    Parameters
    ----------
    system : str
        Key of `SYSTEMS`.

    Returns
    -------
    mf : pyscf.dft.RKS
        Converged mean field.
    """
    atom, basis, auxbasis, xc = SYSTEMS[system]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf = mf.density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"reference SCF for {system} did not converge")
    return mf


def emit_commands(directory, python="/path/to/pythc-env/bin/python"):
    """Print the build commands to run under a PyTHC environment.

    Parameters
    ----------
    directory : str
        Destination directory for the factorisations.
    python : str, optional
        Interpreter to name in the printed commands.
    """

    print(f"mkdir -p {directory}")
    for system, (atom, basis, auxbasis, _) in SYSTEMS.items():
        for method, ranks in SWEEP.items():
            for rank in ranks:
                out = os.path.join(directory, tag(system, method, rank) + ".h5")
                print(
                    f'{python} momentGW/thc_build.py --atom "{atom}" --basis {basis} '
                    f"--auxbasis {auxbasis} --method {method} --rank {rank:g} --out {out}"
                )


def load_or_build(path, mol, method, rank, auxbasis):
    """Read a factorisation from disk, building it first if PyTHC is available.

    Parameters
    ----------
    path : str
        File to read or write.
    mol : pyscf.gto.Mole
        Molecule.
    method : str
        Key of `thc_build.METHODS`.
    rank : float
        Rank or accuracy control.
    auxbasis : str
        Auxiliary basis for the THC construction.

    Returns
    -------
    coll : numpy.ndarray or None
        Collocation matrix, or `None` if the file is missing and PyTHC is unavailable.
    cou : numpy.ndarray or None
        Coulomb matrix, or `None` in the same case.
    """

    if os.path.exists(path):
        return thc_build.read(path)

    try:
        coll, cou = thc_build.build(mol, method=method, rank=rank, auxbasis=auxbasis)
    except ImportError:
        return None, None

    thc_build.write(path, coll, cou)
    return coll, cou


def run_frontier(mf, nmom_max, polarizability, thc_opts=None, **opts):
    """Get the frontier quasiparticle energies from one `G0W0` run.

    Reads the spectrum through `baseline.frontier`, the same aufbau-with-weights route
    `baseline.run` records, rather than by indexing the pole list -- a THC spectrum can
    carry low-weight satellites either side of the gap, and picking the outermost pole
    would compare those instead of the quasiparticles.

    Parameters
    ----------
    mf : pyscf.dft.RKS
        Converged mean field.
    nmom_max : int
        Maximum moment order.
    polarizability : str
        momentGW polarizability keyword.
    thc_opts : dict, optional
        THC options, for the THC polarizabilities.
    **opts
        Any other solver options to set on the `GW` object, e.g. `thc_laplace_tol`.

    Returns
    -------
    ip : float
        Ionisation potential in eV.
    ea : float
        Electron affinity in eV.
    """

    gw = GW(mf)
    gw.polarizability = polarizability
    if thc_opts is not None:
        gw.thc_opts = thc_opts
    for key, value in opts.items():
        setattr(gw, key, value)
    with logging.with_silent():
        _, gf, _, _ = gw.kernel(nmom_max=nmom_max)

    results = frontier.readouts(
        gf.energies,
        gf.couplings,
        mf.mol.nelectron,
        mo_energy=mf.mo_energy,
        mo_occ=mf.mo_occ,
    )

    return results["ip_ev"], results["ea_ev"]


def main(argv=None):
    """Run the sweep and print the table.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector.  `None` reads `sys.argv`.

    Returns
    -------
    status : int
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="thc_factorisations", help="where factorisations live")
    parser.add_argument("--emit-commands", action="store_true", help="print build lines and stop")
    parser.add_argument("--nmom-max", type=int, default=3)
    parser.add_argument("--systems", nargs="*", default=None, help="subset of the system set")
    parser.add_argument(
        "--skip-gw", action="store_true", help="report ranks and integral errors only"
    )
    args = parser.parse_args(argv)

    if args.emit_commands:
        emit_commands(args.dir)
        return 0

    print(f"momentGW: {momentGW.__file__}")
    os.makedirs(args.dir, exist_ok=True)

    for system in args.systems or SYSTEMS:
        mol = build_mol(system)
        nao = mol.nao_nr()
        _, basis, _, _ = SYSTEMS[system]
        auxbasis = SYSTEMS[system][2]

        reference = None
        if nao <= MAX_NAO_FOR_DENSE_ERI:
            reference = thc_build.density_fitted_eri(mol, auxbasis)
            exact = mol.intor("int2e")
            df_frob = float(np.linalg.norm(reference - exact) / np.linalg.norm(exact))
            del exact
        else:
            df_frob = None

        mf = None
        ri_ip = ri_ea = None
        if not args.skip_gw:
            mf = mean_field(system)
            ri_ip, ri_ea = run_frontier(mf, args.nmom_max, "dtda")

        print(f"\n=== {system} / {basis}, nao {nao} ===")
        if df_frob is not None:
            print(f"density fitting alone: relative Frobenius {df_frob:.3e} against exact")
        if ri_ip is not None:
            print(
                f"RI dTDA frontier: IP {ri_ip:.6f} eV, EA {ri_ea:.6f} eV (nmom_max {args.nmom_max})"
            )
        print(
            "   method       rank    rank/nao   ERI max abs   ERI rel Frob    dIP (eV)    dEA (eV)"
        )

        for method, ranks in SWEEP.items():
            for rank in ranks:
                path = os.path.join(args.dir, tag(system, method, rank) + ".h5")
                coll, cou = load_or_build(path, mol, method, rank, auxbasis)
                if coll is None:
                    print(f"   {method:10s} {rank:8g}   (missing: {os.path.basename(path)})")
                    continue

                if reference is not None:
                    max_abs, rel_frob = thc_build.eri_error(coll, cou, reference)
                    err = f"{max_abs:11.3e}   {rel_frob:12.3e}"
                else:
                    err = f"{'-':>11s}   {'-':>12s}"

                shift = f"{'-':>10s}  {'-':>10s}"
                if not args.skip_gw:
                    try:
                        ip, ea = run_frontier(
                            mf, args.nmom_max, "thc-dtda", thc_opts=dict(file_path=path)
                        )
                        shift = f"{ip - ri_ip:10.4f}  {ea - ri_ea:10.4f}"
                    except Exception as error:  # pragma: no cover - reported, not raised
                        shift = f"failed: {type(error).__name__}"

                print(
                    f"   {method:10s} {coll.shape[0]:8d}   {coll.shape[0] / nao:8.2f}   "
                    f"{err}   {shift}"
                )

        del reference

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
