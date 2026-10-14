r"""Build tensor-hypercontracted integrals with PyTHC.

`momentGW.thc.Integrals` can read a factorisation but not make one -- until now the only
route was an external code writing the HDF5 file it imports.  This module supplies the
missing half against `PyTHC <https://github.com/QuantumCorrelators/pythc>`_, the reference
implementation released with `arXiv:2608.17885
<https://arxiv.org/abs/2608.17885>`_.

**This module deliberately imports nothing from momentGW.**  PyTHC's dependency set is
large -- `numba`, `cotengra`, `pytblis`, `scikit-learn`, `matplotlib`, `psutil`, and a
git-sourced `acccholesky`, most of them imported at `pythc.lib` module scope -- and the
environment momentGW's baseline is recorded against does not carry it.  Keeping this file
free of momentGW imports means it runs as a script under a PyTHC environment without the
package `__init__` ever executing::

    /path/to/pythc-env/bin/python momentGW/thc_build.py \
        --atom "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692" \
        --basis cc-pvdz --method cholesky --out water.h5

and, in an environment that has both, as an ordinary import::

    from momentGW import thc_build
    coll, cou = thc_build.build(mol, method="kmeans", rank=5.0)

The file it writes is the layout `momentGW.thc.Integrals.import_thc_components` already
reads, so the two paths are interchangeable and the factorisation is a recordable artefact
rather than a step that has to be repeated.

Notes
-----
The factorisation is always built in PyTHC's ``'ao'`` mode.  momentGW needs `Lp`, `Lx`,
`Li` and `La` -- the full orbital range -- from a single collocation matrix, which the
``'ov'`` mode cannot supply: it fits only the occupied-virtual block and folds the
molecular orbital coefficients into `X`.
"""

if __name__ == "__main__":
    # Running a file puts *its own directory* on `sys.path`, and this one contains
    # `momentGW/logging.py`, which then shadows the standard library module that h5py and
    # PySCF import on the next line.  Drop it before any third-party import happens; as an
    # imported module the entry never exists and this block does not run.
    import os as _os
    import sys as _sys

    _package_dir = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path[:] = [
        entry for entry in _sys.path if _os.path.abspath(entry or _os.getcwd()) != _package_dir
    ]

import argparse

import h5py
import numpy as np

#: PyTHC classes reachable by name, with the keyword each uses to set its rank or
#: accuracy.  ``rank_kwarg`` is `None` where the class exposes no such control -- the
#: measured cost of a THC dRPA goes as the square of the rank, so which of these can be
#: held to a target rank is a first-order question, not a detail.
METHODS = {
    "becke": ("pythc.thc.ls_ri_becke", "LS_RI_Becke", None),
    "cholesky": ("pythc.thc.ls_ri_cholesky", "LS_RI_Cholesky", "cholesky_threshold"),
    "kmeans": ("pythc.thc.ls_ri_kmeans", "LS_RI_KMeans", "ips_per_naux"),
    "qrcp": ("pythc.thc.ls_ri_qrcp", "LS_RI_QRCP", "tolerance"),
    "nnls": ("pythc.thc.ls_ri_nnls", "LS_RI_NNLS", None),
    # `LS_Aux_Becke` keeps the entire parent Becke grid: its `prune` flag is False, so
    # `cholesky_threshold` has no effect and the rank lands around 60 x nao.  The pruned
    # subclass is the usable one; both are exposed so the difference stays visible.
    "aux-becke": ("pythc.thc.ls_aux_becke", "LS_Aux_Becke", "cholesky_threshold"),
    "aux-becke-cd": (
        "pythc.thc.ls_aux_becke",
        "LSTHC_AuxCoulomb_Cholesky",
        "cholesky_threshold",
    ),
    "snri": ("pythc.thc.ls_snri_cholesky", "LS_snRI_Cholesky", "cholesky_threshold"),
}

#: Default rank or accuracy control per method, used when `rank` is not given.
DEFAULT_RANK = {
    "cholesky": 1e-5,
    "kmeans": 2.5,
    "qrcp": 1e-6,
    "aux-becke": 1e-8,
    "snri": 1e-5,
}


def _import_pythc(method):
    """Import the PyTHC class for a method name.

    Parameters
    ----------
    method : str
        Key of `METHODS`.

    Returns
    -------
    cls : type
        The PyTHC builder class.
    rank_kwarg : str or None
        Keyword this class uses to control its rank or accuracy.

    Raises
    ------
    KeyError
        If `method` is not a known method.
    ImportError
        If PyTHC is not importable.
    """

    if method not in METHODS:
        raise KeyError(f"Unknown THC method {method!r}: expected one of {sorted(METHODS)}")

    module_name, class_name, rank_kwarg = METHODS[method]

    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "PyTHC is required to build THC integrals and is not importable in this "
            "environment. Either install it here "
            "(pip install git+https://github.com/QuantumCorrelators/pythc.git) or build "
            "the factorisation in an environment that has it and pass the resulting file "
            "through thc_opts['file_path']."
        ) from error

    return getattr(module, class_name), rank_kwarg


def build(mol, method="cholesky", rank=None, auxbasis=None, grid_level=0, **kwargs):
    """Build a THC factorisation of the electron repulsion integrals.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule.
    method : str, optional
        Point-selection strategy, one of the keys of `METHODS`.  Default value is
        `"cholesky"`.
    rank : float, optional
        Rank control, interpreted per method: a Cholesky threshold for `"cholesky"`,
        `"aux-becke"` and `"snri"`, interpolation points per auxiliary function for
        `"kmeans"`, and a pivot tolerance for `"qrcp"`.  `None` uses `DEFAULT_RANK`.
    auxbasis : str, optional
        Auxiliary basis for the density fitting inside the THC construction.  `None`
        appends `"-ri"` to the orbital basis name.  Default value is `None`.
    grid_level : int, optional
        PySCF grid level for the parent Becke grid.  Default value is `0`.
    **kwargs : dict, optional
        Additional keyword arguments passed to the PyTHC builder.

    Returns
    -------
    coll : numpy.ndarray
        Collocation matrix, shape ``(rank, nao)``, in momentGW's convention.
    cou : numpy.ndarray
        Coulomb matrix, shape ``(rank, rank)``.
    """

    cls, rank_kwarg = _import_pythc(method)
    from pythc.grid import BeckeGrid

    if auxbasis is None:
        auxbasis = f"{mol.basis}-ri"

    options = dict(mol=mol, grid=BeckeGrid(mol, level=grid_level), **kwargs)

    # `LS_Aux_Becke` fits against its own auxiliary basis and takes the orbital-basis one
    # separately; the rest take a single auxiliary basis.
    options["auxbasis"] = auxbasis

    if rank_kwarg is not None:
        value = rank if rank is not None else DEFAULT_RANK.get(method)
        if value is not None:
            options[rank_kwarg] = value

    builder = cls(**options)
    thc_eri = builder.build(mode="ao")
    coll, cou = thc_eri.get_X_Z()

    coll = np.ascontiguousarray(coll)
    if coll.shape[1] != mol.nao_nr():
        raise RuntimeError(
            f"PyTHC returned a collocation matrix of shape {coll.shape}, expected "
            f"(rank, {mol.nao_nr()})"
        )

    return coll, np.ascontiguousarray(cou)


def write(path, coll, cou, kpts=None):
    """Write a factorisation in the layout momentGW reads.

    The layout is the one `momentGW.thc.Integrals.import_thc_components` expects, which
    is COQUI's: a trailing axis of length two holding the real and imaginary parts, and a
    leading k-point axis on the Coulomb matrix.  A molecular factorisation is written as
    the single-k-point case so that the same file is readable by both the molecular and
    the periodic importer.

    Parameters
    ----------
    path : str
        Destination file.
    coll : numpy.ndarray
        Collocation matrix, shape ``(rank, nao)``.
    cou : numpy.ndarray
        Coulomb matrix, shape ``(rank, rank)``.
    kpts : numpy.ndarray, optional
        k-points, shape ``(nkpts, 3)``.  `None` writes a single k-point at the origin.
    """

    if kpts is None:
        kpts = np.zeros((1, 3))

    rank, nao = coll.shape

    collocation = np.zeros((nao, rank, 2))
    collocation[..., 0] = coll.T

    coulomb = np.zeros((len(kpts), rank, rank, 2))
    coulomb[0, ..., 0] = cou

    with h5py.File(path, "w") as handle:
        handle.create_dataset("collocation_matrix", data=collocation)
        handle.create_dataset("coulomb_matrix", data=coulomb)
        handle.create_dataset("kpts", data=np.asarray(kpts, dtype=float))
        handle.create_dataset("qpts", data=np.asarray(kpts, dtype=float))
        handle.create_dataset("Np", data=np.int32(rank))


def read(path):
    """Read a factorisation back, applying momentGW's own conventions.

    Kept here so a round trip can be checked without constructing an `Integrals` object,
    and so the two halves of the format live next to each other.

    Parameters
    ----------
    path : str
        Source file.

    Returns
    -------
    coll : numpy.ndarray
        Collocation matrix, shape ``(rank, nao)``.
    cou : numpy.ndarray
        Coulomb matrix, shape ``(rank, rank)``.
    """

    with h5py.File(path, "r") as handle:
        coll = np.array(handle["collocation_matrix"])[..., 0].T
        cou = np.array(handle["coulomb_matrix"])[0, ..., 0]

    return coll, cou


def density_fitted_eri(mol, auxbasis):
    """Build the density-fitted integral tensor a least-squares THC is fitting.

    This, not `mol.intor("int2e")`, is the reference a THC error should be quoted
    against.  Every `LS_RI_*` construction fits the density-fitted tensor, so measuring
    it against the exact one folds in the auxiliary basis's own error -- on water in
    cc-pVDZ that is `5.7e-3` relative, which is larger than anything the factorisation
    contributes and would hide it completely.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule.
    auxbasis : str
        Auxiliary basis.

    Returns
    -------
    eri : numpy.ndarray
        Density-fitted integrals, shape ``(nao, nao, nao, nao)``.
    """

    from pyscf import df
    from pyscf.lib import unpack_tril

    with_df = df.DF(mol, auxbasis=auxbasis)
    with_df.build()
    Lpq = unpack_tril(np.vstack(list(with_df.loop())))

    return np.einsum("Lpq,Lrs->pqrs", Lpq, Lpq)


def eri_error(coll, cou, reference):
    """Measure a factorisation against a reference electron repulsion tensor.

    Parameters
    ----------
    coll : numpy.ndarray
        Collocation matrix, shape ``(rank, nao)``.
    cou : numpy.ndarray
        Coulomb matrix, shape ``(rank, rank)``.
    reference : numpy.ndarray
        Reference integrals, shape ``(nao, nao, nao, nao)``.  Use
        `density_fitted_eri` unless the auxiliary basis error is wanted in the number.

    Returns
    -------
    max_abs : float
        Largest absolute deviation.
    rel_frobenius : float
        Frobenius norm of the deviation relative to the reference.
    """

    nao = coll.shape[1]
    pairs = np.einsum("Pp,Pq->pqP", coll, coll).reshape(nao * nao, -1)
    approx = (pairs @ cou @ pairs.T).reshape(nao, nao, nao, nao)
    diff = approx - reference

    return float(np.max(np.abs(diff))), float(np.linalg.norm(diff) / np.linalg.norm(reference))


def _main(argv=None):
    """Build a factorisation from the command line and write it to a file.

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
    parser.add_argument("--atom", required=True, help="geometry in PySCF's format")
    parser.add_argument("--basis", default="cc-pvdz", help="orbital basis")
    parser.add_argument("--auxbasis", default=None, help="auxiliary basis for the fit")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--method", default="cholesky", choices=sorted(METHODS))
    parser.add_argument("--rank", type=float, default=None, help="rank or accuracy control")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--out", required=True, help="destination HDF5 file")
    parser.add_argument(
        "--check-eri",
        action="store_true",
        help="also form the full reference ERI tensor and report the factorisation error",
    )
    args = parser.parse_args(argv)

    from pyscf import gto

    mol = gto.M(
        atom=args.atom,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        verbose=0,
    )

    coll, cou = build(
        mol,
        method=args.method,
        rank=args.rank,
        auxbasis=args.auxbasis,
        grid_level=args.grid_level,
    )
    write(args.out, coll, cou)

    print(
        f"{args.method}: nao {mol.nao_nr()}, rank {coll.shape[0]} "
        f"({coll.shape[0] / mol.nao_nr():.2f} x nao) -> {args.out}"
    )

    if args.check_eri:
        auxbasis = args.auxbasis or f"{mol.basis}-ri"
        fitted = density_fitted_eri(mol, auxbasis)
        max_abs, rel_frob = eri_error(coll, cou, fitted)
        print(f"  vs density-fitted: max abs {max_abs:.3e}, relative Frobenius {rel_frob:.3e}")

        exact = mol.intor("int2e")
        diff = fitted - exact
        print(
            f"  auxiliary basis itself: max abs {np.max(np.abs(diff)):.3e}, "
            f"relative Frobenius {np.linalg.norm(diff) / np.linalg.norm(exact):.3e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
