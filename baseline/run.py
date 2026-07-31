"""Record the Milestone 0 baseline for the restricted molecular G0W0/dRPA path.

Each case runs the production pipeline stage by stage, so that every intermediate the
roadmap asks for can be recorded with the time it took to produce and the code revision
that produced it. The stages are the ones the roadmap attributes error to separately:

    integrals -> eta0 -> density-density moments -> self-energy moments -> realization

Run the whole set with `python -m baseline.run`, or a subset with `--systems`,
`--starting-points` and `--orders`. Results are written one JSON file per case into
`baseline/data/`, with an `index.json` recording the sweep and its provenance. The large
arrays -- eta0 and the moments themselves -- go to `baseline/arrays/` as compressed
`.npz`, which is not committed: it is reproducible from this script, and the JSON carries
the norms and spectra that a comparison actually reads.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import time
import traceback

# Both codes report through `rich`. Silence them before import: the numbers below are read
# off the objects, never parsed back out of the rendered prose.
os.environ.setdefault("MOMENTGW_SILENT", "1")
os.environ.setdefault("DYSON_QUIET", "1")

import dyson  # noqa: E402
import numpy as np  # noqa: E402
import pyscf  # noqa: E402
import scipy  # noqa: E402
from pyscf import dft, gto  # noqa: E402

import momentGW  # noqa: E402
from baseline import frontier, systems  # noqa: E402
from momentGW import GW, mpi_helper, util  # noqa: E402
from momentGW.rpa import dRPA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ARRAYS_DIR = os.path.join(HERE, "arrays")


def _git(path, *args):
    """Run a git command in the repository containing `path`, returning `None` on failure."""
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(path), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return out.stdout.strip()


def _pinned_commit(module):
    """Get the commit pip resolved when it installed a module from a version-control URL.

    A dependency pinned in `pyproject.toml` as `name @ git+...@<sha>` unpacks into
    site-packages with no `.git` beside it, so `_git` cannot name its commit. pip records
    what it resolved in `direct_url.json` (PEP 610), which is then the only surviving
    statement of which revision is installed.

    Returns
    -------
    record : dict or None
        The commit, the revision that was requested and the source URL, or
        `None` if the module was not installed from version control.
    """
    try:
        names = importlib.metadata.packages_distributions().get(module.__name__, [])
        for name in names:
            text = importlib.metadata.distribution(name).read_text("direct_url.json")
            if text is None:
                continue
            direct_url = json.loads(text)
            commit = (direct_url.get("vcs_info") or {}).get("commit_id")
            if commit:
                return {
                    "commit": commit,
                    "requested_revision": direct_url["vcs_info"].get("requested_revision"),
                    "url": direct_url.get("url"),
                }
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError, KeyError):
        return None
    return None


def _repo_state(module):
    """Describe the checkout an imported module was loaded from.

    The baseline is normally recorded from a worktree while the installed package resolves
    to the main checkout, so the path is recorded and not assumed.

    A module installed from a pinned git URL rather than checked out has no repository to
    interrogate. Its commit is recovered from what pip recorded instead, because a record
    that cannot name the revision of a pinned dependency defeats the purpose of pinning it.
    """
    path = os.path.abspath(module.__file__)
    record = {
        "path": path,
        "commit": _git(path, "rev-parse", "HEAD"),
        "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(path, "status", "--porcelain", "--untracked-files=no")),
        "version": getattr(module, "__version__", None),
    }
    if record["commit"] is None:
        pinned = _pinned_commit(module)
        if pinned is not None:
            record.update(pinned)
            record["source"] = "pip direct_url"
    return record


def _blas_info():
    """Get the BLAS and LAPACK NumPy was built against, if it will say."""
    try:
        config = np.show_config(mode="dicts")
    except (TypeError, AttributeError):
        return None
    build = config.get("Build Dependencies", {})
    return {
        key: {k: value.get(k) for k in ("name", "version", "detection method")}
        for key, value in build.items()
        if key in ("blas", "lapack")
    }


def provenance():
    """Record the exact software revisions and machine behind a sweep.

    Returns
    -------
    record : dict
        Code revisions, library versions, BLAS identification and host.
    """
    return {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": {
            "momentGW": _repo_state(momentGW),
            "dyson": _repo_state(dyson),
            "baseline_script": {
                "path": HERE,
                "commit": _git(HERE + os.sep, "rev-parse", "HEAD"),
                "branch": _git(HERE + os.sep, "rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git(HERE + os.sep, "status", "--porcelain", "--untracked-files=no")),
            },
        },
        "versions": {
            "python": platform.python_version(),
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "blas": _blas_info(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "threads": {
                var: os.environ.get(var)
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
    }


def _peak_memory_gb():
    """Get the process-wide high-water mark of resident memory, in GB.

    This is the whole process, not the case, so it only ever rises through a sweep. It
    bounds the memory a case needed; it does not measure it.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024**2 if platform.system() == "Darwin" else 1024
    return usage / scale / 1024


def _summarise(array):
    """Summarise an array by the quantities a later comparison can act on.

    Norms detect a change; the singular values detect *where* it is, and give the
    conditioning that the inverse square root in Milestone 2 will depend on.
    """
    array = np.asarray(array)
    flat = array.reshape(-1, array.shape[-1]) if array.ndim > 2 else array
    record = {
        "shape": list(array.shape),
        "frobenius": float(np.linalg.norm(array)),
        "max_abs": float(np.max(np.abs(array))) if array.size else 0.0,
    }
    if flat.ndim == 2 and min(flat.shape) > 0:
        singular = np.linalg.svd(flat, compute_uv=False)
        record["singular_values_head"] = [float(x) for x in singular[:8]]
        record["singular_values_tail"] = [float(x) for x in singular[-3:]]
        record["condition"] = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        record["rank_1e-10"] = int(np.sum(singular > 1e-10 * singular[0]))
    return record


def _per_order(moments):
    """Summarise a stack of moments order by order."""
    moments = np.asarray(moments)
    return {
        "orders": list(range(moments.shape[0])),
        "frobenius": [float(np.linalg.norm(m)) for m in moments],
        "max_abs": [float(np.max(np.abs(m))) for m in moments],
    }


def _moment_errors(errors):
    """Convert a Dyson `MomentErrors` record to a JSON-serialisable dict."""
    record = {
        "orders": list(errors.orders),
        "absolute_frobenius": list(errors.absolute_frobenius),
        "relative_frobenius": list(errors.relative_frobenius),
        "absolute_max": list(errors.absolute_max),
        "relative_max": list(errors.relative_max),
        "total_scaled": float(errors.total),
        "max_relative_frobenius": float(errors.max_relative_frobenius),
        "chempot": float(errors.chempot),
    }
    if errors.shifted is not None:
        record["shifted"] = _moment_errors(errors.shifted)
    return record


def build_mean_field(system, xc):
    """Build the density-fitted mean-field reference for a system.

    Parameters
    ----------
    system : systems.System
        The system to build.
    xc : str
        Exchange-correlation functional, or `"hf"`.

    Returns
    -------
    mf : pyscf.dft.RKS
        Converged mean field.
    record : dict
        Reference metadata.

    Raises
    ------
    RuntimeError
        If the SCF does not converge, rather than recording a baseline
        against an unconverged reference.
    """
    mol = gto.M(
        atom=system.atom,
        basis=system.basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    assert mol.nelectron == system.nelectron

    mf = dft.RKS(mol)
    mf.xc = xc
    mf = mf.density_fit(auxbasis=system.auxbasis)
    mf.conv_tol = systems.SCF_CONV_TOL

    start = time.perf_counter()
    mf.kernel()
    wall = time.perf_counter() - start

    if not mf.converged:
        raise RuntimeError(f"reference SCF for {system.name}/{xc} did not converge")

    occupied = mf.mo_energy[mf.mo_occ > 0]
    virtual = mf.mo_energy[mf.mo_occ == 0]
    record = {
        "xc": xc,
        "basis": system.basis,
        "auxbasis": system.auxbasis,
        "conv_tol": systems.SCF_CONV_TOL,
        "e_tot": float(mf.e_tot),
        "nelectron": int(mol.nelectron),
        "nmo": int(mf.mo_coeff.shape[-1]),
        "nocc": int(np.sum(mf.mo_occ > 0)),
        "homo_ha": float(np.max(occupied)),
        "lumo_ha": float(np.min(virtual)),
        # The mean-field gap, not the correlated one. This is the quantity that sets the
        # smallest particle-hole denominator entering the dRPA integrand, so it is the
        # sense in which a system here is or is not "small-gap".
        "gap_ev": float((np.min(virtual) - np.max(occupied)) * frontier.EV),
        "scf_wall_seconds": round(wall, 3),
    }
    return mf, record


def _eta0_with_diagnostics(rpa):
    """Build the zeroth density-density moment, keeping the quadrature's own diagnostics.

    This repeats `dRPA.build_zeroth_dd_moment`, which reports the grid scale and the nested
    error estimate to the log and then discards them. They are the current accuracy claim
    for eta0, and Milestone 2 has to replace them with a certificate, so the baseline has
    to carry the numbers being replaced.

    Parameters
    ----------
    rpa : momentGW.rpa.dRPA
        The polarizability object, already constructed.

    Returns
    -------
    eta0 : numpy.ndarray
        The zeroth moment of the density-density response.
    record : dict
        Quadrature diagnostics.
    """
    # `eval_main_integral` reads the local slice of the energy differences off the object.
    # In the production path `build_se_moments` populates it before calling through to the
    # quadrature; staged like this, that has to be done here.
    if rpa.d is None:
        rpa._build_d()

    p0, p1 = rpa.mpi_slice(rpa.nov)
    d_full = util.build_1h1p_energies(rpa.mo_energy_w, rpa.mo_occ_w).ravel()

    diag_eri = np.zeros((rpa.nov,))
    diag_eri[p0:p1] = util.einsum("np,np->p", rpa.integrals.Lia, rpa.integrals.Lia)
    diag_eri = mpi_helper.allreduce(diag_eri)

    bare_quad = rpa.gen_clencur_quad_semiinf()
    quad = rpa.optimise_main_quad(d_full, diag_eri)
    scale = float(quad[0][0] / bare_quad[0][0])

    # The optimiser's own objective: how well the quadrature reproduces the one integral
    # whose value is known in closed form, the diagonal.
    exact = float(np.sum(d_full * (d_full * (d_full + diag_eri)) ** -0.5))
    diagonal = float(rpa.eval_diag_main_integral(quad, d_full, diag_eri))

    integral = rpa.eval_main_integral(quad)
    half = float(np.sum((integral[0] - integral[2]) ** 2) ** 0.5)
    quarter = float(np.sum((integral[0] - integral[1]) ** 2) ** 0.5)
    estimate = rpa.estimate_error_clencur(half, quarter)

    record = {
        "method": "clencur",
        "npoints": int(rpa.gw.npoints),
        "grid_scale": scale,
        "diagonal_exact": exact,
        "diagonal_quadrature": diagonal,
        "diagonal_error": abs(diagonal - exact),
        # Nested-grid estimate, from eq. 103 of arXiv:2301.09107. An estimate over a
        # sequence of grids, not a bound on the error of the grid in use.
        "nested_error_estimate": float(estimate) if np.isfinite(estimate) else None,
        "half_grid_difference": half,
        "quarter_grid_difference": quarter,
    }
    return integral[0], record


class _timed:
    """Context manager recording the wall time of a stage into a dict."""

    def __init__(self, into, name):
        self.into = into
        self.name = name

    def __enter__(self):
        """Start the clock."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        """Stop the clock and record the elapsed time."""
        self.into[self.name] = time.perf_counter() - self.start
        return False


#: Largest particle-hole space for which the dense eta0 oracle is built. The oracle forms
#: and diagonalises an `nov` by `nov` matrix, which is exactly the object the production
#: path is built to avoid; it is affordable only because these systems are small.
ORACLE_MAX_NOV = 2000


def eta0_oracle(rpa):
    r"""Compute eta0 exactly, by eigendecomposition, as a reference for the quadrature.

    The projected zeroth dRPA moment has a closed form,

    .. math::
        \eta_0 = L D^{1/2} \tilde{M}^{-1/2} D^{1/2}, \qquad
        \tilde{M} = D^2 + 4 W W^T, \qquad W = D^{1/2} L^T

    with `D` the diagonal particle-hole energy differences and `L` the RI integrals. Formed
    densely this is `nov` by `nov` and does not scale, which is why the production path
    integrates instead. At these system sizes it is affordable, and it turns the recorded
    eta0 error from an estimate into a measurement.

    Parameters
    ----------
    rpa : momentGW.rpa.dRPA
        The polarizability object, with the energy differences already built.

    Returns
    -------
    eta0 : numpy.ndarray
        The exact projected zeroth moment.
    record : dict
        The spectrum and condition number of `Mtilde`.
    """
    d = rpa.d
    Lia = rpa.integrals.Lia
    w = (d**0.5)[:, None] * Lia.T
    mtilde = np.diag(d**2) + 4.0 * (w @ w.T)
    eigvals, eigvecs = np.linalg.eigh(mtilde)
    inv_sqrt = (eigvecs / np.sqrt(eigvals)[None]) @ eigvecs.T
    exact = Lia @ ((d**0.5)[:, None] * inv_sqrt * (d**0.5)[None])
    record = {
        "mtilde_min_eigenvalue": float(eigvals[0]),
        "mtilde_max_eigenvalue": float(eigvals[-1]),
        "mtilde_condition": float(eigvals[-1] / eigvals[0]),
    }
    return exact, record


def run_case(mf, mean_field, system, nmom_max, *, compression, compression_tol, save_arrays=True):
    """Run one baseline case and record everything the roadmap asks a calculation to report.

    Parameters
    ----------
    mf : pyscf.dft.RKS
        Converged mean field.
    mean_field : dict
        Reference metadata from `build_mean_field`.
    system : systems.System
        The system being run.
    nmom_max : int
        Maximum moment order.
    compression : str
        Auxiliary compression sectors, or `""` for none.
    compression_tol : float
        Auxiliary compression tolerance.
    save_arrays : bool, optional
        Whether to write eta0 and the moments to `baseline/arrays/`.
        Default value is `True`.

    Returns
    -------
    record : dict
        The full case record.
    """
    identifier = case_id(system, mean_field["xc"], nmom_max, compression)
    timings = {}
    started = time.perf_counter()

    gw = GW(mf, polarizability="drpa")
    gw.compression = compression
    gw.compression_tol = compression_tol

    with _timed(timings, "integrals"):
        integrals = gw.ao2mo()

    with _timed(timings, "se_static"):
        se_static = gw.build_se_static(integrals)

    mo_energy = dict(g=gw.mo_energy, w=gw.mo_energy)
    rpa = dRPA(gw, nmom_max, integrals, mo_energy=mo_energy)

    with _timed(timings, "eta0"):
        eta0, eta0_record = _eta0_with_diagnostics(rpa)

    # The quadrature's own error estimate is an extrapolation over coarser grids, not a
    # bound. Where the exact answer is affordable, record the error against it instead, so
    # the recorded accuracy of eta0 is measured rather than claimed.
    with _timed(timings, "eta0_oracle"):
        if rpa.nov <= ORACLE_MAX_NOV:
            exact, oracle_record = eta0_oracle(rpa)
            difference = eta0 - exact
            norm = float(np.linalg.norm(exact))
            eta0_record["oracle"] = {
                **oracle_record,
                "absolute_error": float(np.linalg.norm(difference)),
                "relative_error": float(np.linalg.norm(difference) / norm) if norm else None,
                "max_abs_error": float(np.max(np.abs(difference))),
            }
        else:
            eta0_record["oracle"] = {"skipped": f"nov > {ORACLE_MAX_NOV}"}

    with _timed(timings, "dd_moments"):
        dd_moments = rpa.build_dd_moments(integral=eta0)

    with _timed(timings, "se_moments_from_dd"):
        th_staged, tp_staged = rpa.build_se_moments(moments_dd=dd_moments)

    # The production path builds the density-density moments one order at a time through a
    # different recurrence, and is what `gw.kernel` runs. Both are recorded: their
    # disagreement is a direct measure of how much the monomial recurrences amplify
    # roundoff, which is what Milestone 3 has to control.
    with _timed(timings, "se_moments_streaming"):
        th, tp = dRPA(gw, nmom_max, integrals, mo_energy=mo_energy).kernel()

    with _timed(timings, "dyson"):
        converged, gf, se, qp_energy = gw.kernel(
            nmom_max=nmom_max, moments=(th, tp), integrals=integrals
        )

    # Realization diagnostics, per sector, read off the solve above rather than rebuilt.
    # `gw.solve_dyson` keeps the solvers it used and reports what each sector realized on
    # `dyson_diagnostics`; a second set of solvers built here would be a re-derivation that
    # could disagree with the realization these numbers are supposed to describe.
    with _timed(timings, "realization_diagnostics"):
        realization = {}
        reconstructed = {}
        for sector in ("hole", "particle"):
            realized = gw.dyson_diagnostics["realization"][sector]
            # The order that ran is not always the order that was asked for: Dyson steps the
            # recurrence down when an iteration produces an off-diagonal square with no real
            # square root, and solves at the last one it completed. Everything below is read
            # at the achieved order, because that is the realization the Dyson solve above
            # actually used; the requested order is recorded beside it, not in place of it.
            achieved = realized["max_cycle_achieved"]
            # The moments the realized self-energy actually carries, as distinct from the
            # moments it was asked to carry. Recorded in full alongside the errors, because
            # an error norm cannot say which order or which block a discrepancy sits in.
            reconstructed[sector] = np.asarray(
                gw.dyson_solvers[sector].reconstruct_moments(achieved)
            )
            realization[sector] = {
                "requested_nmom_max": nmom_max,
                "moments_supplied": realized["moments_supplied"],
                "max_cycle": realized["max_cycle"],
                "max_cycle_achieved": achieved,
                "order_reduced": realized["order_reduced"],
                "nmom_conserved_requested": realized["nmom_conserved_requested"],
                "nmom_conserved_achieved": realized["nmom_conserved_achieved"],
                "n_poles": realized["n_poles"],
                "reconstructed_moments": _per_order(reconstructed[sector]),
                "errors": _moment_errors(realized["errors"]),
            }

    hole_error, particle_error = gw.moment_error(th, tp, se)
    nelectron = float(gf.occupied().moment(0).trace() * 2)

    record = {
        "schema_version": "1.1",
        "case_id": identifier,
        "status": "complete",
        "system": system.to_dict(),
        "mean_field": mean_field,
        "options": {
            "polarizability": gw.polarizability,
            "nmom_max": nmom_max,
            "npoints": int(gw.npoints),
            "compression": compression,
            "compression_tol": compression_tol,
            "diagonal_se": bool(gw.diagonal_se),
            "optimise_chempot": bool(gw.optimise_chempot),
            "fock_loop": bool(gw.fock_loop),
        },
        "auxiliary": {
            "naux_full": int(integrals.naux_full),
            "naux": int(integrals.naux),
            "compressed": int(integrals.naux) != int(integrals.naux_full),
            # The discarded norm is not available: the compression selects on the
            # eigenvalues of the metric with an absolute cutoff and does not report what it
            # dropped. Milestone 4 replaces that criterion; until then the rank is all this
            # can honestly record.
            "discarded_norm": None,
        },
        "eta0": {**eta0_record, **_summarise(eta0)},
        "dd_moments": _per_order(dd_moments),
        "se_moments": {
            "hole": _per_order(th),
            "particle": _per_order(tp),
            "streaming_vs_staged_max_abs": {
                "hole": float(np.max(np.abs(th - th_staged))),
                "particle": float(np.max(np.abs(tp - tp_staged))),
            },
        },
        "realization": realization,
        "moment_error": {
            "hole": float(hole_error),
            "particle": float(particle_error),
            "definition": "momentGW scaled error between input and realized self-energy moments",
        },
        "green_function": {
            "converged_flag": bool(converged),
            "converged_flag_note": (
                "momentGW returns True unconditionally for one-shot GW; it is not a "
                "numerical convergence result. Milestone 1.4 replaces it."
            ),
            "n_poles": int(np.asarray(gf.energies).size),
            "chempot": float(gf.chempot),
            "nelectron": nelectron,
            "particle_number_error": nelectron - system.nelectron,
            "se_n_poles": int(np.asarray(se.energies).size),
        },
        "results": frontier.readouts(
            gf.energies,
            gf.couplings,
            system.nelectron,
            mo_energy=mf.mo_energy,
            mo_occ=mf.mo_occ,
        ),
        "qp_energy_by_overlap_ha": [float(x) for x in np.asarray(qp_energy).ravel()],
        "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        # Wall time on a shared machine is only interpretable next to the load it was
        # measured under. These are the 1, 5 and 15 minute averages at the end of the case;
        # anything approaching the core count means the timings above are contended and
        # cannot be compared against a run recorded on a quiet machine.
        "load_average": [round(x, 2) for x in os.getloadavg()],
        "cpu_count": os.cpu_count(),
        "peak_memory_gb": round(_peak_memory_gb(), 4),
    }
    record["timings_seconds"]["total"] = round(time.perf_counter() - started, 4)

    if save_arrays:
        os.makedirs(ARRAYS_DIR, exist_ok=True)
        path = os.path.join(ARRAYS_DIR, f"{identifier}.npz")
        np.savez_compressed(
            path,
            eta0=eta0,
            dd_moments=dd_moments,
            se_moments_hole=th,
            se_moments_particle=tp,
            se_static=se_static,
            gf_energies=gf.energies,
            gf_couplings=gf.couplings,
            se_energies=se.energies,
            se_couplings=se.couplings,
            reconstructed_moments_hole=reconstructed["hole"],
            reconstructed_moments_particle=reconstructed["particle"],
        )
        record["arrays"] = os.path.relpath(path, HERE)
    else:
        record["arrays"] = None

    return record


def case_id(system, xc, nmom_max, compression):
    """Build the identifier for a case.

    Parameters
    ----------
    system : systems.System
        The system.
    xc : str
        Starting point.
    nmom_max : int
        Maximum moment order.
    compression : str
        Auxiliary compression sectors, or `""` for none.

    Returns
    -------
    identifier : str
        The case identifier.
    """
    tag = compression.replace(",", "-") if compression else "nocompression"
    return f"{system.name}_{xc}_nmom{nmom_max}_{tag}"


#: Systems additionally run with the auxiliary compression disabled, giving the roadmap's
#: "compressed and uncompressed auxiliary spaces for at least one molecule". Two are needed
#: because the criterion does nothing on one of them: with the `"ia"` metric the compression
#: is selecting on the eigenvalues of a Gram matrix whose rank cannot exceed the number of
#: particle-hole pairs, so it removes exactly the null directions and nothing else.
#: Lithium-hydride has more auxiliary functions than particle-hole pairs and loses 60 -> 34;
#: water has fewer and loses nothing, making it the control that shows the tolerance is not
#: what is doing the work.
COMPRESSION_PAIR = ("lithium-hydride", "water")


def plan(selected_systems, starting_points, orders):
    """Enumerate the cases in a sweep.

    Every system is run at every requested order and starting point with the default
    auxiliary compression. The systems in `COMPRESSION_PAIR` are additionally run with
    compression disabled.

    Parameters
    ----------
    selected_systems : list of str
        System names.
    starting_points : list of str
        Mean-field starting points.
    orders : list of int
        Values of `nmom_max`.

    Returns
    -------
    cases : list of tuple
        Tuples of system, starting point, order, compression.
    """
    cases = []
    for name in selected_systems:
        system = systems.SYSTEMS[name]
        for xc in starting_points:
            # The 6-31g anchor exists to reproduce a published Hartree-Fock number.
            if system.name == "hydrogen-631g" and xc != "hf":
                continue
            for nmom_max in orders:
                cases.append((system, xc, nmom_max, "ia"))
                if system.name in COMPRESSION_PAIR:
                    cases.append((system, xc, nmom_max, ""))
    return cases


def main():
    """Run a baseline sweep and write the records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(systems.SYSTEMS),
        choices=list(systems.SYSTEMS),
        help="systems to run",
    )
    parser.add_argument(
        "--starting-points",
        nargs="+",
        default=list(systems.STARTING_POINTS),
        help="mean-field starting points",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=list(systems.MOMENT_ORDERS),
        help="values of nmom_max",
    )
    parser.add_argument("--output", default=DATA_DIR, help="directory for the JSON records")
    parser.add_argument(
        "--no-arrays", action="store_true", help="skip writing eta0 and the moments to .npz"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cases = plan(args.systems, args.starting_points, args.orders)
    meta = provenance()

    print(f"momentGW {meta['code']['momentGW']['path']} @ {meta['code']['momentGW']['commit']}")
    print(f"dyson    {meta['code']['dyson']['path']} @ {meta['code']['dyson']['commit']}")
    print(f"{len(cases)} cases")

    index = {
        "schema_version": "1.1",
        "provenance": meta,
        "sweep": {
            "systems": args.systems,
            "starting_points": args.starting_points,
            "orders": args.orders,
            "compression": ["ia", ""],
        },
        "cases": [],
    }

    mean_fields = {}
    for system, xc, nmom_max, compression in cases:
        identifier = case_id(system, xc, nmom_max, compression)
        started = time.perf_counter()
        try:
            if (system.name, xc) not in mean_fields:
                mean_fields[(system.name, xc)] = build_mean_field(system, xc)
            mf, mean_field = mean_fields[(system.name, xc)]
            record = run_case(
                mf,
                mean_field,
                system,
                nmom_max,
                compression=compression,
                compression_tol=1e-10,
                save_arrays=not args.no_arrays,
            )
            summary = (
                f"HOMO {record['results']['homo_ha'] * frontier.EV:9.4f} eV  "
                f"dN {record['green_function']['particle_number_error']:+.2e}"
            )
        except Exception as error:
            # A failed case is a recorded result, not a reason to abandon the sweep.
            record = {
                "schema_version": "1.1",
                "case_id": identifier,
                "status": "failed",
                "system": system.to_dict(),
                "options": {"nmom_max": nmom_max, "compression": compression},
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(),
            }
            summary = f"FAILED {type(error).__name__}: {error}"

        record["provenance"] = meta
        with open(os.path.join(args.output, f"{identifier}.json"), "w") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
            handle.write("\n")

        wall = time.perf_counter() - started
        index["cases"].append({"case_id": identifier, "status": record["status"]})
        print(f"  {identifier:52s} {wall:7.1f}s  {summary}")

    failed = [case for case in index["cases"] if case["status"] != "complete"]
    index["n_cases"] = len(index["cases"])
    index["n_failed"] = len(failed)
    with open(os.path.join(args.output, "index.json"), "w") as handle:
        json.dump(index, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"{len(index['cases']) - len(failed)}/{len(index['cases'])} complete")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
