"""Re-run recorded baseline cases and report what moved.

This is what makes the recorded data a baseline rather than an archive. It reruns each
case, compares the quantities the roadmap separates errors into -- eta0, the
density-density moments, the self-energy moments, the realization residuals, the frontier
quasiparticle energies and the particle number -- and reports every one that moved by more
than a tolerance.

    python -m baseline.check                      # every recorded case
    python -m baseline.check --systems water      # a subset
    python -m baseline.check --rtol 1e-8          # a different tolerance

The exit status is non-zero if anything moved, so it can gate a change. Nothing here
asserts that a difference is wrong: a change that is understood and intended is reported
just the same, and the recorded baseline is then re-recorded deliberately.
"""

import argparse
import glob
import json
import os

import numpy as np

from baseline import frontier, systems
from baseline.run import DATA_DIR, build_mean_field, run_case

#: Tolerance for a quantity that is an exact function of the inputs -- an integer count, or
#: something that cannot move without the calculation having genuinely changed.
EXACT = (1e-12, 0.0)

#: Tolerance for a quantity that is a deterministic function of the inputs. The floor is
#: 1e-8 relative rather than machine precision, and that is not slack: it is the measured
#: run-to-run scatter, and it is *propagated from the quadrature grid scale below*. That
#: scale is selected on an objective which is flat to machine precision, so it lands
#: ~10% apart between identical runs; eta0 then differs at its own quadrature-error level,
#: and that difference carries through the moments into the quasiparticle energies, which
#: were measured moving by up to 1e-9 Ha (8e-11 eV in the frontier energies). Tightening
#: this would flag the calculation's own non-determinism as a change.
DETERMINISTIC = (1e-8, 1e-12)

#: Tolerance for a quantity whose recorded value sits at the floating-point noise floor,
#: where a relative comparison is meaningless: a residual of 6e-16 against another run's
#: 9e-16 is not a change in behaviour. The absolute floor is what does the work, and it is
#: set above the largest such residual observed, ~1e-12.
NOISE_FLOOR = (0.5, 1e-11)

#: Tolerance for a quantity that is the output of a search with its own convergence
#: tolerance, and is therefore only reproducible to that tolerance. The chemical potential
#: is located to `conv_tol_nelec`, which defaults to 1e-6.
SEARCHED = (1e-4, 1e-6)

#: Tolerance for the Clenshaw-Curtis grid scale. It is chosen by minimising the error of
#: the one integral known in closed form, and on these systems that objective is flat to
#: machine precision over a wide range of scales -- the recorded diagonal error is 0.0
#: exactly for several cases. The minimiser therefore returns an arbitrary point on a
#: plateau, and last-bit differences in the mean field move it by ~10% between otherwise
#: identical runs. It is compared loosely because it is a free parameter of the method, not
#: a result; a change large enough to trip this would mean the plateau itself had moved.
PLATEAU = (0.5, 0.0)

#: The quantities compared, as (label, path into the record, (rtol, atol)). A path is a
#: tuple of keys; an integer key indexes a list. Values may be scalars or lists of scalars.
COMPARED = (
    ("eta0 frobenius", ("eta0", "frobenius"), DETERMINISTIC),
    ("eta0 max", ("eta0", "max_abs"), DETERMINISTIC),
    ("eta0 condition", ("eta0", "condition"), DETERMINISTIC),
    ("eta0 error vs oracle", ("eta0", "oracle", "relative_error"), NOISE_FLOOR),
    ("Mtilde condition", ("eta0", "oracle", "mtilde_condition"), DETERMINISTIC),
    ("eta0 quadrature scale", ("eta0", "grid_scale"), PLATEAU),
    ("eta0 diagonal error", ("eta0", "diagonal_error"), NOISE_FLOOR),
    ("dd moment frobenius", ("dd_moments", "frobenius"), DETERMINISTIC),
    ("hole se moment frobenius", ("se_moments", "hole", "frobenius"), DETERMINISTIC),
    ("particle se moment frobenius", ("se_moments", "particle", "frobenius"), DETERMINISTIC),
    (
        "hole reconstructed frobenius",
        ("realization", "hole", "reconstructed_moments", "frobenius"),
        DETERMINISTIC,
    ),
    (
        "particle reconstructed frobenius",
        ("realization", "particle", "reconstructed_moments", "frobenius"),
        DETERMINISTIC,
    ),
    (
        "hole realization rel. frobenius",
        ("realization", "hole", "errors", "relative_frobenius"),
        NOISE_FLOOR,
    ),
    (
        "particle realization rel. frobenius",
        ("realization", "particle", "errors", "relative_frobenius"),
        NOISE_FLOOR,
    ),
    ("auxiliary rank", ("auxiliary", "naux"), EXACT),
    ("chemical potential", ("green_function", "chempot"), SEARCHED),
    ("particle number error", ("green_function", "particle_number_error"), SEARCHED),
    ("HOMO (Ha)", ("results", "homo_ha"), DETERMINISTIC),
    ("LUMO (Ha)", ("results", "lumo_ha"), DETERMINISTIC),
    ("QP energies (Ha)", ("results", "qp_energies_ha"), DETERMINISTIC),
)


def _dig(record, path):
    """Follow a path of keys into a nested record, returning `None` if it is absent."""
    value = record
    for key in path:
        if value is None:
            return None
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            return None
    return value


def _deviation(recorded, fresh, rtol, atol):
    """Get the largest absolute and relative deviation between two values.

    Returns
    -------
    worst : tuple or None
        The absolute deviation, the relative deviation and whether it
        exceeds the tolerance. `None` if the values are not comparable.
    """
    if recorded is None or fresh is None:
        return None
    old = np.atleast_1d(np.asarray(recorded, dtype=float))
    new = np.atleast_1d(np.asarray(fresh, dtype=float))
    if old.shape != new.shape:
        return (np.inf, np.inf, True)
    absolute = np.abs(new - old)
    scale = np.maximum(np.abs(old), np.abs(new))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(scale > 0.0, absolute / scale, 0.0)
    index = int(np.argmax(absolute))
    exceeded = bool(np.any(absolute > atol + rtol * scale))
    return (float(absolute[index]), float(relative[index]), exceeded)


def compare(recorded, fresh, rtol=None, atol=None):
    """Compare a fresh case record against a recorded one.

    Parameters
    ----------
    recorded : dict
        The committed record.
    fresh : dict
        The record from the rerun.
    rtol : float, optional
        Relative tolerance, applied to every quantity in place of its own.
        If `None`, each quantity uses the tolerance it is listed with in
        `COMPARED`. Default value is `None`.
    atol : float, optional
        Absolute tolerance, applied in the same way. Default value is
        `None`.

    Returns
    -------
    rows : list of dict
        One row per compared quantity, with the deviations found.
    """
    rows = []
    for label, path, (row_rtol, row_atol) in COMPARED:
        if rtol is not None:
            row_rtol = rtol
        if atol is not None:
            row_atol = atol
        deviation = _deviation(_dig(recorded, path), _dig(fresh, path), row_rtol, row_atol)
        if deviation is None:
            rows.append({"label": label, "status": "missing"})
            continue
        absolute, relative, exceeded = deviation
        rows.append(
            {
                "label": label,
                "status": "moved" if exceeded else "ok",
                "absolute": absolute,
                "relative": relative,
            }
        )
    return rows


def main():
    """Rerun the recorded cases and report the deviations."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DATA_DIR, help="directory holding the records")
    parser.add_argument("--systems", nargs="+", default=None, help="only these systems")
    parser.add_argument("--orders", nargs="+", type=int, default=None, help="only these orders")
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="relative tolerance for every quantity, overriding its own",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="absolute tolerance for every quantity, overriding its own",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.json")))
    paths = [path for path in paths if not path.endswith("index.json")]

    mean_fields = {}
    moved_cases = []
    checked = 0

    for path in paths:
        with open(path) as handle:
            recorded = json.load(handle)
        if recorded.get("status") != "complete":
            continue
        system = systems.SYSTEMS[recorded["system"]["name"]]
        xc = recorded["mean_field"]["xc"]
        options = recorded["options"]
        if args.systems is not None and system.name not in args.systems:
            continue
        if args.orders is not None and options["nmom_max"] not in args.orders:
            continue

        if (system.name, xc) not in mean_fields:
            mean_fields[(system.name, xc)] = build_mean_field(system, xc)
        mf, mean_field = mean_fields[(system.name, xc)]

        fresh = run_case(
            mf,
            mean_field,
            system,
            options["nmom_max"],
            compression=options["compression"],
            compression_tol=options["compression_tol"],
            save_arrays=False,
        )
        rows = compare(recorded, fresh, rtol=args.rtol, atol=args.atol)
        checked += 1

        moved = [row for row in rows if row["status"] == "moved"]
        missing = [row for row in rows if row["status"] == "missing"]
        flag = "MOVED" if moved else "ok"
        homo_shift = (fresh["results"]["homo_ha"] - recorded["results"]["homo_ha"]) * frontier.EV

        # The largest relative deviation of any quantity, reported whether or not it broke a
        # tolerance. This is what the tolerances above were calibrated from: run the check
        # on an unchanged code base and the worst column is the scatter to sit just above.
        comparable = [row for row in rows if row["status"] != "missing"]
        worst = max(comparable, key=lambda row: row["relative"], default=None)
        worst_text = f" | worst {worst['label']} rel {worst['relative']:.1e}" if worst else ""
        print(f"{recorded['case_id']:52s} {flag:6s} HOMO shift {homo_shift:+.3e} eV{worst_text}")
        for row in moved:
            print(f"    {row['label']:38s} abs {row['absolute']:.3e}  rel {row['relative']:.3e}")
        for row in missing:
            print(f"    {row['label']:38s} not comparable (absent from one record)")
        if moved:
            moved_cases.append(recorded["case_id"])

    if args.rtol is None and args.atol is None:
        tolerance = "at each quantity's own tolerance"
    else:
        tolerance = f"at rtol={args.rtol}, atol={args.atol} applied to every quantity"
    print(f"\n{checked - len(moved_cases)}/{checked} cases unchanged {tolerance}")
    return 1 if moved_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
