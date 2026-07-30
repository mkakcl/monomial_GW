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

#: The quantities compared, as (label, path into the record). A path is a tuple of keys;
#: an integer key indexes a list. Values may be scalars or lists of scalars.
COMPARED = (
    ("eta0 frobenius", ("eta0", "frobenius")),
    ("eta0 max", ("eta0", "max_abs")),
    ("eta0 condition", ("eta0", "condition")),
    ("eta0 quadrature scale", ("eta0", "grid_scale")),
    ("eta0 diagonal error", ("eta0", "diagonal_error")),
    ("dd moment frobenius", ("dd_moments", "frobenius")),
    ("hole se moment frobenius", ("se_moments", "hole", "frobenius")),
    ("particle se moment frobenius", ("se_moments", "particle", "frobenius")),
    ("hole realization rel. frobenius", ("realization", "hole", "errors", "relative_frobenius")),
    (
        "particle realization rel. frobenius",
        ("realization", "particle", "errors", "relative_frobenius"),
    ),
    ("auxiliary rank", ("auxiliary", "naux")),
    ("chemical potential", ("green_function", "chempot")),
    ("particle number error", ("green_function", "particle_number_error")),
    ("HOMO (Ha)", ("results", "homo_ha")),
    ("LUMO (Ha)", ("results", "lumo_ha")),
    ("QP energies (Ha)", ("results", "qp_energies_ha")),
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


def compare(recorded, fresh, rtol, atol):
    """Compare a fresh case record against a recorded one.

    Parameters
    ----------
    recorded : dict
        The committed record.
    fresh : dict
        The record from the rerun.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    Returns
    -------
    rows : list of dict
        One row per compared quantity, with the deviations found.
    """
    rows = []
    for label, path in COMPARED:
        deviation = _deviation(_dig(recorded, path), _dig(fresh, path), rtol, atol)
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
    parser.add_argument("--rtol", type=float, default=1e-10, help="relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-12, help="absolute tolerance")
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
        rows = compare(recorded, fresh, args.rtol, args.atol)
        checked += 1

        moved = [row for row in rows if row["status"] == "moved"]
        missing = [row for row in rows if row["status"] == "missing"]
        flag = "MOVED" if moved else "ok"
        homo_shift = (fresh["results"]["homo_ha"] - recorded["results"]["homo_ha"]) * frontier.EV
        print(f"{recorded['case_id']:52s} {flag:6s} HOMO shift {homo_shift:+.3e} eV")
        for row in moved:
            print(f"    {row['label']:38s} abs {row['absolute']:.3e}  rel {row['relative']:.3e}")
        for row in missing:
            print(f"    {row['label']:38s} not comparable (absent from one record)")
        if moved:
            moved_cases.append(recorded["case_id"])

    print(
        f"\n{checked - len(moved_cases)}/{checked} cases unchanged "
        f"at rtol={args.rtol:g}, atol={args.atol:g}"
    )
    return 1 if moved_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
