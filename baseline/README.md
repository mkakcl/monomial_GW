# Milestone 0 — reproducible baseline

The reference point every later change in [`ROADMAP.md`](../ROADMAP.md) is measured against.
Not a benchmark: nothing here is scored against published values or against other codes.
Its only job is to record what the restricted molecular G0W0/dRPA path currently produces,
in enough detail that a later change can be shown to have moved one stage and not another.

```bash
python -m baseline.run                 # record the full set (~1 minute)
python -m baseline.check               # re-run it and report what moved
python -m baseline.check --systems water --rtol 1e-8
```

`check` exits non-zero if anything moved by more than the tolerance. It does not decide
whether a movement is wrong — an intended change is reported just the same, and the
baseline is then deliberately re-recorded.

## What is recorded

One JSON file per case in [`data/`](data), plus `index.json` for the sweep. Each record
carries every item the roadmap's *definition of a trustworthy calculation* asks for, except
where the code cannot currently supply it — those are recorded as `null` with the reason,
rather than filled in with a proxy:

| Field | What it is |
|---|---|
| `provenance` | momentGW and Dyson commits (with the checkout path and dirty flag), PySCF/NumPy/SciPy versions, the BLAS NumPy was built against, host and thread environment |
| `mean_field` | starting point, basis, auxiliary basis, SCF tolerance, total energy, and the **mean-field** HOMO–LUMO gap — the quantity that sets the smallest particle-hole denominator entering the dRPA integrand |
| `auxiliary` | `naux_full`, `naux`, and whether compression fired. `discarded_norm` is `null`: the compression selects on an absolute eigenvalue cutoff and never reports what it dropped (Milestone 4.5) |
| `eta0` | the Clenshaw-Curtis grid scale, the closed-form diagonal integral and the quadrature's error against it, the nested half/quarter-grid error estimate, and the norms, singular values and condition of the resulting zeroth moment |
| `dd_moments` | per-order Frobenius and maximum norms of the density-density moments |
| `se_moments` | the same for the hole and particle self-energy moments, plus `streaming_vs_staged_max_abs` |
| `realization` | per-order absolute and relative Frobenius and maximum-norm errors between the moments handed to `MBLSE` and the moments its output actually reproduces, per sector, from Dyson's `moment_errors` |
| `moment_error` | momentGW's own scaled error between input and realized self-energy moments |
| `green_function` | pole count, chemical potential, electron count and the particle-number error |
| `results` | frontier HOMO/LUMO by Aufbau counting over multiplets, with weights, the threshold plateau, and a per-reference-orbital quasiparticle table |
| `timings_seconds` | wall time per stage: integrals, static self-energy, eta0, dd moments, self-energy moments (both paths), Dyson, realization diagnostics |

The arrays themselves — eta0, the moments, the static self-energy, and the Green's function
and self-energy poles and couplings — go to `arrays/<case_id>.npz`, which is **not
committed**. They regenerate from this script, and a comparison reads the norms and spectra
in the JSON, not the raw arrays. Pass `--no-arrays` to skip writing them.

### Two moment paths, deliberately

`gw.kernel` builds the density-density moments one order at a time (`build_nth_dd_moment`);
`build_dd_moments` builds the whole stack through a different recurrence. Both are run and
`se_moments.streaming_vs_staged_max_abs` records the largest disagreement between the
self-energy moments they produce. They are equal in exact arithmetic, so this is a direct
measure of how much the raw monomial recurrences amplify roundoff — the thing Milestone 3
has to control. The **streaming** path feeds the Dyson solve, because that is what
`gw.kernel` does in production.

## The set

Four GW100 molecules plus one anchor, at `nmom_max` 1, 3, 5 and 7, from both a
Hartree-Fock and a PBE starting point. Water is additionally run with auxiliary
compression disabled. See [`systems.py`](systems.py) for the geometries and why each system
is in the set.

Odd `nmom_max` only: the `MBLSE` construction conserves `2 * iteration + 2` moments, so an
even value supplies a moment the recurrence has no block for. Milestone 1.3 is where that
becomes an explicit, reported constraint rather than a convention.

Geometries, basis, auxiliary basis and SCF tolerance are those of the `gw100` set in
[`mkakcl/molecular-mGW-testing`](https://github.com/mkakcl/molecular-mGW-testing), so
these cases are directly comparable to the results that harness has already recorded for
the same systems. The frontier readout in [`frontier.py`](frontier.py) follows its
`spectrum.py` for the same reason — including the 0.1 quasiparticle weight threshold, which
that repository documents at length after a lower value picked a satellite as the LUMO and
moved a reported gap by 3 eV.

`hydrogen-631g` is the exception: it is the worked example in the top-level
[`README.md`](../README.md), and reproduces its published value —

```
hydrogen-631g_hf_nmom1_ia    HOMO -16.0474 eV    LUMO 6.5348 eV
```

— which is an anchor recorded before any of this roadmap's work started.

## What this baseline is not

* **Not a convergence claim.** `green_function.converged_flag` is `true` in every record
  because momentGW returns `True` unconditionally for one-shot GW. It is recorded with that
  caveat attached, and Milestone 1.4 replaces it with a numerical convergence result.
* **Not a memory measurement.** `peak_memory_gb` is the process-wide high-water mark, so it
  only rises through a sweep. It bounds what a case needed; it does not measure it.
* **Not an accuracy statement about eta0.** `eta0.nested_error_estimate` is an estimate over
  a sequence of grids, not a bound on the error of the grid in use. Milestone 2 replaces it
  with a certified interval and a scalar error certificate; the number is recorded here
  precisely so that the replacement can be shown to be better.

## Dependency pinning

`pyproject.toml` pins Dyson to an immutable commit rather than `@master`. Two installs of
the same momentGW commit previously resolved to whatever Dyson's default branch happened to
be that day, so no recorded result could name the code that produced it.

The pin is currently `mkakcl/dyson@ca60fe8`, which carries the corrected moment-error
diagnostic (`mkakcl/dyson#1`) that upstream master does not yet have. This baseline is
recorded against a diagnostic that reports the error over all conserved orders; before that
fix the comparison silently dropped the two newest moments and returned exactly zero at the
first iteration. Move the pin to a `BoothGroup/dyson` commit once the Milestone 1 work is
upstreamed and accepted, and re-record.
