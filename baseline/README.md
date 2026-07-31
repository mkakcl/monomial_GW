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

### Two quantities do not reproduce, and one global tolerance would hide it

Each compared quantity carries its own tolerance in [`check.py`](check.py), because a
single tolerance is either too loose to catch a real change or flags every case. Rerunning
the recorded set on an unchanged code base found two quantities that move on their own:

* **The Clenshaw-Curtis grid scale moves by ~10%.** It is chosen by minimising the error of
  the one integral known in closed form, and that objective is flat to machine precision
  over a wide range of scales — the recorded `diagonal_error` is `0.0` exactly on several
  cases. The minimiser is therefore picking an arbitrary point on a plateau, and last-bit
  differences in the mean field, from non-deterministic summation order in threaded BLAS,
  move it. It changes the answer by ~1e-14 eV, so it is harmless here; it is worth knowing
  before Milestone 2 replaces the grid, because it means the recorded scale carries no
  information and a plateau is not a well-posed thing to be selecting on.
* **The chemical potential moves by ~1e-6 relative.** It is located by a search with
  `conv_tol_nelec = 1e-6`, so it is reproducible only to its own convergence tolerance.
  That is expected, and the tolerance says so explicitly rather than by accident.

A third quantity appeared to move and did not. `eta0`'s largest matrix element came out 5%
apart between two identical runs of H2/cc-pVDZ — while the Frobenius norm agreed to twelve
decimal places, the singular values to 5e-16, and the error against the dense oracle was
4e-16 in one run and 9e-16 in the other. Both are correct; they are the same operator in a
different basis. H2 has a doubly degenerate singular value, and PySCF may return any basis
within a degenerate orbital subspace. Only gauge-invariant quantities — norms, singular
values, eigenvalues, ranks, residual ratios, energies — can be compared, so the checker
compares eta0's singular values and not its largest element. `max_abs` is still recorded,
because it is a useful scale to read; it just cannot gate anything.

The first of the two real ones is not self-contained. Rerunning the full set showed the plateau
propagating: a grid scale ~10% away gives an eta0 differing at its own quadrature-error
level, that difference carries through the moments, and the quasiparticle energies come
out up to **1e-9 Ha** apart — 8e-11 eV in the frontier energies, and larger in absolute
terms for the deep core states, which sit near −20 Ha and accumulate more absolute
roundoff. So the reproducibility floor of the whole calculation is set by a free parameter
being chosen on a flat objective, not by arithmetic. That is well below anything physically
meaningful and no cause for alarm, but it does mean `DETERMINISTIC` is 1e-8 relative rather
than machine precision, and it is the reason why.

Everything else — the moment norms, the realization residuals, the auxiliary rank —
reproduces to the last few bits.

### It survived a change of machine

This set was re-recorded on different hardware from the one it was first taken on: Linux
with reference BLAS/LAPACK 3.9.0 and PySCF 2.13.1, to macOS with Apple Accelerate and PySCF
2.14.0. Running `check` across that move *before* changing anything else left 46 of the 52
cases inside their own tolerances, and the six that moved did so only in the quantities named
above — the grid-scale plateau on the four H2/HF cases at rel 0.53 against a 0.5 tolerance,
one realization residual at 3.3e-11 against a 1e-11 floor, and one set of QP energies at rel
1.5e-8 against a 1e-8 floor. The largest frontier shift anywhere in the sweep was 1e-9 eV.

So the tolerances are marginally tight rather than wrong, and swapping the BLAS and a PySCF
minor version does not move this calculation by anything physically meaningful. That is worth
having measured: it means a future disagreement can be attributed to a code change without
first having to rule out the machine. The published `hydrogen-631g` anchor below reproduces
to all four recorded decimal places across the move.

## What is recorded

One JSON file per case in [`data/`](data), plus `index.json` for the sweep. Each record
carries every item the roadmap's *definition of a trustworthy calculation* asks for, except
where the code cannot currently supply it — those are recorded as `null` with the reason,
rather than filled in with a proxy:

| Field | What it is |
|---|---|
| `provenance` | momentGW and Dyson commits (with the checkout path and dirty flag), PySCF/NumPy/SciPy versions, the BLAS NumPy was built against, host and thread environment. A dependency installed from its pinned git URL has no repository to interrogate, so its commit is read from what pip recorded in `direct_url.json` and marked `"source": "pip direct_url"` |
| `mean_field` | starting point, basis, auxiliary basis, SCF tolerance, total energy, and the **mean-field** HOMO–LUMO gap — the quantity that sets the smallest particle-hole denominator entering the dRPA integrand |
| `auxiliary` | `naux_full`, `naux`, and whether compression fired. `discarded_norm` is `null`: the compression selects on an absolute eigenvalue cutoff and never reports what it dropped (Milestone 4.5) |
| `eta0` | the Clenshaw-Curtis grid scale, the closed-form diagonal integral and the quadrature's error against it, the nested half/quarter-grid error estimate, the norms, singular values and condition of the resulting zeroth moment, and — under `oracle` — its **true** error against a dense eigendecomposition, with the spectrum and condition number of `Mtilde` |
| `dd_moments` | per-order Frobenius and maximum norms of the density-density moments |
| `se_moments` | the same for the hole and particle self-energy moments, plus `streaming_vs_staged_max_abs` |
| `realization` | per sector: the number of moments supplied, the order that was **requested** and the order that was **achieved** (`max_cycle_achieved`, `order_reduced`, and the moment counts for both), per-order norms of the moments the realized self-energy actually carries, and the per-order absolute and relative Frobenius and maximum-norm errors against the moments it was handed, from Dyson's `moment_errors`. Everything but the requested order is read at the achieved one, because that is the realization the Dyson solve actually used |
| `moment_error` | momentGW's own scaled error between input and realized self-energy moments |
| `green_function` | pole count, chemical potential, electron count and the particle-number error |
| `results` | frontier HOMO/LUMO by Aufbau counting over multiplets, with weights, the threshold plateau, and a per-reference-orbital quasiparticle table |
| `timings_seconds` | wall time per stage: integrals, static self-energy, eta0, dd moments, self-energy moments (both paths), Dyson, realization diagnostics — read next to `load_average` and `cpu_count` |

The arrays themselves — eta0, the input and reconstructed moments, the static self-energy,
and the Green's function and self-energy poles and couplings — go to `arrays/<case_id>.npz`,
which is **not committed**. They regenerate from this script, and a comparison reads the norms and spectra
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
Hartree-Fock and a PBE starting point. See [`systems.py`](systems.py) for the geometries
and why each system is in the set.

Lithium-hydride and water are additionally run with auxiliary compression disabled. Two
systems, not one, because the recorded data shows the criterion doing nothing at all on
water: with the `"ia"` metric the compression selects on the eigenvalues of a Gram matrix
whose rank cannot exceed the number of particle-hole pairs, so at a `1e-10` tolerance it
removes the null directions and nothing else. Lithium-hydride has 60 auxiliary functions
and 34 particle-hole pairs and compresses to 34; water has 71 and 95 and compresses to 71,
which is to say not at all. Only running the pair on water would have recorded a
compressed/uncompressed comparison in which the two sides were the same calculation.

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
* **Not a performance measurement.** The stage timings say which stage dominates, which is
  what Milestone 4 needs before it can choose what to optimise. They are wall clock on a
  shared workstation: the recorded set was taken at a load average of 12–19 on 12 cores,
  and the same ozone case timed 4x faster on an idle machine. `load_average` is recorded
  per case so this is visible rather than assumed. Any *comparison* of timings needs a
  quiet machine and the provenance the roadmap asks for.
## The eta0 error the code reports is not the eta0 error

`eta0.nested_error_estimate` is what momentGW prints as "Error in integral". It is an
extrapolation across coarser grids, not a bound, and the oracle shows how far off it is at
the default `npoints = 48`:

| case | `Mtilde` condition | true relative error | reported estimate |
|---|---:|---:|---:|
| `hydrogen-631g` HF | 5.6 | 5.1e-16 | 4.4e-4 |
| `water` PBE | 6.3e3 | 2.8e-12 | 2.1e-3 |
| `ozone` PBE | 6.7e4 | 1.2e-9 | 4.5e-3 |

Six to twelve orders of magnitude, always pessimistic. Two things follow, and they pull in
opposite directions:

* The Clenshaw-Curtis eta0 is **much more accurate than advertised** on these systems, and
  saturates at the floating-point floor by `npoints = 96`. Milestone 2 should not expect to
  win on accuracy here. Its case is cost — 48 quadrature points, each an `naux`-cubed
  inverse — and having a certificate at all.
* The true error nonetheless tracks the conditioning of `Mtilde`, rising from 5e-16 to
  1e-9 as the condition number rises from 5.6 to 6.7e4. A stiffer system will eventually
  need more points, and **nothing in the current code could tell you that it had**, because
  the one number reported is uninformative in both directions. That is the argument for
  Milestone 2's certified interval, and it is now measured rather than asserted.

## Three cases could not support the order they were asked for

At `nmom_max = 7`, Dyson now stops the recurrence when an iteration produces an
off-diagonal square with no real square root, and solves at the last order it completed.
It fires on three of the fifty-two cases, always in one sector only:

| case | sector | moments conserved | Dyson's own residual | momentGW `moment_error` |
|---|---|---:|---:|---:|
| `lithium-hydride` HF | hole | 8 → **6** | 5.5e-13 → 2.2e-14 | 3.6e-14 → **8.0e-8** |
| `lithium-hydride` PBE | hole | 8 → **6** | 1.9e-13 → 1.6e-14 | 2.4e-14 → **6.1e-7** |
| `hydrogen-631g` HF | particle | 8 → **6** | 1.8e-14 → 9.4e-15 | 2.9e-14 → 2.6e-14 |

The two columns move in opposite directions, and that is the point. Dyson's residual is
measured over the moments it undertook to conserve, so realizing six it can support instead
of eight it cannot *improves* it by an order of magnitude. momentGW's `moment_error`
compares against all eight moments it supplied, so it rises to 1e-7 — the honest size of two
undelivered moments.

The previous baseline recorded ~1e-14 for both. That number was not an accuracy: the
recurrence was clipping a direction with no square root and reporting agreement over moments
it was not conserving. Nothing about the calculation got worse here; a misreported quantity
started being reported.

The physical effect is small — the HOMO moves 5.5e-7 eV on lithium-hydride HF and 1.6e-6 eV
on PBE, and the particle number is unchanged to five figures — but the self-energy is a
visibly smaller object, 133 poles rather than 152 for lithium-hydride and 28 rather than 32
for the anchor. Water and ozone support order 7 in both sectors and are untouched.

`check.py` compares `max_cycle_achieved` exactly, so a case that silently starts or stops
stepping down fails the check rather than hiding inside a residual that got smaller.

## Dependency pinning

`pyproject.toml` pins Dyson to an immutable commit rather than `@master`. Two installs of
the same momentGW commit previously resolved to whatever Dyson's default branch happened to
be that day, so no recorded result could name the code that produced it.

The pin is currently `mkakcl/dyson@054d4b5`, which carries the Milestone 1 realization work
that upstream master does not have: the corrected moment-error diagnostic (`mkakcl/dyson#1`),
the scale-aware `matrix_power` support policy (`#2`), and the feasibility validation and
order step-down (`#3`) described above. Before `#1` the error comparison silently dropped the
two newest moments and returned exactly zero at the first iteration.

Pinning by URL created a second problem, which `run.py` now handles: a dependency installed
from `git+...@<sha>` unpacks into `site-packages` with no `.git` beside it, so the commit
cannot be read back out of a repository. It is recovered from pip's `direct_url.json`
instead. Without that, honouring the pin is exactly the case in which a record cannot name
the Dyson revision that produced it.

This is still the fork. Move the pin to a `BoothGroup/dyson` commit once the Milestone 1 work
is upstreamed and accepted, and re-record.
