# Molecular G0W0/dRPA stability and performance roadmap

## Purpose

This roadmap covers the restricted molecular `G0W0` path with
`polarizability="drpa"`. Its first objective is trustworthy higher-order moment
calculations. Performance work follows only after the numerical error at each stage
can be measured and attributed.

The immediate sequence is:

1. repair the Dyson realization and its diagnostics;
2. replace the zeroth RPA moment integration with a stable HHT/Zolotarev
   inverse-square-root action;
3. control error amplification and conditioning in higher-order moments;
4. optimize the measured bottlenecks;
5. investigate more invasive low-scaling or alternative constructions.

Unrestricted and periodic implementations are deliberately deferred until the
restricted molecular path has passed all acceptance gates. Their spin, complex
conjugation, momentum, and normalization conventions require separate validation.

## Guiding rules

- Never report convergence solely because a linear algebra routine returned.
- Never silently discard a moment direction without reporting its scale, rank loss,
  and effect on reconstructed moments.
- Use theorem-backed spectral enclosures for methods whose accuracy depends on an
  interval. Lanczos Ritz values may be diagnostics, but not certified endpoints.
- Keep the existing Clenshaw-Curtis implementation as an independent reference until
  the new route has passed downstream G0W0 and MPI validation.
- Judge approximations by delivered moment and quasiparticle errors, not by proxy
  thresholds alone.
- Separate numerical errors due to eta0, auxiliary compression, moment truncation,
  realization, and particle-number fitting.
- Prefer deterministic operation/shape checks for scaling tests. Treat wall-clock
  measurements as benchmarks with recorded hardware and software provenance.

## Status legend

- **Planned**: agreed work that has not started.
- **In progress**: implementation exists on a working branch.
- **Gated**: implemented, but not yet accepted as a default.
- **Complete**: all listed acceptance criteria pass.
- **Deferred**: intentionally outside the current restricted molecular scope.

## Milestone 0 - Reproducible baseline

**Status: Complete - 52 cases recorded in [`baseline/`](baseline/README.md), and
reproduced 52/52 by `python -m baseline.check` at each quantity's own tolerance.**

### Work

- [x] Record a compact baseline set of restricted molecular G0W0/dRPA calculations:
  H2, H2O, LiH, and at least one small-gap molecule.
- [x] Save eta0, density-response moments, hole and particle self-energy moments,
  reconstructed moments, frontier QP energies, particle-number error, auxiliary rank,
  and stage timings.
- [x] Include `nmom_max=1, 3, 5, 7` where numerically feasible.
- [x] Run compressed and uncompressed auxiliary spaces for at least one molecule.
- [x] Record the exact momentGW, Dyson, PySCF, NumPy, SciPy, and BLAS versions.
- [x] Correct `[tools.setuptools.dynamic]` to `[tool.setuptools.dynamic]` so builds
  publish the package version correctly.
- [x] Stop installing mutable `dyson@master`; pin the currently tested commit, then
  update that pin only when Milestone 1 is accepted.

### Acceptance gate

- The baseline is reproducible from a fresh environment.
- Every recorded result identifies its code revisions and numerical options.
- Existing restricted molecular G0W0/dRPA regression values remain unchanged.

## Milestone 1 - Repair Dyson realization and error reporting

**Status: Complete - 1.1 to 1.4 implemented in `mkakcl/dyson` and consumed here through the
pin. The Milestone 1 work was accepted at `73cd18d`; the pin has since moved to `3ebd156`
to carry Milestone 4 work, which changes nothing in this section.**

Dyson is an external dependency, so these changes are made in `mkakcl/dyson`, tested there,
and consumed here through an immutable commit. A private runtime patch in momentGW is not
an acceptable final solution.

**Acceptance is that pin, not a merge into `BoothGroup/dyson`** (decided 2026-08-03).
An earlier version of this section said the work had to be upstreamed before the milestone
could close, and that was what "the accepted Dyson commit" in 1.4 meant. It no longer is: a
commit on the fork that the baseline has been recorded against is the accepted state, and
the milestone closes on it. Nothing here is a licence to push to `BoothGroup/dyson` - see
[`CLAUDE.md`](CLAUDE.md).

The baseline has been re-recorded against the pin three times: at `054d4b5`, where three
cases began stepping down from the order they were asked for; at `73cd18d`, which moved
nothing at all; and at `3ebd156`, which moved the frontier by 1e-15 to 2e-13 eV and
improved the realization residuals by 4x to 26x, both by removing error rather than
introducing it. See [`baseline/README.md`](baseline/README.md).

### 1.1 Reconstructed-moment diagnostic

- [x] Fix `MBLSE.reconstruct_moments`: the current implementation reconstructs
  `range(2 * iteration)`, while the error routine compares against
  `2 * iteration + 2` input moments. Python `zip` silently omits the newest two
  moments, and iteration zero checks none.
- [x] Assert equal predicted and reference moment counts before computing an error.
- [x] Report per-order absolute and relative Frobenius and maximum-norm errors.
- [x] Separate errors before and after any intentional chemical-potential pole shift.

### 1.2 Matrix square root and inverse square root

- [x] Replace the fixed absolute `1e-10` eigenvalue cutoff with a scale-aware
  `atol + rtol * lambda_max` policy.
- [x] Treat the square root and inverse square root with one consistent effective
  support.
- [x] Clip only negative eigenvalues that are demonstrably compatible with roundoff.
  Fail on materially negative directions.
- [x] Report the minimum eigenvalue, condition estimate, effective rank, discarded
  norm, and the resulting reconstructed-moment error.
- [x] Do not describe the norm of discarded original eigenvalues as the error in an
  inverse square root.

### 1.3 Realization feasibility

- [x] Validate finite, Hermitian moments before starting the recurrence.
- [x] Validate positive semidefiniteness and causality where required by the measure.
- [x] When the requested order is not supportable, step down to the largest order that
  satisfies a delivered-moment residual and report the reduction.
- [x] Validate moment-order parity. Either require odd `nmom_max` for the current MBLSE
  construction or define and report exactly which supplied moments are used.
- [x] Avoid repeated intermediate diagonalizations used only for diagnostics unless
  error reporting is enabled.

### 1.4 momentGW integration

- [x] Pass explicit Dyson numerical options from `momentGW/gw.py`. `MBLSE._options` was
  `{calculate_errors, force_orthogonality, max_cycle, hermitian}` and `set_options` raises
  on anything else, so the scale-aware tolerances from 1.2 could not be stated and every
  `matrix_power` inside the recurrence took the library defaults. `mkakcl/dyson#4` widened
  both layers - `matrix_power` now forwards `atol`, `rtol`, `neg_atol` and `neg_rtol`, and
  the solvers accept them as ordinary options reaching all 15 call sites - so `dyson_opts`
  now states all seven. The four tolerances are set to the Dyson defaults: the point is
  that they are written down here rather than inherited, so a change of pin cannot move
  them silently.
- [x] Replace unconditional single-shot `conv=True` with a numerical convergence
  result that includes realization and particle-number gates.
- [x] Store structured diagnostics on the GW object instead of only writing log text.
  `gw.dyson_diagnostics` carries them and `gw.dyson_solvers` retains the solvers, so
  `baseline/run.py` no longer rebuilds a second pair to read them off.
- [x] Pin momentGW to the accepted Dyson commit: `mkakcl/dyson@73cd18d`. The fork is the
  accepted state, so this is the terminal pin rather than a placeholder for an upstream
  one. Moving it re-recorded the baseline, which moved nothing: 52/52 unchanged. The pin is
  now at `3ebd156` for the Milestone 4 work; it carries everything accepted here unchanged.

Restricted molecular only. The unrestricted and periodic solvers share this kernel but
override `solve_dyson` without gating it, and keep the unconditional flag until they do.

### Acceptance gate

All criteria pass. The tests named are in `mkakcl/dyson`, where the whole suite is 915
passed, 40 skipped at the pinned commit.

- **Passed** - synthetic positive matrix measures reconstruct every promised moment
  through the requested order (`tests/test_mbl_moment_errors.py`).
- **Passed** - a test fails if predicted and reference moment counts differ;
  `moment_errors` raises rather than comparing a truncated pair.
- **Passed** - small positive support directions are either preserved or explicitly
  rejected by a scale-aware rule with a measurable moment effect
  (`tests/test_matrix_power.py`, and `tests/test_mbl_tolerance_options.py` where a loosened
  `rtol` is shown to degrade the conserved moments).
- **Passed** - materially non-PSD input fails loudly (`tests/test_mbl_realization.py`).
- **Passed** - requested and achieved moment orders are both reported, by the solver and
  on `gw.dyson_diagnostics`.
- **Passed** - existing low-order G0W0 QP energies remain unchanged within the baseline
  tolerance: 52/52 unchanged across both pin moves.

## Milestone 2 - Stable eta0 through HHT/Zolotarev

**Status: Complete - `eta0_method="hht"` is the default for the restricted molecular dRPA
path as of 2026-08-04. Clenshaw-Curtis is retained, unchanged, as the independent
reference the guiding rules ask for, and remains the default for the unrestricted and
periodic solvers, which reject an explicit `"hht"`. The serial/multi-rank MPI agreement
gate was waived unmeasured; every other criterion passed.**

Flipping the default moved one of the 52 baseline cases: `ozone_pbe_nmom7_ia`, the
small-gap system at the highest moment order, by 7.4e-9 eV in the HOMO and 5.7e-7 Ha
across all quasiparticle energies. That movement is Clenshaw-Curtis error being removed,
not introduced. Against a dense eigendecomposition of `Mtilde` for that system
(condition 6.7e4), the two routes measure:

| route | relative error against the dense oracle |
| --- | --- |
| `clencur`, 48 points | 1.151e-09 |
| `hht`, 26 poles | 2.100e-13 |

The two routes differ from each other by 1.151e-09 - that is, the whole of the difference
is the legacy route's error, and the baseline had been recording it. HHT is also the
cheaper of the two: 20 poles against 48 quadrature points on benzene/cc-pVDZ, 1.82x
faster on the eta0 stage alone.

The new method computes the same projected zeroth dRPA moment as the current
Clenshaw-Curtis integral:

```text
eta0 V = D^(1/2) Mtilde^(-1/2) W
Mtilde = D^2 + c W W^T
W = D^(1/2) V
```

For restricted spatial-orbital RI integrals, `c = 4`. Only the RI-projected result is
formed; no particle-hole squared matrix is allowed.

### 2.1 Public API and code layout

- [x] Add a small eta0 module containing pole generation, certified bounds, scalar
  error evaluation, and the projected inverse-square-root action. The scalar layer is
  `momentGW/eta0.py`; the projected action lives beside the MPI plumbing it needs, as
  `dRPA._hht_apply` in `momentGW/rpa.py`.
- [x] Add options equivalent to:
  `eta0_method={"clencur", "hht"}`, `eta0_tol`, optional `eta0_n_poles`, and
  `eta0_check_refinement`. Restricted molecular only: the unrestricted and periodic
  solvers override the zeroth-moment build wholesale, so they refuse
  `eta0_method="hht"` at construction rather than silently ignoring it.
- [x] Preserve the meaning of the existing `npoints` option for the legacy route. It
  is only read by the Clenshaw-Curtis path, and only shown in the options header when
  that path is selected.
- [x] Introduce HHT as opt-in and retain Clenshaw-Curtis as the shadow/reference
  calculation during rollout.

### 2.2 Stable coefficients

- [x] Replace the direct `ellipk(1 - k2)`/real-Jacobi construction when the condition
  number makes it unstable. Both `K'` and the Jacobi functions are evaluated through
  the complementary parameter directly - `K' = pi / (2 AGM(1, k))` and a
  descending-Landen recurrence taking `emmc = k^2 = lmin/lmax` - so `1 - k^2` is
  never formed.
- [x] Use stable complementary-parameter/nome evaluation or generate the small number
  of poles and weights in extended precision before casting to float64. Generation is
  in `numpy.longdouble` (80-bit on the development machine) and cast at the end.
- [x] Assert positive, finite shifts and weights.
- [x] Add explicit handling for zero coupling, an empty particle-hole space, and a
  degenerate spectral interval.

### 2.3 Certified spectral interval

- [x] Require finite, strictly positive particle-hole gaps.
- [x] Use the rigorous distributed enclosure
  `lambda_min = min(D)^2` and
  `lambda_max = max(D^2) + c * min(||W||_F^2, ||W||_1 ||W||_inf)`.
- [x] Pad the lower endpoint outward/down and the upper endpoint outward/up for
  floating-point arithmetic. The padding is a deliberate 1e-8 relative on each side:
  the pole count is logarithmic in the interval, so generosity costs nothing, while a
  few-ulp pad would require an argument about every upstream rounding.
- [x] Use Lanczos only to report how loose the rigorous upper bound is. Implemented
  as a fixed-iteration power estimate (`lambda_max_estimate`, a lower bound on the
  true `lambda_max`), recorded with the ratio `upper_bound_looseness`; it plays no
  role in the certificate. Measured looseness on water is 1.16x.
- [x] Report the interval and condition number in the calculation diagnostics.
  `gw.eta0_diagnostics` carries interval, condition number, norm bounds, pole count,
  scalar error, and per-pole solve residuals.

### 2.4 Accuracy certificate

- [x] Replace the asymptotic `required_poles + safety` rule with selection based on
  the scalar relative error
  `max |1 - sqrt(x) r_N(x)|` over the certified interval. The asymptotic rate
  (`2 pi^2 / (log kappa + 6)`, measured conservative against fitted decay exponents
  from condition 1e2 to 1e16) only seeds the search; the measured supremum in
  extended precision decides, walking the pole count down to the smallest that
  passes.
- [x] Validate against a high-precision scalar oracle. The Zolotarev equioscillation
  formula was not needed: the certificate is the measured error itself, and the tests
  check it pointwise against an extended-precision oracle on an independent random
  grid. A tolerance below the float64 floor, or beyond the pole-count cap, fails
  explicitly.
- [x] Check residuals of every auxiliary-space Cholesky solve.
- [x] Retain `N_p` versus `N_p + 4` as a secondary regression signal, not the sole
  accuracy certificate (`eta0_check_refinement`).
- [x] Derive the default eta0 tolerance from the higher-moment error budget in
  Milestone 3 rather than fixing it permanently at machine precision. Settled
  2026-08-03: 1e-14 is accepted as the standing default on the measurement below,
  rather than held back for the combined budget. The eta0 leg of that budget is now
  measured (2026-07-31, water/HF, LiH/HF, ozone/PBE at `nmom_max = 7`, dense-oracle
  reference, HHT variants at scalar errors 1e-2 down to the floor): the dd-moment
  recurrence does *not* amplify an eta0 perturbation - every order through 6 moves
  by ~1x the scalar error - and the frontier QP energies move by roughly 30-300x
  the scalar error in eV. Below a scalar error of ~1e-13 the float64 kernel
  arithmetic floor (~3e-15 relative on eta0, ~1e-11 eV on frontier energies) takes
  over, so tolerances below 1e-14 buy nothing, and 1e-14 already holds the QP
  contribution far below the baseline's 1e-9 eV reproducibility floor. What remains
  for Milestone 3 is combining this leg with realization and compression
  contributions, not re-measuring it. The measurement is
  [`baseline/studies/eta0_amplification.py`](baseline/studies/eta0_amplification.py).

### 2.5 Projected kernel and MPI

- [x] Cache `D`, `D^2`, bounds, poles, and weights across the per-pole loop; whatever
  further caching pays (contiguous transposes, batched Grams) is Milestone 4's
  profiling question.
- [x] Build one local weighted Gram per pole and all-reduce only the auxiliary-space
  Gram.
- [x] Use Cholesky factorization and solves; never form an explicit inverse.
- [x] Remove the avoidable bare-`Lia`/`-I` cancellation from the HHT path. The
  rational sum absorbs the bare term: at zero coupling each solve is the identity and
  the sum reproduces `Lia` directly.
- [x] Assert the expected local output shape and prohibit particle-hole squared
  intermediates. Every intermediate's shape is recorded in the diagnostics, and a
  test asserts none has two particle-hole dimensions.

### Acceptance gate

- **Passed** - scalar tests cover well-conditioned intervals through extreme
  condition numbers, with a high-precision oracle and an explicit failure when the
  requested tolerance is not representable (`tests/test_eta0.py`).
- **Passed** - dense small-matrix tests compare against an eigendecomposition of
  `Mtilde`.
- **Passed** - H2 and H2O eta0 agree with the legacy route and dense oracle to the
  predicted floating-point limit; ozone/PBE (condition ~1e5, where the legacy true
  error is ~1e-9) holds its certificate as the small-gap case.
- **Passed** - density-response moments, self-energy moments, and final QP energies
  are invariant within their error budgets (water, `nmom_max=3`, to 1e-10).
- **Waived, unmeasured** - serial and multi-rank MPI results agree. The development
  machine has no `mpi4py`, and the milestone was accepted without this gate on
  2026-08-03. What can be said from the code is that the kernel all-reduces only the
  auxiliary-space Gram and otherwise touches nothing outside a rank's own
  particle-hole slice - an argument, not a measurement. Worth running if a multi-rank
  machine becomes available; until then no multi-rank HHT result has been checked
  against a serial one.
- **Passed** - compression on/off, frozen-core, zero-coupling, invalid-gap, and
  small-gap cases are covered.
- **Passed** - shape instrumentation proves that no particle-hole squared matrix is
  formed.
- **Passed** - HHT becomes the default only after all gates pass. Flipped on
  2026-08-04, with the MPI gate waived rather than passed. Legacy code is not removed:
  the guiding rules keep Clenshaw-Curtis as an independent reference, and it stays the
  default for the unrestricted and periodic solvers.

## Milestone 3 - Higher-order moment stability and propagated errors

**Status: Planned - third implementation milestone**

### 3.1 End-to-end error budget

- [ ] Track separate contributions from eta0 approximation, Cholesky solves, auxiliary
  compression, response recurrence, self-energy convolution, MBLSE realization, and
  particle-number fitting.
- [ ] Derive or measure the amplification of an eta0 perturbation at each requested
  response and self-energy moment order.
- [ ] Select the eta0 tolerance and pole count from the requested final moment/QP
  tolerance.
- [ ] Report errors per sector and order; do not reduce all information to one scalar.

### 3.2 Scaled or modified moments

- [ ] Affinely center and scale the hole and particle spectral sectors separately so
  their supports are order one.
- [ ] Transform raw monomial moments to the scaled basis before realization and
  transform poles back afterward.
- [ ] Evaluate a direct Chebyshev/modified-moment plus block-Jacobi backend for orders
  where raw monomial Hankel matrices lose usable precision.
- [ ] Keep the raw monomial backend as an oracle at low order during migration.

### 3.3 Adaptive moment order

- [ ] Increase only through supported odd orders and compare `m` with `m + 2`.
- [ ] Converge requested frontier QP energies, reconstructed moments, particle number,
  and positive spectral weight.
- [ ] Step down automatically when the next block fails the delivered-moment or PSD
  gate.
- [ ] Report requested, built, conserved, and realized orders separately.

### 3.4 Closure and spectral diagnostics

- [ ] Compare independent admissible closures, such as Gauss and Radau, as a
  truncation-error indicator.
- [ ] Do not label closure spread a rigorous QP bound without a supporting theorem.
- [ ] Validate frontier IP/EA separately from deep-state largest-overlap labels.
- [ ] Add a spectral-function comparison against an untruncated small-system oracle
  where satellites or deep states are reported.

### Acceptance gate

- Increasing supported moment order reduces the stated frontier error until the
  predicted floating-point or realization limit.
- No calculation silently continues after a PSD, rank, causality, or residual failure.
- The reported eta0 and realization bounds are consistent with observed moment errors.
- H2O, LiH, and the selected small-gap system have documented order-convergence tables.
- Low-order results remain compatible with the Milestone 0 baseline.

## Milestone 4 - Optimize the verified restricted molecular path

**Status: In progress - the external-orbital loop is batched; everything else is planned.
Measured on benzene, PBE starting point, `nmom_max = 3` unless stated, against the
profiled current code.**

### What the profile actually says

Before optimising further, where the time goes on benzene/cc-pVDZ, `nmom_max = 3`
(12.0 s total): the eta0 stage is 34%, and the external-orbital rotation in
`build_se_moments` is a further ~23%. `build_dd_moments` is already two BLAS-3 GEMMs per
order with nothing to recover, and `convolve` is 1%. Two changes have been taken, both
constant-factor and neither touching how a moment is defined: making HHT the default
(Milestone 2) and batching the loop below. Together they are **1.46x** on
benzene/cc-pVDZ at `nmom_max = 3`, **1.51x** at `nmom_max = 7`, and **1.41x** on
cc-pVTZ.

**That paragraph profiled the stages this milestone was already looking at, and missed the
largest one.** The Dyson stage was never in it, and it was 35-46% of a run. The item below
takes it to 17-27%, after which the correlated moment construction is the majority again
and is the stage to profile next. Two lessons worth keeping, both in
[`DIAGONALISATION_ROADMAP.md`](DIAGONALISATION_ROADMAP.md): a stage absent from the profile
is not a stage that is cheap, and the `O(N^3)`-versus-`O(N^4)` argument for ignoring the
Dyson solve holds only for growing the molecule - at fixed molecule and growing basis
`nocc` is constant, both scale as `O(N^3)`, and the Dyson share *rises* with basis size.

Two measurements worth keeping, because they contradict assumptions written elsewhere in
this roadmap:

- **"Never form an explicit inverse" is an accuracy rule, not a speed rule.** At
  `naux = 551`, `nov = 1953`, `cho_factor` plus `cho_solve` costs 30.2 ms against 28.6 ms
  for `inv` plus a GEMM: both are dominated by the `naux^2 nov` apply, and the triangular
  solves are slower per flop than the GEMM. Cholesky wins below roughly `nov = 2 naux` and
  LU above `nov = 4 naux`. Keep the rule for its numerical merits, but do not expect it to
  pay.
- **Reassociating the back half is not verifiable at high moment order.** The batched loop
  below performs identical arithmetic in a different summation order. The resulting shift
  in the quasiparticle energies, on benzene/cc-pVDZ, is 6.5e-13 eV at `nmom_max = 1` and
  5.2e-8 eV at `nmom_max = 7` - while the HOMO stays at ~1e-13 eV throughout. The frontier
  is well conditioned and the deep states are not, which is Milestone 3.2 and 3.4 exactly.
  Until those land, an optimisation of this stage can only be validated on the frontier.

### Work

- [x] Batch the external-orbital loop in `momentGW/tda.py` into larger BLAS-3
  contractions. 701 ms to 245 ms per moment order on benzene/cc-pVDZ. Most of that is
  layout rather than batching: batching alone reaches 533 ms (17.9 GFLOP/s), and it takes
  reordering the integrals so the external index leads - so that both operands of the
  batch are contiguous - to reach 245 ms (38.9 GFLOP/s). **Memory:** the reorder is a copy,
  so peak usage carries a second `Lpx` (`naux * nmo * nx` doubles) for the duration of
  `build_se_moments` - 0.06 GB on benzene/cc-pVDZ, 0.36 GB on cc-pVTZ. Emitting that
  layout from `Integrals.transform` would remove the copy entirely, but `Lpx` is shared
  with the unrestricted and periodic solvers and so is deferred to Milestone 6.
- [x] Stop the Dyson stage doing its work twice. Profiling the *stage* rather than its
  eigendecompositions put it at 35-46% of a run, and found two things: `Lehmann.moments`
  contracted the poles with a three-operand `einsum` that `np.einsum` cannot route to a
  `tensordot`, so it ran in the unblocked single-threaded kernel (46% of the stage, and
  21-73x slower than the equivalent GEMMs); and `Spectral` diagonalised the supermatrix
  eagerly, then reconstructed its blocks from that eigendecomposition to recover the static
  part and self-energy it had been built from - a round trip returning its own inputs, 21
  of the 51 eigendecompositions in a G0W0, including one of the two `9 * nmo` solves.
  Delivered as `mkakcl/dyson#6`, pin moved to `3ebd156`. **2.97x on the Dyson stage and
  1.45x on a benzene/cc-pVTZ calculation at `nmom_max = 7`**, with the frontier moving
  1e-15 to 2e-13 eV and the realization residuals improving 4-26x. Full analysis, and what
  is left, in [`DIAGONALISATION_ROADMAP.md`](DIAGONALISATION_ROADMAP.md).
- [ ] Refactor convolution so moment orders accumulate locally and perform one final
  MPI reduction and symmetrization instead of reducing full stacks repeatedly.
- [ ] Batch several HHT Gram reductions or overlap nonblocking reductions with local
  work, subject to memory profiling.
- [x] Add a native no-compression integral-transform path instead of multiplying each
  block by a dense identity rotation. 762 ms to 220 ms on benzene/cc-pVDZ, bit-identical
  output. The identity was not merely a redundant multiply: `rot[b0:b1].T @ block`
  scatters each block up to the full auxiliary height, so an uncompressed run contracted
  every block against an `(naux_full, naux_full)` identity and accumulated the result over
  the whole array. Removing it means each block has to land in the rows it belongs to
  instead, which is the native path this item asked for. Applies to the restricted
  molecular `Integrals` only; the unrestricted, periodic and THC classes override
  `transform` and are untouched.
- [ ] Replace absolute auxiliary compression selection with a scale-aware PSD
  criterion. Report rank, discarded norm, and observed eta0/QP impact.
- [ ] Investigate pivoted Cholesky or randomized compression for large auxiliary
  metrics.
- [ ] Implement a genuinely vector-valued diagonal-self-energy path without allocating
  full MO-by-MO moment arrays.
- [ ] Add a target-orbital/window mode for calculations requiring only frontier states,
  with a buffer and an off-diagonal-coupling diagnostic.
- [ ] Use partial/shift-invert diagonalization when only frontier poles of a large
  compact Hamiltonian are requested.
- [ ] Evaluate GPU execution for batched Gram, Cholesky, and back-half contractions.
  FP64 remains the reference; lower precision requires refinement and residual checks.

### Acceptance gate

- Each optimization has before/after operation counts, peak memory, communication
  volume, and wall-clock measurements with provenance.
- Eta0, moments, QP energies, ranks, and diagnostics remain within the accepted error
  budgets.
- The benchmark identifies the new dominant stage before further optimization begins.

## Milestone 5 - Alternative construction and lower scaling

**Status: Deferred until Milestones 1-4**

### Work

- [ ] Port the eta-free positive-RPA-pole contour first as an independent construction
  oracle, not as a replacement default.
- [ ] Compare the contour and HHT-seeded recurrence using predicted cost, spectral-gap
  feasibility, and roundoff amplification; consider an automatic selector only after
  both have independent gates.
- [ ] Prototype block or multi-shift rational Krylov only for extreme condition numbers
  or aggressively compressed right-hand sides.
- [ ] Investigate separable RI/ISDF/THC together with imaginary-time, Laplace, or Cauchy
  separation. Factorizing the RI vertex alone does not remove the nonseparable
  particle-hole energy denominator.
- [ ] Add locality and sparse pair-domain screening for large molecules.

### Rejected as default eta0 approaches

- Full RPA diagonalization or formation of particle-hole squared matrices.
- Newton-Schulz, Denman-Beavers, or other dense full-matrix inverse-square-root
  iterations that destroy the diagonal-plus-low-rank structure.
- Polynomial inverse-square-root expansions as a general replacement for the
  logarithmically convergent rational rule.
- FEAST or full-spectrum contour eigensolvers when all positive RPA poles are required.

## Milestone 6 - Broaden the supported scope

**Status: Deferred**

- [ ] Derive and validate the unrestricted spin factors and block layout.
- [ ] Derive and validate periodic complex-Hermitian adjoints, k-point normalization,
  and momentum conservation.
- [ ] Extend the accepted diagnostics and error budgets to evGW and self-consistent
  variants only after the one-shot path is stable.
- [ ] Remove legacy Clenshaw-Curtis helpers only when no supported path imports them.

## Scientific validation track

This track runs alongside every milestone rather than at the end.

- [ ] Maintain comparisons against PySCF TD-dRPA or an explicit-pole oracle on small
  systems.
- [ ] Add a documented molecular subset beyond minimal bases, including at least one
  GW100-style system.
- [ ] Converge orbital and auxiliary basis sets separately from algorithmic tolerances.
- [ ] Quantify compression and frozen-core errors in meV for frontier QP energies.
- [ ] Compare at least HF and a suitable hybrid starting point for scientifically
  important results; keep starting-point uncertainty separate from numerical error.
- [ ] Report spectral functions rather than relying only on per-orbital pole assignment
  in dense satellite regions.

## Proposed pull-request sequence

1. **Baseline and dependency control** - fixtures, provenance, packaging metadata, and
   an immutable Dyson pin.
2. **Dyson diagnostic correction** - reconstruction-count tests and accurate per-order
   residuals. Delivered as `mkakcl/dyson#1`.
3. **Dyson support/PSD policy** - scale-aware matrix powers, rank reporting, and
   feasibility gates; update the momentGW pin. Delivered as `mkakcl/dyson#2` and `#3`,
   with `#4` making the support policy statable from `dyson_opts`; accepted at the pin
   `73cd18d`, which has since moved on for Milestone 4.
4. **HHT scalar layer** - stable coefficients, rigorous bounds, exact scalar error
   checks, and high-precision tests. Delivered together with 5 on the
   `m2-hht-eta0` branch.
5. **HHT projected eta0** - restricted molecular kernel, MPI path, and dense/legacy
   equivalence tests behind an opt-in flag. Delivered together with 4; the
   multi-rank MPI run is the piece still owed.
6. **HHT default** - downstream G0W0 invariance, small-gap validation and documentation
   delivered 2026-08-04. Clenshaw-Curtis-specific options are *not* deprecated: the
   legacy route stays as the independent reference and as the unrestricted and periodic
   default, so `npoints` keeps its meaning.
7. **Higher-order stabilization** - error propagation, scaled moments, adaptive order,
   and realization gates.
8. **Back-half and communication optimization** - only after profiling the accepted
   stable path.
9. **Independent contour oracle and low-scaling research** - separate experimental
   work with no effect on the default path until gated.

## Definition of a trustworthy calculation

A restricted molecular G0W0/dRPA calculation is considered numerically trustworthy
only when it reports:

- the exact software revisions and major numerical options;
- the auxiliary compression rank and discarded-error measure;
- eta0 method, certified spectral interval, condition number, pole count, scalar
  approximation error, and solve residual;
- requested, built, conserved, and realized moment orders;
- per-sector reconstructed-moment residuals and effective realization rank;
- Hermiticity, PSD/causality, and positive-residue diagnostics;
- particle-number error and chemical potential;
- frontier QP convergence with respect to moment order;
- stage timings and peak-memory information.

