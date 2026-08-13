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

- [x] Track separate contributions. `gw.error_budget()` gathers every source that is
  measured anywhere - eta0, its Cholesky solves, the MBLSE realization per sector and
  order, particle-number fitting, and moment truncation - into one report, each with its
  native value and unit, and converts to eV on the frontier only where an amplification
  has been measured. Auxiliary compression, the response recurrence and the self-energy
  convolution are *named as unquantified* rather than omitted: compression records neither
  rank nor discarded norm (Milestone 4 holds that item), and the two contractions are
  exact, so their error is summation order, which Milestone 4 measured at 5.2e-8 eV on the
  deep states but never attributed per stage.
  Measured on water/cc-pVDZ at `nmom_max = 7`: truncation 7.0e-2 eV against eta0 6.0e-13 eV,
  eleven orders apart. A budget listing only the numerical terms would have described the
  wrong error entirely.
- [x] Derive or measure the amplification of an eta0 perturbation. Done under Milestone
  2.4 and recorded there: the dd-moment recurrence does not amplify it, and the frontier
  moves by 30-300x the scalar error in eV. `error_budget` uses the upper end, so the eta0
  entry is a bound rather than an estimate.
- [ ] Select the eta0 tolerance and pole count from the requested final moment/QP
  tolerance.
- [x] Report errors per sector and order; do not reduce all information to one scalar.
  The realization entry carries the per-order residuals and the conserved order for each
  sector, and the report deliberately has no total: the contributions are in different
  units and only some have a measured path to the frontier, so summing them would invent
  the missing conversions. A test pins the absence of a `total`.

### 3.2 Scaled or modified moments

- [x] Affinely center and scale the hole and particle spectral sectors separately so
  their supports are order one. **Measured and rejected** (2026-08-12,
  `baseline/studies/affine_recursion.py`): applied in front of the recursion the transform
  changes nothing it was proposed to change and costs accuracy. See the next item.
- [x] Transform raw monomial moments to the scaled basis before realization and
  transform poles back afterward. **Measured and rejected** (2026-08-12). The transform is
  exact and the poles map back exactly, but `MBLSE` never forms a Hankel matrix, so there
  is no conditioning for it to improve: in exact arithmetic the recursion is invariant
  under the similarity, and in floating point the binomial sum
  `sum_k C(n,k) (-mu)^(n-k) T_k` merely adds its own cancellation. Three systems, both
  sectors, orders 7 and 11, and three centre/scale choices including one given the *true*
  pole range: the lithium-hydride hole stall is unmoved to three significant figures
  (6/12 conserved, 57 poles, 7.04e-6 residual in every variant), while healthy sectors lose
  up to 500x (lih_hf particle at order 11, 2.57e-15 to 1.31e-12). Damage tracks `|mu|`,
  which is the cancellation. The conditioning gain this was premised on -- 13.7x to 2.3e6x
  on `cond(H0)`, `HANKEL_PENCIL.md` section 3 -- is real but applies to a Gram matrix only
  a one-shot route forms, not to this one.
- [ ] Evaluate a direct Chebyshev/modified-moment plus block-Jacobi backend for orders
  where raw monomial Hankel matrices lose usable precision. **Premise unmeasured on this
  path** (2026-08-10): no such order was found up to `nmom_max = 15`, from H2 to
  benzene/cc-pVTZ, with residuals at 5-8e-15 throughout `[corrected 2026-08-11: the step-downs were
  described as rank-limited, which was read off a classifier that could not see the cause.
  Every step-down in this code is a PSD failure. The evidence for parking 3.2 is the
  residual at the orders that complete, which is unaffected]`.
  MBLSE runs a block Lanczos recurrence and never forms a raw monomial Hankel matrix, which
  is likely why; `mkakcl/chebyshev-gw`, which Cholesky-factorises a Gram, is arith-limited
  at 13-27 depending on the system. Independent cross-check: lithium-hydride rank-limits at
  5-6 conserved orders in both codes `[corrected 2026-08-12: this was read as showing the
  limit is the molecule rather than the moment basis. Two codes agreeing was the whole
  evidence, and a third disagrees. The one-shot block Hankel pencil of `HANKEL_PENCIL.md`
  has no PSD gate -- it deflates on eigenvalue magnitude instead -- and on the same hole
  moments reaches 68 poles against MBLSE's 57, conserving all 12 at `K = 11`. That is
  consistent with, not contrary to, the PSD-gate finding below: loosening `neg_atol`/
  `neg_rtol` by 1e4 already buys lithium-hydride 2 more orders at 60x the residual, and the
  pencil sits further along the same trade rather than escaping it. So the limit is a
  gate *policy*, not the molecule and not the moment basis. This does not revive 3.2: the
  frontier difference between the two routes is 5.6e-7 eV against a truncation error of
  tens of meV]`.
  Do not build this until a system is found where the residual actually degrades;
  if one is, the source-agnostic `realization/` package in that repo is the thing to reuse,
  behind an option with monomial remaining the default.
- [ ] Keep the raw monomial backend as an oracle at low order during migration.

### 3.3 Adaptive moment order

**Why this is now first rather than third.** Measured 2026-08-10 on benzene/cc-pVTZ, raw
monomial, `nmom_max` swept 1 to 15 from one moment build: the realization shows **no
arithmetic ceiling** - the reconstructed-moment residual sits at 5-8e-15 at every order and
the single step-down at 15 is rank-limited - while the **frontier is nowhere near
converged**. The LUMO is still moving 41 meV at `nmom_max = 15` and 159 meV at 7; the HOMO
21 meV at 7. Quasiparticle weight drifts steadily with order too, 0.953 to 0.845. There is
no level crossing behind any of it: the dominant orbital is unchanged throughout.

Truncation is therefore the largest term in the error budget by nine orders of magnitude -
tens of meV against the ~1e-11 eV that every numerical effect in Milestones 1, 2 and 4
moves - and nothing reported it. That also removes the premise for 3.2 on this path; see
below.

- [x] Compare `m` with `m - 2` and report the frontier movement. `moment_order_convergence`
  (default `False`) realizes the self-energy at `nmom_max - 2` as well and records both
  frontiers, the shift, and the dominant reference orbital of each so a level crossing is
  visible rather than read as a large shift. Cheap because the moments in hand contain
  every lower order exactly - truncating to `nmom_max - 1` entries is identical to having
  built at `nmom_max - 2`, verified against two independent calculations - so it costs a
  realization and a Dyson solve, not a second moment construction: **1% on benzene/cc-pVTZ
  at `nmom_max = 7`**. Off by default because it is a second solve; at 1% it is a candidate
  for on by default, which the "trustworthy calculation" definition below already asks for.
- [x] Increase automatically through supported odd orders until the frontier stops moving.
  `nmom_max_tol` (default `None`) turns `nmom_max` into a cap: the walk realizes at
  1, 3, 5, ... and stops at the first order where the frontier has settled, then runs the
  ordinary path at that order. No second moment construction - the build at the cap
  contains every lower order - and stopping early *skips* the expensive high-order solves,
  so it can cost less than the plain run it replaces. Reaching the cap without meeting the
  tolerance is reported as unconverged and fails a `moment_order` gate, so the calculation
  says so rather than returning a number that looks finished.
  **Two consecutive orders must qualify, not one.** The shift is not monotonic in the
  order - on water/cc-pVDZ it runs 0.464, 0.382, 0.070, 0.096, 0.016, 0.042 eV - and a
  single-shift rule stops at order 11 for a 1e-3 Ha tolerance, immediately after which the
  frontier moves another 42 meV.
- [x] Carry the other quantities through the walk, distinguishing what converges from
  what must simply hold. The **particle-number error** converges with the order and is
  required as well as the frontier, against the tolerance its own gate already uses. The
  **spectral weight** does not converge: `Tr[G(0)] = nmo` with non-negative residues is a
  sum rule that holds at every order, so it is recorded as a validity check, not a
  stopping criterion - calling it converged would be a category error. Measured on
  water/cc-pVDZ it is satisfied by construction at every order (deficit ~1e-14, smallest
  residue exactly 1.0), so the check has no discriminating power today and earns its place
  only as a guard against a future change breaking it. The **reconstructed-moment**
  residual is already reported per sector and order by 1.1, at the realized order, and is
  a realization-fidelity measure rather than an order-convergence one; it stays at ~1e-15
  throughout (see 3.3's opening note).
- [x] Step down automatically when the next block fails the delivered-moment or PSD
  gate. Delivered by 1.3 and unchanged since: the recurrence stops at the last order it
  could complete, `order_reduced` records it, the realization gate fails and `converged`
  turns false. Verified on lithium-hydride at `nmom_max = 7`, where the hole sector steps
  from `max_cycle` 3 to 2 and the calculation reports itself unconverged.
- [x] Report requested, built, conserved, and realized orders separately. Delivered by 1.1
  and 1.3; `realization_record` carries all four per sector under
  `moments_supplied` (built), `nmom_conserved_requested`, `nmom_conserved_achieved`
  (conserved) and `n_poles` with `max_cycle_achieved` (realized). On the lithium-hydride
  case above the hole reads 8 built, 8 requested, 6 conserved, 76 poles realized, against
  8/8/8/95 for the particle.

### 3.4 Closure and spectral diagnostics

- [x] Compare independent admissible closures. `closure_spread` (default `False`)
  re-realizes the self-energy with a Gauss-Radau rule in place of the natural Gauss
  truncation and reports how far the frontier moves. Both conserve the moments that were
  supplied - Gauss to order `2K - 1`, Radau to `2K - 2`, validated - and differ only in
  what they assume about the ones that were not. Implemented entirely here: the recurrence
  exposes its Jacobi blocks, so the Gauss rule can be rebuilt from them bit-identically and
  re-closed, with no change to Dyson and no pin move.
  **The pin is per sector, and that is not a detail.** The sectors' supports are disjoint
  and far apart - on water/cc-pVDZ the hole runs to -40 Ha and the particle starts at
  +1.1 - so a single shared pin, the chemical potential included, sits ~1 Ha outside one
  of them and inflates the spread about sixfold. Each sector is pinned just outside its own
  edge, 1% of the inter-sector gap; the spread is flat over that region (-0.196, -0.205,
  -0.224 eV at 0.005, 0.02, 0.05 Ha) and inflates once the pin sits well off the edge
  (-1.69 eV at 0.6 Ha), where the rule spends a node on empty space.
  **It has a floor, and that is a real limitation** `[corrected 2026-08-11]`. It was first
  recorded here as earning its place by disagreeing with the `m` versus `m - 2` estimate -
  at `nmom_max = 11` on water differencing says 0.016 eV and the closure 0.197 eV, with the
  frontier then moving 42 meV. The order-convergence tables show the fuller picture: the
  spread barely falls with order at all. On water it is 206, 219, 197, 204, 205 meV across
  `nmom_max` 7 to 15 while the differencing falls to 0.5 meV; on lithium-hydride it sits at
  ~188 meV while the differencing reaches 0.02 meV. That is not the pin: pushing the pin to
  the support edge gives 193 meV at order 7 and 187 at order 13, a 3% fall against a 40%
  fall in the error it is meant to indicate.
  So the spread is a **conservative one-sided signal, not an error estimate**. It can say
  "not converged" and it caught a case where the differencing was fooled, but it cannot
  certify convergence and must not be read as a magnitude. Whether the floor is intrinsic to
  pinning a node at a support edge or an artefact of this construction is open; one
  candidate is now ruled out, since the two closures pick the same frontier state on water
  and lithium-hydride at every order, and the estimate records the comparison so a case
  where they do not - ozone - is flagged rather than silently differenced.
- [x] Do not label closure spread a rigorous QP bound. The Gauss/Radau bracket holds for
  an integral of a function with derivatives of constant sign; a quasiparticle energy from
  an upfolded eigenproblem is not one, and no theorem here covers that step. The record
  carries `is_a_bound: False`, the log line says "indicator, not a bound", the module
  docstring states the gap, and a test asserts the flag - so it cannot quietly be promoted.
- [ ] Validate frontier IP/EA separately from deep-state largest-overlap labels.
- [ ] Add a spectral-function comparison against an untruncated small-system oracle
  where satellites or deep states are reported.

### Acceptance gate

**Four of five pass; the first does not, and ozone is why.** Tables from
[`baseline/studies/order_convergence.py`](baseline/studies/order_convergence.py), swept to
`nmom_max = 15` from one moment build per system, PBE / cc-pVDZ. The frontier is read by
Aufbau counting over the correlated multiplets, which survives a level crossing.

- **Partly passed** - increasing supported moment order reduces the stated frontier error
  until the predicted limit. Water falls 464 to 0.5 meV per step, lithium-hydride 789 to
  0.02 meV, and both then stop at a limit. **Ozone does not**: it reaches `nmom_max = 15`
  with no shortfall at all and the frontier still moving 83 meV per step, so no limit has
  been reached and this criterion is unmet for the small-gap case. It needs a higher cap.
- **Passed** - no calculation silently continues after a failure. The tables carry the gate
  states per row: every stepped-down order shows `realization` failed and `converged`
  false, which is the criterion demonstrated rather than asserted.
- **Passed** - the reported bounds are consistent with the observed moment errors. The
  reconstructed-moment residual sits at 2.4e-15 to 1.0e-14 across all three systems and
  every order that completes, and never degrades.
- **Passed** - H2O, LiH and the small-gap system have documented order-convergence tables.
- **Passed** - low-order results remain compatible with the Milestone 0 baseline: 52/52.

**What the limit actually is** `[corrected 2026-08-11]`. An earlier version of this section
classified step-downs as `rank` or `arith` by the reconstructed-moment residual and called
them genuine. Both readings were wrong. `MBLSE.kernel` steps down in exactly one place,
catching a PSD failure on the next block's square root, so there is no dichotomy to
classify; and the residual is measured at the *achieved* order, which the failure never
touched, so it reports ~1e-15 for every step-down and cannot indicate a cause. The probe
that pronounced the limits genuine scaled `atol`/`rtol`, which set the support mask and
provably cannot move a step-down - it returned bit-identical results.

Measured with `neg_atol`/`neg_rtol`, which do govern it, the two systems differ and neither
matches what was claimed:

| system | conserved | residual | with the PSD gate loosened 1e4 | reading |
| --- | --- | --- | --- | --- |
| lithium-hydride, `K = 7` | 6 | 3.58e-15 | **8** at 2.22e-13 | the gate is binding; it costs 2 orders, at 60x the residual |
| water, `K = 15` | 14 | 5.60e-15 | 14 at 5.60e-15 | not the tolerance: the direction is materially negative |

**What a step-down costs against what was asked for** `[added 2026-08-12]`. The residual
column above is measured at the *achieved* order, which is why it stays at ~1e-15 through
every step-down and cannot indicate a cause. Measured instead against the **requested**
moment set (`baseline/studies/pencil_vs_mblse.py`), the lithium-hydride hole sector reaches
2.6e-7 at `K = 7`, 2.0e-6 at `K = 9` and 7.0e-6 at `K = 11` -- the error in the moments the
caller asked for and did not get. The two numbers are not in conflict; they answer different
questions, and only the second says what the step-down cost. Frontier impact is small
regardless: 5.6e-7 eV on the HOMO against the pencil, which conserves the full set.

**Two findings that bear on reading the tables.** Lithium-hydride's frontier is a satellite,
weight 0.458, so its 0.02 meV convergence says nothing about a quasiparticle; the study warns
below 0.7. Ozone crosses a level between orders 1 and 3, and its two closures disagree on
the frontier state at least once - so part of its closure column compares different states,
and the study now says so. Both are 3.4's remaining item showing up in data.

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
takes it to 22.8%, after which the correlated moment construction is the majority again.

**Re-profiled after that landed** (benzene/cc-pVTZ, `nmom_max = 7`, production path, one
process, 38.83 s total), which is what this milestone's acceptance gate asks for before
optimising further:

| stage | s | share |
| --- | --- | --- |
| moment construction (dd + se) | 20.58 | **53.0%** |
| Dyson | 8.85 | 22.8% |
| integrals | 6.84 | 17.6% |
| static self-energy | 2.57 | 6.6% |

and within the moment construction, by self time: `build_se_moments` 8.34 s (21.5% of the
run), `convolve` 3.65 s (9.4%), `_hht_apply` 3.33 s (8.6%), `cho_solve` 1.41 s (3.6%), and
a single `ascontiguousarray` at 1.28 s (3.3%) - the layout copy the batched loop pays for.

**`convolve` is 9.4%, not the 1% recorded above.** The refactor item below was written
against the old profile and is worth about nine times what its placement suggests; it and
the `Lpx` layout item are now the two cheapest wins in this milestone. Two lessons worth
keeping, both in
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
- [x] Refactor convolution. ~~so moment orders accumulate locally and perform one final
  MPI reduction and symmetrization instead of reducing full stacks repeatedly~~ - the
  reduction was not where the time went. `convolve` already all-reduces once at the end,
  which is a no-op on one rank; the 9.4% measured above was local. Two things caused it,
  both in the same loop. The contraction `sum_{k,t} f[t] e[k]^(n-t) eta[k,t,pq]` was one
  `einsum` whose summation indices appear in all three operands, so NumPy cannot express it
  as a `tensordot` even with `optimize=True` and it ran in the unblocked, single-threaded
  kernel; and each output order re-selected `eta[mask]`, a fancy-index copy of an array that
  is 147 MB on benzene/cc-pVTZ, inside a loop over a mask that does not depend on the loop
  variable. Folding the weights into one operand leaves a matrix product, and stacking the
  orders into its rows reads `eta` once per call rather than twice per order. **23x** on the
  full sweep at `nmo = 264` (6.511 s to 0.283 s), 8.6x at `nmo = 114`; `convolve` leaves the
  profile entirely. Identical arithmetic in a different summation order: 52/52 baseline
  cases unchanged, frontier moving at most 3.3e-11 eV with a median of 1.0e-13 eV, and the
  realization residuals scattering in both directions at the 1e-15 floor rather than
  degrading. Re-recorded in the same commit.
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

