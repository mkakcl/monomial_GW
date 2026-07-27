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

**Status: Planned**

### Work

- [ ] Record a compact baseline set of restricted molecular G0W0/dRPA calculations:
  H2, H2O, LiH, and at least one small-gap molecule.
- [ ] Save eta0, density-response moments, hole and particle self-energy moments,
  reconstructed moments, frontier QP energies, particle-number error, auxiliary rank,
  and stage timings.
- [ ] Include `nmom_max=1, 3, 5, 7` where numerically feasible.
- [ ] Run compressed and uncompressed auxiliary spaces for at least one molecule.
- [ ] Record the exact momentGW, Dyson, PySCF, NumPy, SciPy, and BLAS versions.
- [ ] Correct `[tools.setuptools.dynamic]` to `[tool.setuptools.dynamic]` so builds
  publish the package version correctly.
- [ ] Stop installing mutable `dyson@master`; pin the currently tested commit, then
  update that pin only when Milestone 1 is accepted.

### Acceptance gate

- The baseline is reproducible from a fresh environment.
- Every recorded result identifies its code revisions and numerical options.
- Existing restricted molecular G0W0/dRPA regression values remain unchanged.

## Milestone 1 - Repair Dyson realization and error reporting

**Status: Planned - first implementation milestone**

Dyson is an external dependency, so these changes should be made upstream in
`BoothGroup/dyson`, tested there, and then consumed here through an immutable commit.
A private runtime patch in momentGW is not an acceptable final solution.

### 1.1 Reconstructed-moment diagnostic

- [ ] Fix `MBLSE.reconstruct_moments`: the current implementation reconstructs
  `range(2 * iteration)`, while the error routine compares against
  `2 * iteration + 2` input moments. Python `zip` silently omits the newest two
  moments, and iteration zero checks none.
- [ ] Assert equal predicted and reference moment counts before computing an error.
- [ ] Report per-order absolute and relative Frobenius and maximum-norm errors.
- [ ] Separate errors before and after any intentional chemical-potential pole shift.

### 1.2 Matrix square root and inverse square root

- [ ] Replace the fixed absolute `1e-10` eigenvalue cutoff with a scale-aware
  `atol + rtol * lambda_max` policy.
- [ ] Treat the square root and inverse square root with one consistent effective
  support.
- [ ] Clip only negative eigenvalues that are demonstrably compatible with roundoff.
  Fail on materially negative directions.
- [ ] Report the minimum eigenvalue, condition estimate, effective rank, discarded
  norm, and the resulting reconstructed-moment error.
- [ ] Do not describe the norm of discarded original eigenvalues as the error in an
  inverse square root.

### 1.3 Realization feasibility

- [ ] Validate finite, Hermitian moments before starting the recurrence.
- [ ] Validate positive semidefiniteness and causality where required by the measure.
- [ ] When the requested order is not supportable, step down to the largest order that
  satisfies a delivered-moment residual and report the reduction.
- [ ] Validate moment-order parity. Either require odd `nmom_max` for the current MBLSE
  construction or define and report exactly which supplied moments are used.
- [ ] Avoid repeated intermediate diagonalizations used only for diagnostics unless
  error reporting is enabled.

### 1.4 momentGW integration

- [ ] Pass explicit Dyson numerical options from `momentGW/gw.py`.
- [ ] Replace unconditional single-shot `conv=True` with a numerical convergence
  result that includes realization and particle-number gates.
- [ ] Store structured diagnostics on the GW object instead of only writing log text.
- [ ] Pin momentGW to the accepted Dyson commit.

### Acceptance gate

- Synthetic positive matrix measures reconstruct every promised moment through the
  requested order.
- A test fails if predicted and reference moment counts differ.
- Small positive support directions are either preserved or explicitly rejected by a
  scale-aware rule with a measurable moment effect.
- Materially non-PSD input fails loudly.
- Requested and achieved moment orders are both reported.
- Existing low-order G0W0 QP energies remain unchanged within the baseline tolerance.

## Milestone 2 - Stable eta0 through HHT/Zolotarev

**Status: Planned - second implementation milestone**

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

- [ ] Add a small eta0 module containing pole generation, certified bounds, scalar
  error evaluation, and the projected inverse-square-root action.
- [ ] Add options equivalent to:
  `eta0_method={"clencur", "hht"}`, `eta0_tol`, optional `eta0_n_poles`, and
  `eta0_check_refinement`.
- [ ] Preserve the meaning of the existing `npoints` option for the legacy route.
- [ ] Introduce HHT as opt-in and retain Clenshaw-Curtis as the shadow/reference
  calculation during rollout.

### 2.2 Stable coefficients

- [ ] Replace the direct `ellipk(1 - k2)`/real-Jacobi construction when the condition
  number makes it unstable.
- [ ] Use stable complementary-parameter/nome evaluation or generate the small number
  of poles and weights in extended precision before casting to float64.
- [ ] Assert positive, finite shifts and weights.
- [ ] Add explicit handling for zero coupling, an empty particle-hole space, and a
  degenerate spectral interval.

### 2.3 Certified spectral interval

- [ ] Require finite, strictly positive particle-hole gaps.
- [ ] Use the rigorous distributed enclosure
  `lambda_min = min(D)^2` and
  `lambda_max = max(D^2) + c * min(||W||_F^2, ||W||_1 ||W||_inf)`.
- [ ] Pad the lower endpoint outward/down and the upper endpoint outward/up for
  floating-point arithmetic.
- [ ] Use Lanczos only to report how loose the rigorous upper bound is.
- [ ] Report the interval and condition number in the calculation diagnostics.

### 2.4 Accuracy certificate

- [ ] Replace the asymptotic `required_poles + safety` rule with selection based on
  the scalar relative error
  `max |1 - sqrt(x) r_N(x)|` over the certified interval.
- [ ] Use the Zolotarev equioscillation/error formula where practical and validate it
  against a high-precision scalar oracle.
- [ ] Check residuals of every auxiliary-space Cholesky solve.
- [ ] Retain `N_p` versus `N_p + 4` as a secondary regression signal, not the sole
  accuracy certificate.
- [ ] Derive the default eta0 tolerance from the higher-moment error budget in
  Milestone 3 rather than fixing it permanently at machine precision.

### 2.5 Projected kernel and MPI

- [ ] Cache `D`, `D^2`, `sqrt(D)`, contiguous `W.T`, bounds, poles, and weights.
- [ ] Build one local weighted Gram per pole and all-reduce only the auxiliary-space
  Gram.
- [ ] Use Cholesky factorization and solves; never form an explicit inverse.
- [ ] Remove the avoidable bare-`Lia`/`-I` cancellation from the HHT path.
- [ ] Assert the expected local output shape and prohibit particle-hole squared
  intermediates.

### Acceptance gate

- Scalar tests cover well-conditioned intervals through extreme condition numbers,
  with a high-precision oracle and an explicit failure when the requested tolerance
  is not representable.
- Dense small-matrix tests compare against an eigendecomposition of `Mtilde`.
- H2 and H2O eta0 agree with the legacy route and dense oracle to the predicted
  floating-point limit.
- Density-response moments, self-energy moments, and final QP energies are invariant
  within their error budgets.
- Serial and multi-rank MPI results agree.
- Compression on/off, frozen-core, zero-coupling, invalid-gap, and small-gap cases are
  covered.
- Shape instrumentation proves that no particle-hole squared matrix is formed.
- HHT becomes the default only after all gates pass; legacy code is removed in a later
  change.

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

**Status: Planned**

### Work

- [ ] Batch the external-orbital loop in `momentGW/tda.py` into larger BLAS-3
  contractions.
- [ ] Refactor convolution so moment orders accumulate locally and perform one final
  MPI reduction and symmetrization instead of reducing full stacks repeatedly.
- [ ] Batch several HHT Gram reductions or overlap nonblocking reductions with local
  work, subject to memory profiling.
- [ ] Add a native no-compression integral-transform path instead of multiplying each
  block by a dense identity rotation.
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
2. **Dyson diagnostic correction** - upstream reconstruction-count tests and accurate
   per-order residuals.
3. **Dyson support/PSD policy** - scale-aware matrix powers, rank reporting, and
   feasibility gates; update the momentGW pin.
4. **HHT scalar layer** - stable coefficients, rigorous bounds, exact scalar error
   checks, and high-precision tests.
5. **HHT projected eta0** - restricted molecular kernel, MPI path, and dense/legacy
   equivalence tests behind an opt-in flag.
6. **HHT default** - downstream G0W0 invariance, small-gap validation, documentation,
   and deprecation of Clenshaw-Curtis-specific options.
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

