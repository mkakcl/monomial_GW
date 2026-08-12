# The one-shot block Hankel pencil: momentGW's analogue of Cayley's Toeplitz QR

Written 2026-08-12. **Nothing here is implemented.** Every number is reproduced by
[`baseline/studies/hankel_pencil.py`](baseline/studies/hankel_pencil.py) on this machine
(macOS, Accelerate, PySCF 2.14.0, `mgw-monomial`) at the pin `mkakcl/dyson@73cd18d`.

The question this answers: the Cayley project (`mkakcl/cayley-gw`) has two realization
backends — `taylor`, a recursion structurally like `MBLSE`, and `toeplitz-qr`, which
replaces the whole recursion with one Gram matrix and survives to roughly three times the
moment order as a result. Is there an equivalent for momentGW?

## Summary

**Yes, the analogue exists and is exact.** It is the symmetric-definite pencil on the block
Hankel moment matrices. It rank-deflates as cleanly as `toeplitz-qr` does, and on exact
moments it recovers poles to 1e-12 Ha at order 41 where the recursion has long since failed.

Four things qualify that, and they change what is worth building:

1. **The stability win does not transfer.** `toeplitz-qr`'s margin comes from `|u| = 1`,
   which is a property of the *Cayley moments*, not of the algorithm. On the real line the
   retained singular values sit at 1e-9 to 1e-5 relative rather than O(1), so node error is
   roughly `noise / 1e-9`. At 1e-12 relative moment noise the hole sector is already wrong
   by 4e-2 Ha at K=7 and by tens of Ha beyond. **No realization backend can recover this**
   — it is set by the moments.
2. **The affine renormalisation is not a detail, it is the entire mechanism.** In raw
   monomial coordinates the rank cut that makes the singular Gram usable has a gap of
   **0.1 decades** — it is not identifiable at all. After an exact affine map it is
   **7.8 to 10.8 decades**. Same data, same algorithm.
3. **That renormalisation is free, exact, and helps the *existing* recursion too** — it is
   not tied to the new backend. It is the single cheapest item here, and half of it is
   already implemented and tested in dyson, just not on the compute path.
4. **So the pencil's value is diagnostic, not production.** It measures something `MBLSE`
   structurally cannot: how many moments the data actually supports.

Recommended order of work is §6: **(A) affine renormalisation first, (B) pencil as a
second opinion, (C) do not chase K≈40.**

---

## 1. The shared trait

`toeplitz-qr` works because of a Gram identity. For a **unitary** shift the Krylov Gram
block depends only on the index difference,

```
<U^i B†, U^j B†> = C_{j-i}
```

so the Gram is block Toeplitz and is written down directly from the moments — no recursion
needed to discover it. One `eigh`, one rank-revealing square root, one orthogonal Procrustes
step for the shift, and the realization is done.

momentGW has the same identity with a different index law. The self-energy moments are those
of a **self-adjoint** operator, so

```
<H^i B†, H^j B†> = T_{i+j}
```

and the Gram is block **Hankel**. The one-shot analogue is the symmetric-definite pencil

```
H0 = [T_{i+j}]        H1 = [T_{i+j+1}]        i, j = 0 ... m
```

whose eigenvalues are the poles and whose eigenvectors give the couplings. `eigh(H0)`,
rank-revealing square root, one generalized eigensolve. No recursion, no repeated inverse
square roots.

That last clause is the point. `MBLSE._recurrence_iteration_hermitian`
(`mblse.py:300-315`) forms
`off_diagonal_squared`, takes `matrix_power(·, -0.5)`, and applies it on **both sides** of
the next coefficient — inside the loop, so every later cycle inherits the amplification.
That is structurally identical to `taylor`'s `1/√defect`, which is the thing `toeplitz-qr`
was written to avoid.

## 2. It deflates, and deflation is what makes it work

`H0` is PSD by construction and becomes singular once the Gram dimension passes the number
of poles. In `toeplitz-qr` that is a documented feature rather than a failure — "rank
deflation is a feature, not a fallback". The Hankel Gram does the same thing.

Water, HF, `nmom_max=7`, moments synthesised exactly from the recorded Lehmann
representation (96 poles per sector, 24 physical), affine-mapped onto its true support:

| K | dim | num. rank | cond | sv[n−1]/sv[0] | sv[n]/sv[0] | node err (Ha) |
|---|---|---|---|---|---|---|
| 3 | 48 | 48 | 7.4e+06 | — | — | 1.87e+00 *(under-resolved)* |
| 7 | 96 | **96** | 9.1e+09 | 1.10e-10 | — | 2.04e-07 |
| 13 | 168 | **96** | 1.4e+19 | 4.00e-09 | 6.89e-17 | 8.88e-10 |
| 21 | 264 | **96** | 8.7e+19 | 5.87e-09 | 9.34e-17 | 2.41e-09 |
| 41 | 504 | **96** | 5.7e+20 | 7.31e-09 | 1.38e-16 | 4.62e-10 |

Rank saturates at exactly the pole count while the matrix grows to 504. **The `cond = 1e20`
is not ill-conditioning** — it is the measure reporting its own finite support, and the null
space is discarded rather than regularised. Accuracy *improves* with order past saturation.

The K=3 row is marked because below saturation the rule has 48 nodes for a 96-pole measure:
it is an under-resolved quadrature rule by construction, and its distance to the nearest
exact pole is not a measure of its accuracy. Only rows at or past saturation are comparable.

The particle sector behaves the same and is roughly two decades better throughout
(7.9e-11 at K=7, 4.1e-12 at K=41), because its support sits at 1.7 to 27 Ha rather than the
hole sector's −44.8 to −1.7.

## 3. The affine renormalisation is the whole mechanism

The transform is the exact one induced by `x → (x − μ)/s`:

```
T̃_n = s^(−n) Σ_k C(n,k) (−μ)^(n−k) T_k
```

It needs only the moments, and it is undone exactly on the recovered block-tridiagonal
matrix as `J → sJ + μI` — a similarity, so **nothing is approximated by using it**.

Without it, there is no rank cut to find. Hole sector, K=13:

| coordinates | sv[n−1]/sv[0] | sv[n]/sv[0] | gap |
|---|---|---|---|
| raw monomial (Ha) | 2.68e-19 | 2.15e-19 | **0.1 decades** |
| affine → [−1,1] | 4.00e-09 | 6.89e-17 | **7.8 decades** |

Particle is 0.2 decades raw against 10.8 affine. In raw coordinates the entire spectrum has
already sunk below the noise floor; there is nothing to threshold.

**This is not confined to the new backend.** Across all 26 sectors of the recorded baseline
set at `nmom_max=7`, using a centre and scale estimated from the first three moments alone
(no pole knowledge — available inside the solver, where the support is exactly what is
unknown):

```
cond(H0) gain: min 13.7x, median 3223x, max 2.3e6x
```

Hole sectors gain most, because their support sits ~20 Ha off the origin and the monomial
powers are dominated by the offset rather than by the spread. That offset is precisely what
also degrades the recursion's `off_diagonal_squared`.

## 4. Where it fails, and why that is not fixable here

Median max node error in Ha, relative noise added entrywise to every moment:

**Hole sector**

| noise | K=7 | K=13 | K=21 | K=33 |
|---|---|---|---|---|
| exact | 2.0e-07 | 8.9e-10 | 2.4e-09 | 7.8e-10 |
| 1e-14 | 5.3e-04 | 1.4e-05 | **4.2 Ha** | **4.5 Ha** |
| 1e-12 | 3.9e-02 | **43 Ha** | **39 Ha** | **36 Ha** |
| 1e-10 | **191 Ha** | **78 Ha** | **134 Ha** | **51 Ha** |

**Particle sector** (materially more robust)

| noise | K=7 | K=13 | K=21 | K=33 |
|---|---|---|---|---|
| exact | 7.9e-11 | 2.1e-11 | 6.1e-12 | 6.1e-12 |
| 1e-14 | 9.9e-09 | 3.4e-09 | 2.2e-09 | 1.7e-09 |
| 1e-12 | 2.0e-06 | **16 Ha** | **14 Ha** | **14 Ha** |
| 1e-10 | 6.4e-05 | **46 Ha** | **58 Ha** | **34 Ha** |

The mechanism is straightforward perturbation theory: node error ≈ noise / (smallest
retained singular value). With `sv[n−1]/sv[0]` at 4e-9 for holes, 1e-14 noise gives ~1e-5
relative, times a 21.5 Ha scale — which is what the table shows.

On the circle those retained singular values are O(1). **That is the entire difference, and
it is a property of the moments.** This is the honest answer to "can we get the
`toeplitz-qr` win in momentGW": no, not by changing the realization backend.

Two caveats on this section specifically, because it carries the negative conclusion:

- The moments here are **exact by construction** (synthesised from a recorded Lehmann
  representation) and the noise is synthetic i.i.d. Gaussian. Real RPA-quadrature error is
  neither i.i.d. nor entrywise, and its magnitude on this path **has not been measured** —
  §6 lists that as the first gate.
- One system, one basis. The hole/particle asymmetry is large enough that it should not be
  assumed to hold shape elsewhere.

## 5. What this does *not* claim

- **No head-to-head against `MBLSE`.** Everything above measures the pencil against exact
  poles. Whether it beats the recursion at equal noise is untested, and is the obvious next
  measurement.
- **No timing.** The Gram is `(m+1)·nphys` square — 504 for water at K=41, but it grows
  with `nphys`, and for benzene/cc-pVTZ at moderate K it is large. Cost has not been
  compared to the recursion.
- **Nothing about the `dd` moments or the RPA.** This is the realization stage only.
- **The affine centre/scale heuristic is not tuned.** §3 uses trace Rayleigh quotients,
  which is the cheapest defensible estimator, not an optimised one. §2's tables use the
  *true* support, which is unavailable in practice — the gap between the two is unmeasured.

## 6. Proposed order of work

### A. Affine renormalisation in front of the existing recursion — do this first

Independent of any new backend, and the best ratio of gain to effort here.

`shift_moments` already exists and is tested at
`_mbl.py:70`, but it is used **only
in the error-reporting path** (`_mbl.py:668`,
comparing predicted against reference moments about the chemical potential). It never
touches the moments that feed the recursion.

Steps:

1. **Measure the real moment noise on this path first.** Everything downstream is calibrated
   against it, and §4 is currently a proxy. Without this number we cannot say whether any
   of this helps.
2. Add the scale to the existing shift (`s^{-n}`), giving the full affine transform.
3. Apply it before `initialise_recurrence`; undo as `J → sJ + μI` on the block-tridiagonal
   assembled at [`gw.py:452`](momentGW/gw.py#L452). Exact similarity — assert it.
4. Choose μ, s from the first three moments (§3), per sector.
5. Re-record `baseline/` and check: the QP energies must be unchanged within tolerance, and
   the recursion's reported `error_inv_sqrt` should fall.

Gate: no baseline case moves beyond its recorded tolerance, and at least the hole sectors
show a measurable drop in recursion error metrics. This is a **fork-only** change; see
`CLAUDE.md` on upstream.

### B. The pencil as a second opinion — build after A

Its value is not accuracy, it is that it measures what `MBLSE` cannot infer: **the sv cliff
gives the effective pole count of the data**, so "is order 11 supported?" becomes an
observable rather than a guess from whether the recursion returned. That serves the ROADMAP
rule *"never report convergence solely because a linear algebra routine returned."*

Steps:

1. Head-to-head against `MBLSE` at realistic noise (the §5 gap). **If the pencil loses
   everywhere, stop here** — record the negative result and close it out.
2. If it holds at low K, expose it as a diagnostic reporting effective rank and the sv gap,
   not as a `solve_dyson` backend.
3. Only then consider it a selectable backend, and only with the deflation thresholds
   treated as first-class options the way `toeplitz-qr` treats
   `block_rank_absolute`/`block_rank_relative`.

### C. Do not chase K≈40 on monomial moments

The ceiling in §4 is set by the representation. The two routes that lift it — Cayley and
Chebyshev — already exist as separate projects (`mkakcl/cayley-gw`,
`chebyshev-momentGW`). Modified moments would have to be built **at source** in
`build_se_moments` ([`tda.py:300`](momentGW/tda.py#L300)); post-transforming monomial
moments into a Chebyshev basis cannot work, because the ill-conditioning is already baked
into the data by then. That is a moment-construction project, not a realization one, and it
is out of scope for this document.

## 7. Reproducing

```bash
python -m baseline.studies.hankel_pencil                        # all four sections
python -m baseline.studies.hankel_pencil --section deflation    # one section
```

`baseline/arrays` is gitignored and reproducible, so a fresh worktree has none. Either run
`python -m baseline.run` or point at a tree that already has them:

```bash
python -m baseline.studies.hankel_pencil --arrays ../../../baseline/arrays
```

Run from the repository root, and read the printed `momentGW.__file__` before believing any
cross-tree comparison — a driver script kept elsewhere silently imports the main tree.
