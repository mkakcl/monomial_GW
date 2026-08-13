# The one-shot block Hankel pencil: momentGW's analogue of Cayley's Toeplitz QR

Written 2026-08-12 and revised the same day as each proposal was measured: §4 and §6.A by
the noise gate, §5 and §6.B by the head-to-head, then §6.A again — to a rejection — by the
affine test.
**Nothing here is implemented.** Every number is reproduced by
[`baseline/studies/hankel_pencil.py`](baseline/studies/hankel_pencil.py) and
[`baseline/studies/moment_noise.py`](baseline/studies/moment_noise.py) on this machine
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

1. **The stability win does not transfer, but the operating point is better than it
   first looked.** `toeplitz-qr`'s margin comes from `|u| = 1`, a property of the *Cayley
   moments* rather than of the algorithm; on the real line the retained singular values sit
   at 1e-9 to 1e-5 relative instead of O(1), so node error runs at about `noise / 1e-9` and
   **no realization backend can change that**. The gate in §4.1 then measured the real
   moment error at **1.4e-15 to 8.9e-15**, three decades better than first assumed, which
   leaves a usable window: **particle to K≈21, hole to K≈13**, and failure beyond.
2. **The affine renormalisation is not a detail, it is the entire mechanism.** In raw
   monomial coordinates the rank cut that makes the singular Gram usable has a gap of
   **0.1 decades** — it is not identifiable at all. After an exact affine map it is
   **7.8 to 10.8 decades**. Same data, same algorithm.
3. **That renormalisation does *not* help the existing recursion — measured, §6.A.** It is
   load-bearing for the pencil and irrelevant to `MBLSE`, which never forms a Hankel matrix.
   Applied to the recursion it leaves the stall untouched to three significant figures, even
   given the true support, and costs up to 500x accuracy on healthy sectors through the
   binomial transform's own cancellation. §3's conditioning gain is real but applies to a
   matrix the production path does not build.
4. **The head-to-head settles what the pencil is for, and it is less than proposed.**
   Measured against `MBLSE` on the same real moments (§6.B): where the recursion is healthy
   it loses by three to five decades, so it is not a backend; where the recursion *stalls*
   it wins by five, holding 12 moments to 1.4e-11 where the recursion keeps 6 and drifts to
   7.0e-6. But the stall is **already reported** by `nmom_conserved`, so it is not a
   diagnostic either, and the two routes' frontier energies agree to ≤1.1e-8 eV — the extra
   fidelity buys sub-µeV. The sweep in §6.B.1 then found a failure that *does* clear the
   bar — 46 µeV on hydrogen, where truncation error is designed out — so the pencil is kept
   rather than closed, but scoped to `K ≤ 15`, above which it is worse than what it replaces.

Where this leaves §6: **(A) is closed — measured and rejected.** **(B) is measured** —
the bar is cleared on one system, so the pencil survives as a scoped fallback for failed
sectors at `K ≤ 15`, pending a cost/benefit call rather than implementation. (C) stands:
do not chase K≈40 on monomial moments.

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

Two caveats on the tables above: the moments are **exact by construction** (synthesised
from a recorded Lehmann representation) and the noise is synthetic i.i.d. Gaussian, which
real error is not; and it is one system in one basis, with a hole/particle asymmetry large
enough that it should not be assumed to hold shape elsewhere.

### 4.1 The gate: what the real moment error actually is

Measured by [`moment_noise.py`](baseline/studies/moment_noise.py), which runs the production
pipeline from an **exact** eta0 (a dense eigendecomposition of Mtilde) and differences it
against the shipped path at `eta0_method="hht"`, `eta0_tol=1e-14`, compression off.

The structure of the path is what makes this the whole story. `build_dd_moments`
(`rpa.py:382`) takes the zeroth moment from `build_zeroth_dd_moment`, sets
`moments[1] = Lia * d` exactly, and builds every higher order by an exact algebraic
recursion. **There is exactly one numerical error source on this path: eta0.** There is no
quadrature left to refine — `npoints` is read only by the legacy Clenshaw-Curtis route
(`base.py:67`).

| system | eta0 error | dd (max) | hole (max) | particle (max) |
|---|---|---|---|---|
| lih_hf | 2.1e-15 | 2.8e-15 | **1.4e-15** | **1.6e-15** |
| water_hf | 3.4e-14 | 1.3e-14 | **3.7e-15** | **3.7e-15** |
| ozone_pbe | 1.8e-12 | 2.9e-13 | **4.0e-15** | **8.9e-15** |

Two things follow, and both matter:

- **The convolution to the self-energy damps the eta0 error rather than amplifying it.**
  Ozone carries 1.8e-12 in eta0 and 2.9e-13 in the zeroth dd moment, but only 4.0e-15 and
  8.9e-15 in the self-energy moments. The per-order error also *falls* with order — n0 is
  the worst row in every case. This is consistent with
  [`eta0_amplification.py`](baseline/studies/eta0_amplification.py), which found no
  amplification through the dd recursion and a float64 floor below ~1e-13 scalar error.
- **The real operating point is at or below the best row of the tables above.** At 1e-14 the
  particle sector holds 1.7e-9 to 9.9e-9 Ha through K=33, and the hole sector holds
  5.3e-4 Ha at K=7 and 1.4e-5 Ha at K=13 before failing at K=21.

So the pencil is **more viable than the tables alone suggest** — it has a real working
range, and that range covers the orders momentGW is normally run at. It is still not the
`toeplitz-qr` win: the hole sector dies at K=21 where `toeplitz-qr` reaches ≈40.

What this bounds, and what it does not: the reference is a dense eigendecomposition in the
same float64 arithmetic, so it bounds the error **relative to exact eta0**. Basis, SCF
convergence and density fitting are common to both sides and cancel out rather than being
measured. Compression is off by choice; `--compression ia` puts that error back and it has
not been swept.

## 5. What this does *not* claim

- ~~No head-to-head against `MBLSE`.~~ **Done — §6.B.** Neither route dominates; the split
  is by whether the recursion has stalled.
- **No timing.** The Gram is `(m+1)·nphys` square — 504 for water at K=41, but it grows
  with `nphys`, and for benzene/cc-pVTZ at moderate K it is large. Cost has not been
  compared to the recursion.
- **No compression sweep.** §4.1 measures with compression off; the auxiliary
  compression error is excluded by choice and its size here is unmeasured.
- **The affine centre/scale heuristic is not tuned.** §3 uses trace Rayleigh quotients,
  which is the cheapest defensible estimator, not an optimised one. §2's tables use the
  *true* support, which is unavailable in practice — the gap between the two is unmeasured.

## 6. Proposed order of work

### A. Affine renormalisation — measured, and rejected

**Do not build this.** Measured by
[`affine_recursion.py`](baseline/studies/affine_recursion.py), which applies the exact
transform to the moments, runs the recursion, and maps the poles back, scoring
reconstruction against the *original* moments throughout. Three systems, both sectors,
orders 7 and 11, three centre/scale choices.

**It does not move the stall.** On the lithium-hydride hole sector every variant is
identical to three significant figures — including `support`, which is handed the true
pole range and so is the best the transform could possibly do:

| variant | μ | s | conserved | cycles | poles | recon |
|---|---|---|---|---|---|---|
| none | 0.000 | 1.000 | 6/12 | 2/5 | 57 | 7.04e-06 |
| centre-only | −2.759 | 1.000 | 6/12 | 2/5 | 57 | 7.04e-06 |
| trace | −2.759 | 1.617 | 6/12 | 2/5 | 57 | 7.04e-06 |
| support | −3.737 | 3.125 | 6/12 | 2/5 | 57 | 7.04e-06 |

**And on healthy sectors it costs accuracy**, never gaining more than about 2x and
frequently losing one to three decades:

| case, order 11 | none | best transformed |
|---|---|---|
| lih_hf particle | 2.57e-15 | 1.31e-12 *(support)* — **500x worse** |
| water_hf particle | 5.28e-15 | 5.13e-14 *(support)* |
| ozone_pbe hole | 6.66e-15 | 6.43e-14 *(support)* |

**The mechanism is the transform's own arithmetic.** ROADMAP 3.2 already noted that
`MBLSE` runs a block Lanczos recurrence and **never forms a Hankel matrix** — so §3's
`cond(H0)` gains apply to a matrix the production path does not build. In exact arithmetic
the transform is a similarity and the recursion is invariant under it; in floating point
all it adds is the binomial sum `Σ_k C(n,k)(−μ)^{n−k} T_k`, which for μ ≈ −23 and n = 11
cancels terms of order 1e15 against each other. That is the loss, and it explains why the
damage tracks |μ|: water and ozone, with the largest shifts, degrade most.

So §3's headline result — a 13.7x to 2.3e6x conditioning gain — is real but **irrelevant to
this code**. It would matter to a route that forms the Gram, which is the pencil (§6.B),
and there it is already load-bearing rather than optional.

This closes the two ROADMAP 3.2 items *"affinely center and scale the hole and particle
spectral sectors separately"* and *"transform raw monomial moments to the scaled basis
before realization"*, on measurement rather than on preference.

### B. The pencil — measured, and the answer is narrower than proposed

**Step 1 is done**, by [`pencil_vs_mblse.py`](baseline/studies/pencil_vs_mblse.py): both
routes given the *same* real moments from the production dRPA path, compared on moment
reconstruction and on frontier quasiparticle energies through the same Dyson solve.
`nmom_max = 3, 5, 7, 9, 11` on lih_hf, water_hf and ozone_pbe, compression off.

**Neither route dominates, and the split is clean.**

Where the recursion is healthy — water, ozone, and lih_hf up to order 5 — `MBLSE` holds
flat at 3e-15 to 9e-15 at *every* order, and the pencil degrades with order exactly as §4
predicts:

| system, order 11 | `MBLSE` hole / particle | pencil hole / particle |
|---|---|---|
| water_hf | 6.5e-15 / 5.3e-15 | 2.1e-11 / 4.2e-10 |
| ozone_pbe | 6.7e-15 / 8.1e-15 | 1.3e-13 / 1.9e-10 |

**`MBLSE` wins by three to five decades there.** That is the expected result and it settles
the backend question: the pencil is not a replacement.

Where the recursion *stalls*, it reverses. On lih_hf the hole sector stops advancing at
order 7 and never recovers:

| order | `MBLSE` hole | poles | conserved | pencil hole | poles |
|---|---|---|---|---|---|
| 5 | 5.4e-15 | 57 | 6/6 | 2.0e-14 | 57 |
| 7 | **2.6e-07** | 57 | **6/8** | 8.1e-13 | 68 |
| 9 | **2.0e-06** | 57 | **6/10** | 1.1e-12 | 68 |
| 11 | **7.0e-06** | 57 | **6/12** | 1.4e-11 | 68 |

The recursion keeps 6 moments however many it is given, and its reconstruction error climbs
to 7e-6. The pencil finds the 11 extra poles and holds all 12 moments to 1.4e-11.

**Two things stop this being a bigger claim than it is.**

- **A stall is already reported, and already diagnosed.** `max_cycle_achieved = 2`
  against a requested `max_cycle = 5`, and `nmom_conserved = 6`, which `_frontier_from_solvers`
  ([`gw.py:550`](momentGW/gw.py#L550)) already reads into the frontier readout, alongside
  `nmom_conserved_requested` and `nmom_conserved_achieved` in the Dyson diagnostics
  ([`gw.py:134-135`](momentGW/gw.py#L134-L135)). The pencil is **not** detecting an
  unreported failure here — the premise of the old step 2 was wrong for stalls. What it
  does is *proceed past a failure the code already admits to*.

  **That holds for stalls only, and there is a second failure mode it does not cover.**
  ROADMAP 3.3 now records lithium-hydride's particle sector conserving 20 of 20 at
  `nmom_max = 19` with a residual eighteen orders above the healthy band — a realization
  that passes its gate while not reproducing the moments it was given. Nothing gates on
  that, and a residual gate in `dyson_diagnostics` is recorded there as the next thing to
  do. So "already reported" is true of the conserved count and false of the residual.

  ROADMAP 3.2 goes further and names the cause: `MBLSE.kernel` steps down in exactly one
  place, **a PSD failure on the next block's square root**, and loosening the gate
  (`neg_atol`/`neg_rtol`) by 1e4 already buys lithium-hydride 2 more orders at 60x the
  residual. So the pencil is not escaping the limit — it has no PSD gate, deflating on
  eigenvalue magnitude instead, and sits *further along the same trade* the loosened gate
  makes. That is a weaker and more accurate claim than "a third route gets past it".

  One number needs care when reading this against the roadmap's tables. Those measure the
  residual at the **achieved** order, which is why they stay at ~1e-15 through every
  step-down; the 7.0e-6 here is measured against the **requested** moment set — the error
  in what the caller asked for and did not get. Neither is wrong, and only the second says
  what a step-down costs.
- **It barely matters physically at these orders.** Frontier energies from the two routes
  agree to ≤1.1e-8 eV everywhere, and the stalled lih_hf case differs by 5.6e-7 eV on the
  HOMO. The 5 extra decades of moment fidelity buy sub-µeV.

**Revised proposal.** Not a backend (it loses where the recursion is healthy), and not a
diagnostic (a stall is already diagnosed). The remaining niche is a **fallback for sectors
the recursion has failed on** — worth taking only if a failure is found that costs more
than a µeV.

### 6.B.1 The sweep: the condition is met, and the window is narrow

[`stall_sweep.py`](baseline/studies/stall_sweep.py) sweeps the five Milestone 0 systems at
both starting points, orders 3 to 21 — past `nmom_max = 19`, where ROADMAP 3.3 finds all
three acceptance-gate systems pin. It counts both failure modes 3.3 distinguishes: a
step-down, and a realization conserving everything asked while not reproducing it. Each
failure is priced by realizing the same moments with the pencil and differencing the
frontier through the same Dyson solve — but **only where the pencil itself clears
`RESIDUAL_MAX`**, since §4 predicts it degrading with order and it does.

```
56 failing cases.  14 priced.  42 not priced (pencil worse than the failed recursion).
Largest priced cost: 4.622e-05 eV  (hydrogen_hf, K = 11)
```

**The condition is met.** 46 µeV is 46x the bar, and 7 priced failures exceed it. So the
pencil is not closed out.

It matters more than 46 µeV sounds, because of *which* system it is.
[`baseline/systems.py`](baseline/systems.py) puts hydrogen in the set precisely because
"nine RPA poles, so the moment expansion is converged by order 7 — moment-truncation error
can be driven out of the comparison here". On the one system where truncation is designed
out, this failure is the **leading** error term rather than a rounding detail. Everywhere
else it sits four orders below truncation and does not matter.

**But the window is narrow, and it closes exactly where it would be most useful.** Raising
the cap from 15 to 21 doubled the failures found, 28 to 56, and added **no** priced rows:

| K | typical `MBLSE` recon | typical pencil recon |
|---|---|---|
| 9–15 | 1e-07 to 4e-05 | 1e-13 to 2e-09 |
| 19–21 | 1e-12 to 1e-04 | 2e-03 to 1e+05 |

Past the pinning order the pencil is worse than the thing it would replace, on every system.
Its usable range is **K ≤ 15**, and the recursion's failures are worst above that.

**Verdict: keep it, scoped.** A fallback for failed sectors at K ≤ 15, whose entire measured
value is on a system built to have no truncation error. That is a real but small niche, and
the next step is a cost/benefit call on carrying a second realization route for it — not
implementation.

One limit on the sweep, stated because it looks like a disagreement otherwise: this study
builds moments fresh at each order, while `order_convergence.py` builds once at the cap and
slices. 3.3 records that the sliced route amplifies roundoff without bound once a
realization stops reproducing its moments — the same `K = 19` particle reading 7.7e+03,
3.3e+04 and 3.5e+04 at three different caps. This sweep therefore does not reproduce those
magnitudes and is not evidence for or against them.

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
python -m baseline.studies.moment_noise                         # the §4.1 gate
python -m baseline.studies.pencil_vs_mblse                      # the §6.B head-to-head
python -m baseline.studies.affine_recursion                     # the §6.A test
python -m baseline.studies.stall_sweep                          # the §6.B.1 sweep
```

`baseline/arrays` is gitignored and reproducible, so a fresh worktree has none. Either run
`python -m baseline.run` or point at a tree that already has them:

```bash
python -m baseline.studies.hankel_pencil --arrays ../../../baseline/arrays
```

Run from the repository root, and read the printed `momentGW.__file__` before believing any
cross-tree comparison — a driver script kept elsewhere silently imports the main tree.
