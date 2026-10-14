# Tensor hypercontraction: what was tried, what changed, and what to build

Written 2026-08-25 from a read of `~/Desktop/Old THC` in date order, the PyTHC repository
and preprint ([arXiv:2608.17885](https://arxiv.org/abs/2608.17885),
[QuantumCorrelators/pythc](https://github.com/QuantumCorrelators/pythc)), and
`Tensor_hyper_contraction__THC_/main.tex`.

## The requirement

**Every step on a THC path must be `O(N³)` in operations and `O(N²)` in memory. That is the
entire reason this work is being done.** A THC route that is merely faster than the
density-fitted one, but still quartic, has no reason to exist: the density-fitted path is
already there, simpler, and carries a certified error budget. THC is worth its accuracy
loss and its rank-squared prefactor only because it changes the exponent.

The corollary matters as much as the rule. The prefactor is set by the THC rank, which is
a property of the *construction* — measured at 10 to 13 × `nao` today (§4.3) and expected
to improve as construction algorithms do. **An exponent, by contrast, no improvement in
the integrals can rescue.** So a bad constant is a reason to wait; a bad exponent is a
defect. This is also recorded in [`CLAUDE.md`](CLAUDE.md), where every session sees it.

It has already caught one: `thc.dTDA.build_se_moments` was `O(N⁴)` in operations and
`O(N³)` in memory, inherited unchanged from upstream and untouched by the first pass of
work here. §5, M1.

Implemented on this branch: the PyTHC builder and its handoff (§4.2, M0.1), the dTDA
housekeeping and the cubic self-energy build (M1), and the whole separable `Q` layer
(M2). The dRPA solver itself (M3) is designed but not written.

Every number in §3 is reproduced by
[`baseline/studies/thc_separable_q.py`](baseline/studies/thc_separable_q.py), every number
in §4.3 by [`baseline/studies/thc_rank.py`](baseline/studies/thc_rank.py), and every number
in M2.2 by
[`baseline/studies/thc_rpa_assembly.py`](baseline/studies/thc_rpa_assembly.py) and every
number in M3 by
[`baseline/studies/thc_rpa_frontier.py`](baseline/studies/thc_rpa_frontier.py), on this
machine (macOS, Accelerate, PySCF 2.14.0, `mgw-monomial` and `mgw-thc`,
`mkakcl/dyson@73cd18d`). The numbers in §2.4 are read off the recorded alkane runs in
`~/Desktop/Old THC/THC/thc_tda_alkanes_test/*/all_alkanes_metrics.npz`.

---

## Summary

Two things were asked for: put PyTHC into the workflow, and finish THC dRPA. The survey
changes the second one substantially.

1. **The one-year-old THC dRPA design is no longer the right design, and the reason is a
   change on our side.** Every attempt from `take1_thc_rpa` (2025-09) to
   `RPA_Solids_THC_Working_Q` (2026-03) spends almost all of its complexity partitioning
   the particle-hole pairs into `D < z` and `D > z` regions at each quadrature point, so
   that a truncated series can be used on one side and a Laplace/cosine integral on the
   other. That partition is *not* required by THC. It is required by the oscillatory inner
   quadrature, which cannot cover a wide range of `D` at one scale. Replace the inner
   quadrature with a fitted, shared exponential grid and the partition disappears — along
   with roughly 900 of the 1461 lines of the most recent version.

2. **The kernel those attempts already had is the right kernel.** `get_D_exp_list` — the
   binomial expansion of `D^m exp(-tD)` into like-signed products of orbital energies — is
   measured here as the only one of three candidate routes that survives to high moment
   order. Keep it. §3.3.

3. **What monomial GW changed:** `momentGW.rpa._build_zeroth_dd_moment_hht` is the doc's
   "new integral" (`main.tex` §"Returning to the new integral"), with certified HHT poles
   in place of an optimised Clenshaw-Curtis grid. `optimise_main_quad`, `optimise_q_quad`,
   `get_optimal_quad_q` and the whole grid-scale optimisation the old THC code inherited
   are gone from the production path. The THC work should target the pole route, not the
   quadrature route it was written against.

4. **The cost case is weaker than the THC framing suggests, and it is now measured
   rather than assumed.** Solver cost goes as the *square* of the THC rank, and §4.3
   measures what rank each PyTHC construction needs: **about 11 × `nao`** to hold every
   baseline system inside a few meV on the frontier. At that rank the predicted crossover
   against the RI dRPA is near **990 orbitals**; 8 × `nao` moves it to 430 but costs
   88 meV on the hardest system. The rank does not fall with system size, and the grid is
   already near its floor at 33 exponents — so neither lever has much left. The recorded
   alkane runs agree in spirit: THC dTDA beat Cholesky dTDA by only 1.76× at `nmo = 394`.

5. **The measured first bottleneck is not the solver at all, it is building the THC.**
   In the alkane runs the in-tree `Gen_integrals` took **1515 s at C16 against 262 s for
   the whole THC-dTDA GW step** — six times the calculation it feeds. That is the concrete
   thing PyTHC integration fixes, and it is why the order asked for (PyTHC first) is right.

---

## 1. The two reference points

### 1.1 PyTHC

`LS_RI_THC` is the abstract least-squares/ISDF base: choose interpolation points, form the
collocation matrix `X`, build `Z = D^T D` with `D = E S^{-1}`, where `S` is the co-density
metric and `E` the co-density contracted with density-fitted three-centre integrals. The
concrete classes differ only in how the points are chosen:

| class | point selection | reference |
| --- | --- | --- |
| `LS_RI_Becke` | none; the whole Becke grid | Hohenstein/Parrish/Martinez |
| `LS_RI_Cholesky` | randomly-pivoted Cholesky on the metric, matrix-free | Matthews 2020 |
| `LS_RI_QRCP` | sketched (FFT) QR with column pivoting | Lu and Ying 2015 |
| `LS_RI_KMeans` | density-weighted k-means per atom, **rank set directly** | Lee/Lin/Head-Gordon 2020 |
| `LS_RI_NNLS` | pivoted Cholesky, then NNLS re-weighting against the overlap | — |
| `LS_Aux_Becke` | auxiliary-basis fit; no 3- or 4-index grid contraction | Hillers-Bendtsen/Martinez 2025 |
| `LS_snRI_Cholesky` | as Cholesky, but RI on the *real-space* Coulomb kernel | Zhang 2026 |

Three of these matter to us for reasons the in-tree code cannot supply:

- **`LS_RI_KMeans` takes `ips_per_naux`** — the interpolation-point count as a multiple of
  the auxiliary basis size. That is direct rank control. `LS_RI_QRCP` takes a tolerance
  that also maps onto rank more predictably than a Cholesky threshold does.
- **`LS_snRI_Cholesky` and `LS_Aux_Becke` avoid the `O(N⁴)` three-centre grid
  contraction** — the step that dominated the alkane timings — by resolving the Coulomb
  kernel on the grid instead (`int2c2e` between a `fakemol` of grid charges and the
  auxiliary basis).
- `ThcEri.get_X_Z()` returns exactly the `(rank, nao)` and `(rank, rank)` pair momentGW's
  `thc.Integrals` calls `coll` and `cou`. **The conventions already match**; see §4.1 for
  the one thing that does not.

`pythc.methods.rpa` also contains a THC-RPA *correlation energy*: `thc_rpa` (a numba
`Pi(iω)` build, `O(N_grid² N_occ N_vir)` per frequency) and `thc_rpa_spacetime`
(imaginary-time `M(τ)` then a cosine transform to `M(iω)`). The second is the same
imaginary-time factorisation this document proposes for the moments, done for a scalar
energy. It is a useful cross-check target, not something to import: it computes an RPA
energy, not moments of the density-density response, and its `RPA.kernel` depends on
`laplace_minimax`, which is not in its own dependency list.

### 1.2 The preprint

*Benchmarking Least-Squares Tensor Hypercontraction Techniques for Molecular Systems*,
Paulicks and Tölle, arXiv:2608.17885 (18 Aug 2026). It is a systematic accuracy/efficiency
comparison of the THC construction algorithms above across molecular systems, and PyTHC is
the reference implementation released with it. It is a paper about *building* the
factorisation, not about using it in a many-body method — which is the division of labour
we want: take their construction, keep our solver.

---

## 2. The old work, in date order

Paths are relative to `~/Desktop/Old THC/THC`.

### 2.1 THC dTDA: 2023-11 to 2026-03

`k-thc-tda-mGW/THC` (2023-11) is a k-point `KIntegrals` plus a `dTDA` whose
`build_dd_moments` is the ancestor of the `dTDA` now in
[`momentGW/thc.py`](momentGW/thc.py). `working_but_messy_thc.py` (2025-01) is the same with
pickling added. `changed_order_left_right` (2026-03) is the latest k-point version, by then
carrying `build_zeroth_dd_moment` / `build_nth_dd_moment` per `q`, an inline convolution and
a head/wing correction.

The molecular line ends at `pre_final_version/old_thc_tda_molecules` (2026-01-07), which
contains a **better recursion than the one currently in the repository**:

```text
Dcal[m] = Z · (V^T D^m V)                       # one binomial build per order
moments = Dcal.copy()
for i in 1..nmom:
    moments[i] += sum_{j<i} 2·Dcal[i-1-j] · moments[j]
```

against `momentGW/thc.py`'s `build_dd_moments`, which threads three intermediate arrays
(`cou_d_left`, `cou_d_only`, `cou_left`) through a doubly-nested loop to reach the same
result. The short form is `main.tex` eq. for `W̄^(m)` written directly. Worth taking on its
own merits, independently of anything below.

`old_thc_SE` (2026-01-16) is a `build_se_moments` that convolves order by order instead of
building the full `(nmo, nmom+1, nmo, nmo)` `eta`. On the recorded alkanes the THC path
peaked at 5.68 GB against Cholesky's 1.63 GB, so this is not cosmetic.

### 2.2 THC dRPA: 2025-09 to 2026-03

The whole line shares one architecture. Per outer quadrature point `z`, build

```text
Q^(m+1)(z) = V^T [ D^(m+1) / (D² + z²) ] V ,     m even, in (aux, aux)
```

then accumulate `N^(m) += w · (I + 4Q^(1))^{-1} Q^(m+1) · 2/π`, and close with the odd/even
moment recursion. That accumulation is `main.tex` eq. `eq_compact_integral`, and it is
correct — `rpa_new_clean` (2025-10-28) is the same thing without THC and it works.

What varies is how `Q^(m+1)` is obtained, and it is always a three-way split of the
particle-hole pairs by `D` against `z`:

- `D ≪ z` — the asymptotic series `D^(2r+1)/(D²+z²) = Σ_c (-1)^c D^(2r+1+2c)/z^(2c+2)`
  (`build_small`), needing `Dcal` at powers far above the moment order. `self.expan` is the
  number of extra terms: **41** in `thc_26_10`, 51 in `small_Q_working_THC`, 12 by the
  November refactors.
- `D ≫ z` — the exact finite part `Σ_{k<r} (-z²)^k Dcal^(2r-1-2k)` (`build_large`).
- the remainder — `(-z²)^r · (2/π)∫cos(zp) M(p) dp` with
  `M(p)_PQ = [Σ_i X_iP X_iQ e^{e_i p}]·[Σ_a X_aP X_aQ e^{-e_a p}]`, evaluated on an inner
  quadrature whose scale is re-optimised at every outer point
  (`build_correction_large`, `optimise_q_quad`).

The evolution is a steady fight with the bookkeeping of that split, never with the physics:

| date | file | what changed |
| --- | --- | --- |
| 2025-09-01 | `thc-drpa-mGW/take1_thc_rpa` | first version; a per-pair `book` array tracks which `(i,a)` have been moved between branches |
| 2025-10-13 | `thc-drpa-mGW/working_drpa_thc_code` | `book` replaced by index slices; `build_large_1/2/3` for one, two or three rectangles |
| 2025-10-15/16 | `working THC`, `working_THC_2` | two independent inner quadratures |
| 2025-10-22 | `pre_combine_integrals_thc`, `final_pre_decou_removal_thc` | integrals combined; the Cholesky factorisation of `Z` (`decou`) removed |
| 2025-10-26 | `thc_26_10` | splits precomputed once into a `splits` array indexed by quadrature point; inline check of every `Q[m]` against an exact `O(N⁴)` evaluation |
| 2025-10-27 | `small_Q_working_THC` | `Q` halved — only even orders stored; `get_D_alt` vectorised |
| 2025-10-29 | `THC_pre_one_integral_type` | numba `poly_builder` for the binomial contraction |
| 2025-11-06/12 | `midway_refactor_THC`, `thc_tail_attempt`, `THC_take2_pre_cleanup` | `get_D_start`/`get_D_alt2` split the binomial by parity to halve the `cou_occ`/`cou_vir` working set; a `tail` branch |
| 2026-01-03 | `RPA_solids_thc_logic` | the periodic plan: do both integrals over the full `nk²` rather than per `q`, because `Z` differs per `q` and a per-`q` split would force a resummation |
| 2026-02-02/04 | `rpa_thc_solids_wip_2_2_26`, `backup_get_D_mask_start` | index slices replaced by **boolean masks on the orbital indices**, which is the honest form: separability means the region can only ever be a union of rectangles in `(i,a)` |
| 2026-03-04 | `thc_drpa_pbc_code/RPA_Solids_THC_Working_Q` | three mask sets (`series` / `low_int` / `high_int`), an exact `Q_ERI` reference computed alongside for validation |

### 2.3 Where it broke

The `8_alkane` log (2025-10-28) records the failure directly. At quadrature points where
only one region is active the inline check passes:

```text
point 2.6834535605444727
One integral
final True 2.47955322265625e-05        # allclose, max abs diff
```

At points where the split is active it does not:

```text
point 1.6810945450652695
Two integrals
final False 0.0013410896062850952      # relative error 2.6e-3 in Q[10]
```

That is the boundary, not the kernel. It is the reason for `expan = 41`, for `d_thres`,
`integral_switch`, `sort_high_cut`, `sort_low_cut`, `get_splits_logic_low`, and for the
several thousand lines that exist only to decide which pairs go where.

### 2.4 What the alkane benchmarks measured

From `thc_tda_alkanes_test/All_THC_TDA/all_alkanes_metrics.npz` and
`RPA_TDA_no_THC/all_alkanes_rpa_tda_metrics.npz`, C4 to C16 (`nmo` 106 to 394):

| quantity | C4 | C8 | C12 | C16 |
| --- | ---: | ---: | ---: | ---: |
| THC dTDA GW step (s) | 5.9 | 30.5 | 111.1 | 262.2 |
| Cholesky dTDA GW step (s) | 9.6 | 50.4 | 175.8 | 462.2 |
| **THC construction (s)** | **2.7** | **42.5** | **306.3** | **1515.3** |
| Cholesky integral construction (s) | 0.2 | 1.8 | 7.6 | 52.4 |
| THC rank / `nmo` | 6.6 | 8.7 | 9.9 | 10.2 |
| THC peak memory (GB) | 0.11 | 1.02 | 2.91 | 5.68 |
| Cholesky peak memory (GB) | 0.10 | 0.35 | 0.77 | 1.63 |

Three things follow. The solver gain is real but modest (**1.76× at C16**). The
construction is the bottleneck and scales as roughly `N^4.3` — 1515 s against 262 s of
actual GW. And the rank ratio **grows with system size** under a fixed Cholesky threshold,
6.6 → 10.2, which is the worst possible behaviour given that solver cost goes as the square
of the rank.

---

## 3. The design decision, measured

Reproduce with `python -m baseline.studies.thc_separable_q`.

### 3.1 What the HHT route actually needs

`momentGW.rpa.dRPA._hht_apply` evaluates

```text
eta0·V = Σ_j w_j (I + 4 G_j)^{-1} Y_j ,    Y_j = V f_j ,    f_j = D/(D² + s_j) ,
                                            G_j = Y_j V^T
```

over ~18 certified poles at the default `eta0_tol = 1e-14`. Projected into the auxiliary
space this is exactly `main.tex` eq. `eq_compact_integral` — so the only system-sized
objects a THC dRPA has to produce are

```text
Q^(m+1)(s) = V^T [ D^(m+1)/(D² + s) ] V ,   m = 0, 2, …, nmom_max-1,
```

and the odd moments follow from the shift-independent `Dcal^(j) = V^T D^j V` by the
recursion in `rpa_new_clean.build_dd_moments`. Nothing else in the dRPA path touches the
particle-hole index.

### 3.2 One shared exponential grid replaces the split

Fit, per pole, on the *true* particle-hole gaps with relative-error weighting:

```text
D/(D² + s)  ≈  Σ_k c_k(s) exp(-t_k D) ,     exp(-t_k D) = exp(t_k(e_i - μ))·exp(-t_k(e_a - μ))
```

The expensive factor is the Hadamard product of the two auxiliary-space matrices, and it
**depends on `t_k` alone** — not on the pole and not on the moment order. Build the grid
once; every pole is then a linear combination of it. Referencing to the chemical potential
is load-bearing: it bounds both exponential factors by one and makes §3.3's binomial a sum
of like-signed terms.

Worst relative error over the whole pole set, benzene/cc-pVDZ, geometric grid over
`[10⁻²/D_max, 30/D_min]`. Amplification is `Σ_k |c_k| / max g`, how far the fit leans on
cancellation:

| `n_t` | on `g` | on `Q^(1)` | amplification |
| ---: | ---: | ---: | ---: |
| 20 | 1.8e-04 | 2.9e-06 | 1.3e+05 |
| 30 | 1.5e-06 | 3.2e-08 | 2.5e+04 |
| 40 | 1.8e-09 | 4.7e-11 | 3.9e+02 |
| 60 | 2.2e-10 | 7.1e-12 | 2.8e+01 |

There is no oscillatory integrand, no scale to optimise per point, and no dependence on
where `D` sits relative to anything.

**One scalar number certifies every moment order.** With `δ` the residual and
`ε = max_ia |δ(D_ia)| / g(D_ia)`, the error in the operator is
`ΔQ^(m+1) = V diag(D^m δ) V^T`, so for any unit `u`

```text
|u^T ΔQ^(m+1) u|  ≤  ε Σ_ia D^m g(D_ia) (V^T u)_ia²  =  ε · u^T Q^(m+1) u ,
```

because `D^m g(D) > 0` makes `Q^(m+1)` positive semidefinite. Hence
`‖ΔQ^(m+1)‖₂ ≤ ε ‖Q^(m+1)‖₂` at **every** `m` — which is why route B in §3.3 is flat in
the moment order rather than degrading. Measured against a dense reference on the baseline
systems, the realised operator error runs one to two decades *inside* that bound
(`tests/test_laplace.py::test_operator_bound_holds_at_every_order`).

**The grid is selected, not laid out.** The least-squares effective rank of the exponential
basis saturates near 30 on a real gap distribution — measured on ozone, 27 to 34 whether
the grid holds 48 points or 120 — so a geometric grid reaches the floor only by
oversampling. Choosing the exponents by pivoted QR on the basis reaches it with far fewer:

| `n_points` | geometric (benzene) | pivot-selected |
| ---: | ---: | ---: |
| 36 | 4.4e-08 | 1.4e-10 |
| 42 | 4.5e-09 | **1.0e-12** |
| 60 | 1.9e-11 | 2.9e-12 |

[`momentGW/laplace.py`](momentGW/laplace.py) does the selection and the fit, and searches
for the smallest grid meeting a requested residual. Measured sizes at `eta0_tol = 1e-14`
(18 to 19 shifts on the Hartree-Fock references this study builds; a PBE reference needs
21 to 26, see M2.2):

| system | residual 1e-6 | 1e-9 | 1e-11 |
| --- | ---: | ---: | ---: |
| lithium-hydride | 16 | 19 | 21 |
| water | 22 | 28 | 32 |
| ozone | 23 | 28 | 36 |
| benzene | 25 | 33 | 40 |

Below a few times 1e-13 the fit is limited by float64 rather than by the grid, so
`laplace.TOL_FLOOR` is 1e-12 and a tighter request fails immediately rather than silently
returning a number the arithmetic cannot support. Note this floor sits **above** the
`eta0_tol = 1e-14` default: a THC path needs its own, looser tolerance, and saying so is
part of M3.

### 3.3 The higher powers: three routes, one survivor

Measured against an exact evaluation of the same pole sum, relative error on `N^(m)` in the
max norm, benzene, `n_t = 40`:

| `m` | A: polynomial division | B: binomial in `e_i`, `e_a` | C: fit the target directly |
| ---: | ---: | ---: | ---: |
| 0 | 5.3e-13 | 5.3e-13 | 5.3e-13 |
| 2 | 9.7e-11 | 9.7e-13 | 6.8e-08 |
| 4 | 1.0e-08 | 1.0e-12 | 2.3e-04 |
| 6 | 1.1e-06 | 1.0e-12 | 1.2e-01 |
| 8 | **1.2e-04** | **9.8e-13** | **9.3e-01** |

- **A** uses the exact identity
  `D^(2r+1)/(D²+s) = Σ_{k<r} (-s)^k D^(2r-1-2k) + (-s)^r D/(D²+s)`. It is by far the
  cheapest — two auxiliary builds per grid point and one set of `Dcal` for the whole
  calculation — and it is what `build_large` + `build_correction_large` already amount to.
  It loses about **1.5 decades per moment order** to cancellation between the `(-s)^k` terms
  and does not recover with a finer grid. Not usable past low order.
- **B** expands `D^m = Σ_p C(m,p)(e_a-μ)^(m-p)(μ-e_i)^p`, exact and separable term by term,
  at `(m+1)` auxiliary builds per grid point. **Flat in the moment order at ~1e-12.** This
  is `get_D_exp_list` — the old code already had it.
- **C** fits `D^(m+1)/(D²+s)` on the same grid. It saturates: at `n_t = 100` it is no better
  than at `n_t = 40`, because a growing function cannot be built from decaying exponentials
  without cancellation float64 will not carry. **Rejected on measurement.**

So the old kernel stays and the machinery around it goes. Route B is now implemented as
[`momentGW/thc_rpa.py`](momentGW/thc_rpa.py) and measured against a quartic reference on
real THC factorisations rather than an RI vertex; see M2.2 in §5.

### 3.4 What it costs, and where that leaves the case for building it

Cost model, both sides counting the whole moment set, and both counting the passes their
implementations actually take (`_hht_apply` takes three per pole, `build_dd_moments` two
per order):

```text
separable :  n_t (nmom+1) N_thc² N  +  ((nmom+1)²/2) N_thc³  +  smaller terms
RI        :  (3 n_poles + 2 nmom) N_aux² N_ov
```

Crossover orbital count, holding benzene's shape ratios fixed, `nmom_max = 9`, 18 poles at
`eta0_tol = 1e-14`:

| THC rank | `n_t = 24` | `n_t = 33` | `n_t = 42` |
| --- | ---: | ---: | ---: |
| 5 × `nao` | 114 | 134 | 155 |
| 8 × `nao` | 380 | 433 | 486 |
| 10 × `nao` | 686 | 769 | 853 |
| 12 × `nao` | 1121 | 1241 | 1361 |

**This is where the case for a THC dRPA is decided, and §4.3 does not decide it
favourably.** The measured rank for a few-meV frontier agreement on the *hardest* of the
baseline systems is about **11 × `nao`** — 8.7 on lithium-hydride, 10.4 on water, 11.4 on
ozone, for the most efficient construction. At 11 × `nao` and 33 exponents the model puts
the crossover near **990 orbitals**. Dropping to 8 × `nao` moves it to about 430, but
costs 88 meV on lithium-hydride, whose diffuse LUMO is the first thing the fit loses.

Two levers move it and both are now measured rather than assumed. `n_t` is linear and
already close to its floor: 33 exponents at a `1e-9` residual, 40 at `1e-11`, and the
selection in §3.2 has taken most of what was available. **The rank is quadratic and is the
whole ballgame**, which makes §4 the deciding work rather than the enabling work.

---

## 4. Part 1 — PyTHC in the workflow

### 4.1 What already matches, and the one thing that does not

`thc.Integrals` wants `coll` as `(rank, nao)` and `cou` as `(rank, rank)`; `ThcEri.X` and
`ThcEri.Z` from `build(mode='ao')` are exactly that. `transform` then rotates `coll` by
`mo_coeff`, so **`mode='ao'` is the required mode** — momentGW needs `Lp`, `Lx`, `Li` and
`La` from one factorisation, which the `'ov'` mode cannot supply. PyTHC's `'ov'` mode is
for its own MP2 and is not usable here.

The mismatch is dependencies. `pythc.lib` imports `cotengra`, `opt_einsum`, `pytblis`,
`matplotlib` and `psutil` at module scope, and everything in `pythc` imports `pythc.lib`;
the package also pulls `numba`, `scikit-learn`, `pymongo`, `basis-set-exchange`,
`gitpython`, `pyinstrument`, `python-dotenv`, and `acccholesky` from a git URL. None of
those are in `mgw-monomial` today, and momentGW currently needs none of them. Version pins
are compatible (`numpy <2.6`, `scipy <2`, `pyscf <3`, Python ≥3.11 against our 2.3.5 /
1.17.1 / 2.14.0 / 3.11.15), so this is a size question, not a conflict question. §6 asks
which way to take it.

### 4.2 Shape, and the staged environment

Decided 2026-08-25: **PyTHC lives in a separate environment for now, handing over an HDF5
file, and folding it into `mgw-monomial` is a later step of this plan rather than a
different plan.** `baseline/` is recorded against `mgw-monomial`, and ten new packages —
one of them a git dependency, two of them needing a compiler on this machine — is not
something to put underneath a reference without a reason.

[`momentGW/thc_build.py`](momentGW/thc_build.py) is written to serve both stages from one
implementation. It **imports nothing from momentGW**, so running it as a file under the
PyTHC environment never executes `momentGW/__init__.py`:

```bash
/path/to/pythc-env/bin/python momentGW/thc_build.py \
    --atom "..." --basis cc-pvdz --method cholesky --out water.h5 --check-eri
```

while in an environment that has both it is an ordinary `from momentGW import thc_build`.
The file it writes is the layout `import_thc_components` already reads — COQUI's, with a
trailing real/imaginary axis and a leading k-point axis — so a built factorisation is a
recordable artefact that can be diffed against a COQUI one, and the two stages differ only
in when the file is produced.

The in-tree `Gen_integrals` from `pre_final_version/molecules/thc.py` is **not** ported.
PyTHC's `LS_RI_Cholesky` is the same algorithm with a matrix-free randomly-pivoted Cholesky
and a blocked three-centre contraction; `LS_snRI_Cholesky` and `LS_Aux_Becke` avoid that
contraction altogether.

The acceptance question for this part is not "does it import" but **what rank each PyTHC
method needs to hit a fixed ERI error, and what it costs to get there** — measured on the
alkane series against the 1515 s / 10.2× rank the in-tree factoriser recorded, since that is
the number to beat and the number §3.4 turns on.

#### Environment notes (this machine, 2026-08-25)

`mgw-thc`, **Python 3.12**, conda-forge. PyTHC declares `requires-python = ">=3.11"` but
uses PEP 695 `type` statements (`pythc/tracking/experiment_run.py`, `pythc/thc/thc_base.py`),
which are a syntax error before 3.12 — so 3.11 gets as far as the first import and no
further. Two of its dependencies also do not install from PyPI on an Intel Mac and are
taken from conda-forge instead:

- **`llvmlite`/`numba`** — recent releases have stopped publishing `macosx_*_x86_64`
  wheels, so pip falls back to building LLVM from source.
- **`pytblis`** — publishes no wheels at all, and its CMake build fails with
  `openmp requested but not available` under Apple clang.

`pythc.lib` imports `pytblis` at module scope even though its default `einsum` backend is
`opt_einsum`, so it has to be importable regardless of whether it is used. It builds once
conda-forge's `clang`, `clangxx` and `llvm-openmp` are in the environment and `CC`/`CXX`
point at them.

One more trap, this one ours: `momentGW/logging.py` shadows the standard library `logging`
whenever the package directory is on `sys.path`, which is exactly what running a file
inside it does. `thc_build.py` removes its own directory from `sys.path` under
`__main__` before any third-party import, so the CLI works from the repository root.

### 4.3 What the constructions actually cost in rank — measured

Reproduce with [`baseline/studies/thc_rank.py`](baseline/studies/thc_rank.py). The
reference is the **density-fitted** tensor, not `int2e`: every `LS_RI_*` method fits the
density-fitted integrals, and on water in cc-pVDZ the auxiliary basis's own error is
`4.4e-3` relative — larger than most of what the factorisation contributes, and it would
hide it completely. That correction matters: measured against `int2e`, a converged THC
looks like a `5.7e-3` failure when it is in fact exact against its own reference.

cc-pVDZ, THC and mean field sharing the `weigend` auxiliary basis; `dIP`/`dEA` are against
the RI dTDA at `nmom_max = 3`. `cholesky` is shown because it is the efficient one; the
worst `dIP` across the three systems is quoted so the number is not the easy case:

| rank/`nao` | ERI rel. Frobenius | worst dIP (meV) | on |
| ---: | ---: | ---: | --- |
| 4.6 – 5.7 | 2.8e-02 – 6.9e-02 | 278 | lithium-hydride |
| 7.3 – 8.9 | 2.1e-03 – 3.6e-03 | 88 | lithium-hydride |
| 8.1 – 10.7 | 6.4e-04 – 1.0e-03 | 7.7 | ozone |
| 8.7 – 11.4 | 1.8e-04 – 3.5e-04 | 2.6 | ozone |
| 9.0 – 13.6 | 4.5e-05 – 1.8e-04 | 1.1 | lithium-hydride |

Per system, `cholesky` reaches a few times `1e-4` relative at **8.7 × `nao`** on
lithium-hydride, **10.4** on water and **11.4** on ozone. The other constructions at the
same accuracy: `snri` within a few per cent of `cholesky` everywhere, `qrcp` 8.6 to 13.9,
`kmeans` 12.6 to 21.0.

Four things follow.

1. **Frontier error tracks integral error, and the worst case is the one to quote.** At
   `2e-3` relative — rank 8 × `nao` — water and ozone are already inside 6 meV but
   lithium-hydride is out at 88 meV, because its diffuse LUMO is the part the fit drops
   first. At a few times `1e-4` every system is inside 8 meV and typically 1 to 3.
   **So about 11 × `nao` is the operating point for a few-meV target**, not 8.
2. **`cholesky` and `snri` are the efficient constructions; `kmeans` is the worst of them
   despite being the one with a rank dial.** The coarse sweep is misleading on its own —
   `cholesky` and `snri` look as though they plateau near `3e-3` only because it stopped at
   a threshold of `1e-5`; at `1e-7` and `1e-8` they go two decades further. `ips_per_naux`
   buys predictability, not efficiency: it maps directly onto the rank, which makes a
   rank-versus-error curve reportable rather than something to be discovered threshold by
   threshold, but at equal accuracy it costs about 1.7 times the rank. Note that
   `cholesky`, `snri` and `nnls` pivot *randomly*, so their ranks move by a few per cent
   between builds at the same threshold — read them as a band, and do not chase a
   difference of that size between two of them.
3. **`aux-becke` is not usable as it stands, in either variant.** Unpruned it keeps the
   whole parent Becke grid — 61 to 69 × `nao` — and is still only `2e-2` relative. Its
   pruned subclass `LSTHC_AuxCoulomb_Cholesky` reaches 3.5 to 4 × `nao`, but at *O(1)*
   relative error, and the rank does not respond to its threshold at all, because the
   pruning is against the auxiliary Gram rather than the co-density. It is the one
   construction that avoids the three-centre grid contraction, so it is worth returning to
   with its own `fit_auxbasis`, `regularization` and grid level tuned — but not on this
   pass, and not as the default.
4. **The rank does not fall with system size.** Benzene at `nao = 114`, the largest system
   here, needs the same ratio the small ones do — its dense reference tensor is too large
   to form, so the frontier is the measurement:

   | method | rank/`nao` | dIP (meV) | dEA (meV) |
   | --- | ---: | ---: | ---: |
   | cholesky | 5.58 | -164.6 | 144.8 |
   | **cholesky** | **9.96** | **-2.5** | **-0.2** |
   | snri | 11.17 | -3.7 | -2.1 |
   | qrcp | 12.45 | -2.8 | -2.4 |
   | kmeans | 12.58 | 10.9 | -3.6 |

   So `cholesky` reaches a few meV at **10 × `nao`** at `nao = 114`, against 8.7 to 11.4 on
   the small systems — the same band, with no sign of improving. The crossover in §3.4 is
   therefore not something that gets better on its own, and the ordering of the
   constructions holds at this size too.

---

## 5. Part 2 — THC dRPA, revised

Milestones, in dependency order. Each is gated on a measurement, not on code existing.

**M0 — PyTHC (§4).**

- **M0.1 the builder module and the HDF5 handoff — done.**
  [`momentGW/thc_build.py`](momentGW/thc_build.py) builds seven constructions under
  `mgw-thc` and writes the layout `thc.Integrals.import_thc_components` reads; the
  round trip is exact, and a `THC-dTDA` `G0W0` on the result agrees with the RI one to
  the factorisation error (§4.3).
- **M0.2 the rank study — done for the molecular set, §4.3.** Rank against integral error
  and frontier error for `cholesky`, `kmeans`, `qrcp`, `snri` and both `aux-becke`
  variants, on water, lithium-hydride, ozone and benzene, against the **density-fitted**
  reference. Still outstanding: the same sweep on the alkane series against the in-tree
  factoriser's 1515 s at C16, which is the *construction-time* half of the comparison.
- M0.3 **fold PyTHC into `mgw-monomial`** as an optional extra in `pyproject.toml`, so
  `thc_opts["file_path"] = None` means "build it in process" rather than raising. Gate: the
  full suite and `python -m baseline.check` unchanged after the install, since the point of
  staging it is that the reference environment is not disturbed silently. The conda-forge
  and Python 3.12 constraints in §4.2 carry over, and are the reason this is a step of its
  own rather than the default.

**M1 — THC dTDA, made cubic — done.** Two changes to
[`momentGW/thc.py`](momentGW/thc.py), the second of which is the one that matters.

*The density-density recursion* now reads
`zeta[m] = Dcal[m] + 2 Σ_{j<m} Dcal[m-1-j] Z zeta[j]` in place of three threaded
intermediates, turning `O(nmom²)` collocation contractions into `O(nmom)`. The two
recursions agree to 1.7e-12 per order through `nmom_max = 7`.

*The self-energy build was `O(N⁴)`* — the requirement above is what found it. It looped
over the `nmo` Green's function orbitals, each iteration contracting
`einsum("Pp,Qq,PQ->pq")` at `naux² nmo + naux nmo²`, into an `eta` array of `nmo³` doubles.
Neither is necessary: the orbital index only reaches the output through the weighted sums
the convolution takes, so moving that sum inside the auxiliary contraction gives

```text
Σ^(n) = Σ_{t ≤ n} C(n,t) (±1)^t · X^T ( ζ'^(t) ∘ G^(n-t) ) X ,
        G^(j)_PQ = Σ_{x ∈ sector} ε_x^j X_xP X_xQ
```

with the whole `t` sum accumulated in the auxiliary space before one rotation per order and
sector. `eta` and the orbital loop both disappear. **Operations `O(nmom² (naux² nmo + naux
nmo²))`, memory `O(nmom (naux² + nmo²))` — cubic and quadratic.**

Gates: agreement with the previous implementation to **1e-15 relative** on
lithium-hydride, water, ozone and benzene, for both `diagonal_se` settings; the
independent `test_moments_vs_cderi` passes; the full suite is 246 passed; and
`python -m baseline.check` is 52/52 with every HOMO shift exactly zero.

Measured, `nmom_max = 5`, against the implementation it replaces:

| system | `nao` | rank | time before | after | speedup | peak memory before → after |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lithium-hydride | 19 | 138 | 0.050 s | 0.006 s | 8.0x | — |
| water | 24 | 191 | 0.067 s | 0.010 s | 6.7x | — |
| ozone | 42 | 375 | 0.193 s | 0.051 s | 3.8x | 5.7 → 17.0 MB |
| benzene | 114 | 1136 | 3.727 s | 0.822 s | 4.5x | 56.8 → 156.1 MB |

**The memory constant is worse even though the scaling is better, and for the same reason
the cost model has a crossover.** The new peak is about `2 (nmom+1) naux²` against the old
`nmo³`; at a rank of 10 × `nao` those cross near `nao ≈ 1200`. That is not a defect to fix
— it is the rank showing up again, it falls as the square of the rank ratio, and the
alternative would be to keep a cubic array. The exponent is what was bought.

**M2 — the separable `Q` layer — M2.1 and M2.3 done.**
[`momentGW/laplace.py`](momentGW/laplace.py), by analogy with
[`momentGW/eta0.py`](momentGW/eta0.py): candidate exponents, pivoted selection, per-pole
relative-error-weighted fit, a search for the smallest grid meeting a residual, and
`report`. Gate met: the residual is realised rather than bounded, the operator bound of
§3.2 holds at every order to `nmom_max = 9`, the selection beats a geometric grid at equal
size, and an unrepresentable tolerance raises. 16 tests in `tests/test_laplace.py`.

- **M2.2 the binomial `D^m` assembly against `Li`/`La` — done.**
  [`momentGW/thc_rpa.py`](momentGW/thc_rpa.py). The first thing on this path that touches
  the vertex, and the first place the memory constant bites rather than the exponent.

  *What it computes.* `Q_X^(m+1)(s_j) = Σ_k c_k(s_j) Σ_p C(m,p) O_p(t_k) ∘ U_{m-p}(t_k)`,
  route B of §3.3, with the Cholesky factor of `Z` absent by the push-through identity so
  the solver never forms it.

  *Why it streams.* The exponent loop has to be outermost — that is what makes the grid
  shared — and the pole loop cannot be folded inside it, because `(I + 4 Q^(1) Z)^{-1}` is
  nonlinear in `Q^(1)`. Assembling every `Q^(m+1)(s_j)` before the first solve is formally
  `O(N²)` but with a constant near three hundred: **3.4 GB at benzene and 41 GB at a C16
  alkane**, on a 16 GB machine. Every step of the assembly is elementwise in `(P, Q)` and
  the solve takes its right-hand side column-wise, so the whole thing blocks over the
  second auxiliary index at a byte budget rather than a shape.

  *Gates, measured on PyTHC `cholesky` factorisations at the §4.3 operating point.*

  | system | `nao` | rank | poles | exponents | residual | assembly vs the same fit | vs the resolvent |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | lithium-hydride | 19 | 161 | 18 | 20 | 1.5e-11 | 9e-16 – 3.5e-15 | 3.6e-13 – 7.1e-12 |
  | water | 24 | 249 | 21 | 30 | 4.6e-11 | 1.2e-15 – 2.1e-15 | 5.1e-12 – 8.6e-12 |
  | ozone | 42 | 482 | 26 | 32 | 8.7e-11 | 2.2e-15 – 4.1e-15 | 1.5e-11 – 2.1e-11 |
  | benzene | 114 | 1201 | 21 | 38 | 2.6e-11 | 1.5e-15 – 3.6e-15 | 4.4e-14 – 3.4e-13 |

  Against the *same* fitted scalar the assembly is an identity — the binomial expansion of
  `D^m` is exact — and it lands at **1e-15 at every order**, which is the check that the
  expansion, the sector factorisation and the blocking are all right rather than merely
  close. Against the true `D^(m+1)/(D²+s)` it lands **inside the residual `laplace`
  measured, and flat in the moment order**: that is §3.2's bound realised on a real
  factorisation, and it is what replaces the pair-space partition. Benzene is checked on
  the first and last pole only — the quartic reference costs `naux² nov` per pole and
  order, which is ten minutes each at rank 1201.

  *The stream, measured.* Benzene, `nmom_max = 9`, 256 MB budget:

  | assembly | time | resident rise | predicted working set | block |
  | --- | ---: | ---: | ---: | ---: |
  | blocked | 10.0 s | 62 MB | 255 MB | 89 |
  | whole | 14.9 s | 3803 MB | 3445 MB | 1201 |

  **Bitwise identical results, 61× less memory, and faster** — the blocked path wins the
  wall clock too, because the working set fits cache. The blocked resident rise is *below*
  its predicted working set because a high-water mark only ever rises and the mean field
  had already set one; the prediction is the number to trust, and the unblocked row
  confirms the model to 10%. 11 tests in `tests/test_thc_rpa.py`.

  *The exponent is right; the constant has headroom.* The operation count is cubic by
  construction — `n_t (nmom_max+1) N_thc² N` for the sector builds, and nothing in the
  routine carries a pair index — but the measured wall clock is **sub**-cubic across the
  baseline systems, at an exponent of 2.2 to 2.5 in `nao`, because the achieved rate rises
  with size rather than because the work does not:

  | system | rank | GEMM | elementwise traffic | time | achieved |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | lithium-hydride | 161 | 0.3 GF | 0.3 GB | 0.110 s | 2.5 GF/s |
  | water | 249 | 1.2 GF | 1.1 GB | 0.277 s | 4.3 GF/s |
  | ozone | 482 | 7.6 GF | 4.5 GB | 1.147 s | 6.6 GF/s |
  | benzene | 1201 | 124 GF | 33 GB | 9.971 s | 12.4 GF/s |

  Against a square-`dgemm` ceiling of 140 GF/s on this machine, 12.4 is 9%. The routine is
  bandwidth-bound, and for a legible reason: the sector GEMMs are skinny — the contracted
  index is `nocc` or `nvir`, 21 and 93 at benzene against a rank of 1201 — and the binomial
  assembly itself is elementwise, 33 GB of traffic carrying no flops at all. Both are
  addressable (batch the sector builds across exponents, fuse the binomial into the
  coefficient contraction) and neither changes the exponent, so by the rule at the head of
  this document they are optimisations to consider later rather than defects to fix now.
  Recorded here so that a future timing comparison is read against the right baseline.

  Two corrections fall out of measuring this. The C16 figure quoted when M2.2 was scoped,
  11.6 GB, counted only the assembled `Q`; the grid buffer the shared-exponent loop needs
  is larger still, and the honest number is **41 GB** — which is why the stream is not
  optional at the size M4 targets.

  And **§3.2's "18 to 19 shifts" holds only for a Hartree-Fock reference.**
  `thc_separable_q.py` builds its cases with `scf.RHF`; the baseline set, `thc_rank.py`
  and any real momentGW run use PBE, whose smaller frontier gap widens the certified
  interval by one to two decades and needs more poles:

  | system | gap, RHF | poles | gap, PBE | poles |
  | --- | ---: | ---: | ---: | ---: |
  | water | 0.679 | 18 | 0.259 | 21 |
  | ozone | 0.442 | 19 | 0.063 | 26 |
  | benzene | 0.470 | 18 | 0.191 | 21 |

  The hypercontraction has nothing to do with it: the coupling bound computed through `Z`
  on the THC vertex agrees with the RI one to five significant figures and gives the
  identical pole count on all three systems. It is the mean field. The cost model in §3.4
  assumes 18 poles, and `n_poles` enters only the RI side of it, so a PBE reference moves
  the crossover **in THC's favour** — not far, since the term is `3 n_poles` against
  `2 nmom_max`, but §3.4's numbers are the conservative end rather than the middle.

**M3 — the solver — done.** [`momentGW/thc_rpa.py`](momentGW/thc_rpa.py) gains
`dRPA(thc.dTDA)`, reached as `polarizability="thc-drpa"`. It produces the same
`(nmom_max+1, naux, naux)` `zeta` that `thc.dTDA.build_se_moments` already consumes, so
nothing downstream changed: that routine applies the two outer factors of `Z` at
contraction time, which is exactly the convention the push-through identity leaves the
solver in. Three module-level pieces do the numerics, so each is testable without a `GW`
object — `polynomial_moments` (the odd family), `screened_moments` (the pole solve) and
`combine` (the recursion).

*The option.* `thc_laplace_tol`, default `1e-10`, with `thc_laplace_n_points` to fix the
grid. It cannot inherit `eta0_tol`: `laplace.TOL_FLOOR` is 1e-12 and the `eta0_tol`
default is 1e-14. `eta0_method="clencur"` now raises rather than being silently ignored —
the separable route is derived against the pole set, and there is no hypercontracted
quadrature to fall back to.

*The gates, layered so that each isolates one thing.*

1. **The recursion**, against the *pair-space* form `rpa.dRPA` implements — a reference
   that carries the particle-hole index throughout and shares none of the auxiliary-space
   route. On water/cc-pVDZ through `nmom_max = 7`, **2.5e-16 to 5.6e-16** relative at
   every order, against `rpa.dRPA.build_dd_moments` projected.
2. **The whole solver on an identical vertex.** `test_drpa_moments_vs_cderi` builds the
   Cholesky-decomposed vertex *from* the THC factorisation, so the density-fitted `dRPA`
   sees exactly the same integrals and the factorisation error is identically zero by
   construction. The self-energy moments agree to **eight decimal places** relative.
3. **The frontier, with the rank swept.** `dIP` in meV against the RI dRPA, PyTHC
   `cholesky`, `nmom_max = 3`, rank/`nao` in parentheses:

   | system | 1e-3 | 1e-5 | 1e-6 | 1e-7 | 1e-8 |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | lithium-hydride | 151.0 (5.1) | 18.7 (7.4) | 22.2 (7.9) | 3.2 (8.5) | **1.1** (8.9) |
   | water | 108.0 (4.7) | -4.5 (8.0) | 4.3 (9.2) | -0.6 (10.5) | **-0.3** (11.1) |
   | ozone | -231.5 (5.2) | -21.0 (8.7) | -0.5 (10.3) | 0.7 (11.6) | **-0.6** (13.4) |
   | benzene | -77.1 (5.1) | -1.2 (10.6) | 0.5 (13.2) | 0.5 (15.4) | **-0.0** (17.5) |

   **There is no floor.** The error falls with the rank until it is under a meV on every
   system, which is what "the factorisation error and the `Q` error are separated rather
   than added" had to mean. The stronger statement is in the numbers next to these:
   **`thc-drpa` and `thc-dtda` deviate from their own density-fitted references by the
   same amount at every rank** — -0.03 against -0.03 on benzene at 1e-8, 0.65 against 0.86
   on ozone at 1e-7. The factorisation error is shared; the dRPA solver adds nothing to
   it.

*The separable tolerance is not the binding error, and here is where it would start to
be.* `dIP` in meV at the 1e-7 rank, sweeping `thc_laplace_tol` over nine decades:

| `thc_laplace_tol` | lithium-hydride | water | ozone | benzene |
| --- | ---: | ---: | ---: | ---: |
| 1e-11 | 3.226 | -0.617 | 0.648 | 0.473 |
| 1e-8 | 3.226 | -0.617 | 0.648 | 0.473 |
| 1e-6 | 3.226 | -0.617 | 0.648 | 0.473 |
| 1e-4 | 3.207 | -0.616 | 0.649 | 0.473 |
| **1e-2** | **4.215** | -0.623 | 0.650 | 0.475 |

Flat to `1e-4` on every system — and at `1e-2` the fit really is coarse, not a request the
search quietly overshot: 9 exponents at a measured 2.7e-3 residual on lithium-hydride, 15
at 1.3e-3 on benzene, against 21 and 39 at `1e-11`. The one system that moves is
**lithium-hydride, by 1 meV**; benzene got an equally coarse fit and moved by 0.002. That
is the same system §4.3 flags as the one whose diffuse LUMO the *factorisation* loses
first, so both approximations degrade on the same feature. The separable layer therefore
has three or four decades of headroom over the factorisation, and the first place to look
when that stops being true is the diffuse frontier. The default stays at `1e-10`: loosening it to `1e-4`
saves 13% of the solver's wall clock at benzene, which is not worth spending certainty on.

*Cost, `nmom_max = 3`.* At the 1e-5 rank: 0.1 s at lithium-hydride, 0.3 s at water, 0.9 s
at ozone, 10.8 s at benzene, against 8.4 s for the RI dRPA at benzene. Slower at these
sizes, as §3.4 predicts — the crossover is M4's business, not a gate here.

*One judgement recorded.* The coupling bound the pole selection needs is `4 ||W||_F^2`,
which for a hypercontracted vertex is exactly `4 sum_PQ Z_PQ [X D X^T]_PQ` — `O(naux^2)`,
no pair index. The density-fitted route also tries `||W||_1 ||W||_inf` and keeps the
smaller; those norms need the vertex *with* its pair index, which is the `O(N^3)` array
this path exists to avoid. Measured, it costs nothing worth having: the enclosure is
dominated by `max(d^2)`, and replacing the trace with the exact largest eigenvalue moves
the pole count by **one** on all four systems.

21 tests in `tests/test_thc_rpa.py`, three more in `tests/test_thcgw.py`.

**M4 — the crossover, measured.** The alkane series, THC dRPA against RI dRPA, reporting
the rank actually used. Gate: publish the crossover, whatever it is. §3.4 and §4.3 together
predict **about 990 orbitals at a few meV, or 430 if 88 meV on the hardest system is
acceptable** — so the honest expectation is already a scoped fallback rather than a
default, and M4 exists to confirm or overturn that rather than to celebrate it.

**Deferred.** Periodic and unrestricted, per `CLAUDE.md` — the `RPA_solids_thc_logic` note
about summing over `nk²` rather than per `q` is the right starting point when that time
comes, and the mask machinery in `RPA_Solids_THC_Working_Q` should be re-read then rather
than ported now.

---

## 6. Decided, and what is still open

**Decided 2026-08-25: the work continues, and the crossover is not the criterion.** The
rank is a property of today's construction algorithms and is expected to improve; the
exponent is a property of our solver and is the thing that has to be right now, because no
improvement in the integrals can rescue it. That is the requirement at the head of this
document, and it displaces the crossover as the acceptance test: **a stage is accepted when
it is `O(N³)` in operations and `O(N²)` in memory, and rejected when it is not, whatever it
does to the wall clock at today's sizes.** The crossover is still worth publishing (M4), as
a statement about the constants rather than a gate.

Still open:

1. **What `nmom_max` should M3 target?** Cost is linear in `nmom_max` through the binomial
   and quadratic through the self-energy build, so 9 rather than 5 is a real factor on the
   whole solver.
2. **Is the RI dRPA the right comparison, or is `dTDA` the intended production path?** §3.4
   compares against RI dRPA; if THC dTDA is what actually gets run, the relevant comparison
   is a different one, and the alkane runs already put that at 1.76x in THC's favour at
   `nmo = 394` — before M1 made the self-energy build cubic.
3. **Does anything else on the path violate the requirement?** M1 found one `O(N⁴)` stage
   that had been there since the code was written. `Integrals.transform`, `get_j`, `get_k`
   and the recursion have been checked by hand and are cubic; the periodic THC path in
   `momentGW/pbc/thc.py` has **not** been, and it carries its own copies of both routines.
4. **`aux-becke`.** The only construction that avoids the `O(N⁴)` three-centre grid
   contraction, and the only lever left on the rank. Unusable as shipped (§4.3, point 3),
   and worth its own pass now that the rank is known to be the binding constraint.
