# Roadmap: making the Dyson stage cheap

Written 2026-08-06, revised the same day after the stage-level measurement in §0.2.
Supersedes §6 and §7 of [`DIAGONALISATION.md`](DIAGONALISATION.md).

Scope is the restricted molecular `G0W0`/`drpa` path with default options —
`optimise_chempot=False`, `fock_loop=False`. The self-consistent paths are out of scope
and are not costed here.

Provenance: macOS, 12 cores, Accelerate BLAS, no thread environment variables set,
NumPy 2.3.5, SciPy 1.17.1, PySCF 2.14.0, `mgw-monomial`, momentGW `master`, dyson pin
`73cd18d`. Benzene and naphthalene, PBE starting point, `nmom_max = 7`.

**Absolute wall times on this machine drift by tens of percent between processes** —
enough that cross-process totals are not comparable, and long runs appear to throttle.
Every speedup quoted below is an A/B/C measured back-to-back **in one process** against
the same mean field and the same input moments. Stage shares are also within-process.
Nothing is implemented; all numbers come from monkeypatch prototypes.

---

## 0. What the measurements say

### 0.1 A correction to the previous revision of this document

The first revision of this roadmap counted *eigendecompositions* — because that is what
`DIAGONALISATION.md` is about — and concluded the whole area was worth ~1.2x. That was
measuring the wrong thing. Counting the **Dyson stage as a whole** puts it at 35–46% of
a G0W0 run, and its largest single item is not an eigendecomposition at all.

It also repeated `DIAGONALISATION.md`'s claim that the `O(N^3)`-versus-`O(N^4)` trend
works against this area as systems grow. That is true in one direction only; see §0.3.

### 0.2 Where the time actually goes

Stage split, measured within a single process for each row:

| | `nmo` | `nocc` | non-Dyson stages | **Dyson stage** | share |
| --- | --- | --- | --- | --- | --- |
| benzene / cc-pVDZ | 114 | 21 | 4.31 s | 2.97 s | **40.8%** |
| benzene / cc-pVTZ | 264 | 21 | 23.32 s | 20.11 s | **46.3%** |
| naphthalene / cc-pVTZ | 412 | 34 | 229.09 s | 124.73 s | **35.3%** |

An independent sequential run of the same decomposition gave 40.7% and 48.3% for the two
benzene rows, so the shares are reproducible to a few points.

Inside the Dyson stage at benzene / cc-pVTZ, by cost:

| item | time | share of run | what it is |
| --- | --- | --- | --- |
| `Lehmann.moments` (`lehmann.py:371`) | 5.94 s | ~9% | one three-operand `einsum`, 15 calls |
| `Spectral` round trip | ~8 s | ~12% | 21 eigendecompositions + two large `einsum`s |
| the recurrence (`_recurrence_iteration_hermitian`) | ~3.9 s | ~6% | many modest GEMMs |
| the one real arrowhead solve (`fock.py:557`) | 1.8 s | ~3% | what `DIAGONALISATION.md` is about |

**The single largest item in the Dyson stage is an `einsum` that never reaches BLAS**,
and the second is work that does not need to happen at all. The eigensolver that
`DIAGONALISATION.md` spends forty pages on is fourth.

### 0.3 Scaling — the doc's claim is half right

`DIAGONALISATION.md` §6 says the `O(N^3)` eigendecompositions shrink against the
`O(N^4)` moment construction as systems grow. Which direction you grow in decides it:

- **Bigger basis, same molecule** (benzene cc-pVDZ → cc-pVTZ). `nocc` is fixed, so
  `nov = nocc * nvir = O(N)` and the correlated stages are `O(naux^2 * nov) = O(N^3)` —
  the *same* order as the Dyson solve, whose prefactor is `(9 nmo)^3`. The Dyson share
  **rises**, 40.8% → 46.3%.
- **Bigger molecule, same basis** (benzene → naphthalene at cc-pVTZ). `nocc` grows,
  `nov = O(N^2)`, the correlated stages are genuinely `O(N^4)`, and the Dyson share
  **falls**, 46.3% → 35.3%.

So the Dyson stage does not become negligible with basis-set extension, which is the
direction that matters for converging a published number. It stays around a third to a
half of the calculation across everything measured here.

### 0.4 What the two Tier 1 changes are worth

A/B/C in one process. `B` fixes `Lehmann.moments`; `C` additionally removes the
`Spectral` round trip.

| system | variant | Dyson stage | stage speedup | **whole run** | max abs QP shift |
| --- | --- | --- | --- | --- | --- |
| benzene / cc-pVDZ | A current | 2.97 s | — | — | — |
| | B moments → GEMM | 2.43 s | 1.220x | 1.079x | **0** |
| | C + round trip gone | 1.56 s | **1.904x** | **1.240x** | 7.3e-13 eV |
| benzene / cc-pVTZ | A current | 20.11 s | — | — | — |
| | B moments → GEMM | 14.39 s | 1.398x | 1.152x | **0** |
| | C + round trip gone | 7.32 s | **2.746x** | **1.417x** | 5.3e-13 eV |
| naphthalene / cc-pVTZ | A current | 124.73 s | — | — | — |
| | B moments → GEMM | 86.43 s | 1.443x | 1.121x | **0** |
| | C + round trip gone | 48.31 s | **2.582x** | **1.276x** | 4.9e-12 eV |

The Dyson share falls from 40.8% to 26.5% (cc-pVDZ), 46.3% to 23.9% (cc-pVTZ) and 35.3%
to 17.4% (naphthalene). The largest QP shift anywhere is 4.9e-12 eV, three orders below
the baseline's 1e-9 eV reproducibility floor, and the `B` step is **bit-identical**.

### 0.5 What the implementation delivers

The table above is the monkeypatch prototype. With §1.1, §1.2 and §1.3 actually
implemented, benzene / cc-pVTZ at `nmom_max = 7`, run adjacently against the pinned dyson
and against the working tree:

| | pinned `73cd18d` | working tree | |
| --- | --- | --- | --- |
| integrals | 6.66 s | 6.72 s | untouched |
| static self-energy | 2.59 s | 2.57 s | untouched |
| eta0 | 0.15 s | 0.15 s | untouched |
| dd + se moments | 20.98 s | 21.06 s | untouched |
| **Dyson stage** | **27.21 s (47.2%)** | **9.15 s (23.1%)** | **2.97x** |
| — eigendecomposition | 51 calls, 13.51 s | 25 calls, 4.21 s | |
| — remainder | 13.71 s | 4.93 s | |
| **total** | **57.60 s** | **39.66 s** | **1.45x** |

Ahead of the prototype's 2.75x / 1.42x, because §1.3 is done properly here. These are
separate processes, so the four untouched stages are the control: they agree to under 1%,
which is what makes the comparison usable despite §0's warning about cross-process drift.
The `9 * nmo` solve now runs **once** (2811 ms) rather than twice (5717 ms), and the
eigenvalues-only path for the log line costs 484 ms across two calls.

---

## 1. Tier 1 — exact, no accuracy sacrificed

Both changes are in dyson, so one pin bump and one baseline re-record covers them. Do
not split them.

### 1.1 `Lehmann.moments`: replace the three-operand `einsum` with GEMMs

**Status: implemented** on `mkakcl/dyson` branch `m4-lehmann-moments-gemm`, uncommitted.
Suite is **957 passed, 40 skipped, 0 failed** (919 before, plus 38 new tests). The pin has
not moved, so momentGW is unaffected so far. Verifying it turned up §2.3.

**The single biggest item in the Dyson stage, and the cheapest to fix.**

**Where.** `dyson/representations/lehmann.py:371`.

```python
subscript = "pk,qk,nk->npq"          # Reduction.NONE
moments = util.einsum(subscript, right, left.conj(), self.energies[None] ** orders[:, None])
```

`dyson.util.einsum` is `functools.partial(np.einsum, optimize=True)`, so this is not a
missing `optimize` flag — the contraction is a *weighted outer product* summed over a
single shared index in all three operands, and NumPy's pairwise optimiser cannot express
it as `tensordot`. It falls into the single-threaded `c_einsum` kernel. Measured: 5.94 s
in 15 calls at cc-pVTZ, 46% of the whole Dyson stage.

It is a GEMM. `out[n,p,q] = sum_k v[p,k] w[n,k] u*[q,k]` is one matrix product per order
once the left operand is scaled, or a single product if the orders are stacked:

```python
scaled = (right[None] * powers[:, None]).reshape(nmom * nphys, naux)
out = (scaled @ left.conj().T).reshape(nmom, nphys, nphys)
```

Microbenchmark on the real shapes, identical arithmetic in a different summation order:

| `nphys` | `naux` | `einsum` | per-order GEMM | stacked GEMM | speedup | max rel. diff |
| --- | --- | --- | --- | --- | --- | --- |
| 114 | 912 | 94.7 ms | 4.5 ms | 1.7 ms | **21–57x** | 2.4e-15 |
| 264 | 2112 | 1210.2 ms | 17.7 ms | 22.3 ms | **54–68x** | 3.3e-15 |
| 264 | 1056 | 587.5 ms | 9.3 ms | 12.7 ms | **46–63x** | 1.6e-15 |
| 510 | 4080 | 10185.0 ms | 138.8 ms | 143.4 ms | **71–73x** | 3.8e-15 |

The gap widens with size. The stacked form allocates a second `nmom * nphys * naux`
array, so prefer the per-order loop when memory matters — it is within 30% and at
`nphys = 510` it is faster anyway.

**Why the QP energies do not move at all.** On the G0W0 path the only consumers of
`moments` are the reconstructed-moment diagnostic (`reconstruct_moments`) and
`momentGW.gw.moment_error`. Neither feeds the quasiparticle energies, so `B` is
bit-identical in the results and moves only the reported diagnostic values, by ~3e-15
relative. Note that the win is **not** merely "cheaper diagnostics": with
`dyson_opts.calculate_errors=False` the fix is still worth 1.134x on the stage at
cc-pVTZ, because `momentGW`'s own `moment_error` is not gated by that option.

**Steps.** Rewrite the three reductions (`NONE`, `DIAG`, `TRACE`) as GEMMs; keep the
`einsum` as the reference in a test that asserts agreement to ~1e-14 relative on random
Lehmann representations, Hermitian and non-Hermitian, real and complex, including
`squeeze` and non-contiguous `order` arrays.

**Kill criterion.** None needed — it is measured, 20–70x, and the arithmetic is the same.
If the GEMM form is *not* faster on some platform, that is a platform bug worth knowing.

**Payoff.** 1.22x to 1.44x on the Dyson stage; 1.08x to 1.15x on the whole run.
**Effort**: under a day, plus the shared pin ceremony.

### 1.2 Remove the `Spectral` round trip

**Status: implemented** on `m4-lehmann-moments-gemm`, uncommitted. `from_self_energy` stores
a `_SelfEnergySource` and defers; `get_static_self_energy`, `get_self_energy`, `overlap`,
`hermitian` and `neig` are served from it; `eigvals`/`eigvecs` and everything downstream go
through `_diagonalise()`, which reruns `__init__` so deferred construction keeps its
validation. A `diagonalised` property makes "this did not diagonalise" testable, and
`tests/test_spectral_deferred.py` asserts it for every cheap accessor.

The exactness is algebraic, not just numerical: the physical-block orthogonalisation is
undone on the way out, so `S^(1/2)† S^(-1/2) f S^(-1/2) S^(1/2) = f` and likewise for the
border and the overlap. The three quantities come back **exactly**, not to 1e-13.

**Where.** `dyson/representations/spectral.py`; reached from `momentGW/gw.py:426` and
`dyson/solvers/static/mblse.py:237`.

dyson builds a `Spectral` by diagonalising a supermatrix, then reconstructs that
supermatrix's blocks from the eigendecomposition and re-diagonalises them to recover its
own inputs:

```python
Spectral.from_self_energy(static, lehmann).get_self_energy()         # == lehmann
Spectral.from_self_energy(static, lehmann).get_static_self_energy()  # == static
```

`get_auxiliaries` rebuilds the `(aux, aux)` block as `sum_k x_aux,k lambda_k x_aux,k^†`
— which is `diag(e)` — and diagonalises *that* to recover `e`; it rebuilds `(phys, aux)`
to recover `v`. 21 of the 51 eigendecompositions in a G0W0 run are this, plus two
`O(n * naux^2)` contractions per call. The largest solve in the calculation, `9 * nmo` —
the one `DIAGONALISATION.md` §3 measured at 12.15 s for `nmo = 384` and built Options 2
and 3 around — **happens twice**, once here and once for real at `momentGW/fock.py:557`.

Verified an identity on water / cc-pVDZ at `nmom_max = 7`:

| round trip | poles | max `dE` | max `dMoment` |
| --- | --- | --- | --- |
| sector (hole) | 96 | 1.07e-13 | 1.24e-14 |
| sector (particle) | 96 | 4.62e-14 | 3.02e-14 |
| combined, `combine_for_self_energy` | 192 | 8.53e-14 | 1.99e-14 |

and the final Dyson eigenvalues agree between the two routes to **9.2e-14 Ha**.

**Shape of the change — make `Spectral` lazy, do not special-case the call sites.**
Store `static` and the auxiliary `Lehmann` at construction; return them from
`get_static_self_energy` / `get_self_energy`; diagonalise on first access to `eigvals`,
`eigvecs`, `get_dyson_orbitals()`, `get_greens_function()` or `overlap`, and cache it.
`combine_for_self_energy` then costs a concatenation.

This is preferable to patching the two call sites because it is behaviour-preserving:
`momentGW/bse.py:330` calls `solver.result.get_greens_function()`, a genuine use of the
eigenvectors, and must keep working. It also removes the waste from the unrestricted and
periodic solvers without extending anything into them — no Milestone 6 boundary crossed.

**Steps.**

1. Store the inputs; diagonalise lazily.
2. Preserve the `chempot` and `overlap` handling `diagonalise_matrix` applies. The
   overlap path is where this can go wrong quietly — `combine_for_self_energy` passes
   `overlap = sum(spectral.overlap)`, and the identity above was verified *with* that
   overlap in play. Pin it with a test.
3. `combine_for_self_energy` concatenates. Consider sorting the concatenated auxiliaries
   by energy, matching the `sort=True` the current code passes: not enough to make the
   change bit-identical, but it keeps downstream summation order closer to the baseline.
4. Tests: the two identities above, with and without overlap, Hermitian and not, with
   degenerate auxiliary energies; and an assertion that the eigendecomposition did not
   run, so a later refactor cannot silently reintroduce it.

**Expect the baseline to move**, by ~1e-13 Ha in the Green's function poles and below
5e-12 eV in the quasiparticle energies. That movement is *removed* roundoff, not
introduced: the reconstructed-moment diagnostic improves by an order of magnitude
(1.25e-13 → 1.68e-14) because the auxiliaries are exactly what the recurrence produced.
Confirm the frontier holds, then re-record deliberately.

**Payoff.** A further 1.6x to 1.8x on the Dyson stage on top of 1.1. **Effort**: 2–3 days.

**Kill criterion.** If the lazy form cannot reproduce `get_self_energy()` under a
non-identity overlap to the tolerance above, fall back to special-casing
`combine_for_self_energy` alone, which is worth roughly half of it.

### 1.3 Stop diagonalising in order to report a count and a range

**Status: implemented.** `n_poles` reads `solver.result.neig`, now a shape rather than
`eigvals.size` — and compatible with both the pinned and the new dyson, so momentGW can
carry it before the pin moves. `__post_kernel__` reads a new `Spectral.spectrum`, which
takes an eigenvalues-only route (`Lehmann.eigenvalues_of_matrix`, factored out of
`diagonalise_matrix` so the overlap handling is shared rather than duplicated). Measured
below at 484 ms for two calls, against ~2 s for the naive three.

**Required for 1.2 to pay in full.** Under a lazy `Spectral`, three diagnostics still
touch `eigvals` and would drag the eigendecomposition back in:

| site | what it wants | what it costs |
| --- | --- | --- |
| `momentGW/gw.py:74`, `realization_record` | `n_poles` = `eigvals.size` | a full solve, for a **shape** |
| `dyson/.../_mbl.py:290-291`, `__post_kernel__` | min and max root, one log line | a full solve, twice |

Measured with the naive replacement (values-only `eigvalsh`, three calls): 0.11 s at
cc-pVDZ and **2.06 s at cc-pVTZ** — 3.8% of that run, for a count and a log line.

`n_poles` becomes `nphys + naux`, needing no arithmetic. The `__post_kernel__` range
becomes a single cached `eigvalsh`, or reports the auxiliary energy range, which is what
the line is about and is free. The Tier 1 numbers in §0.4 were measured *with* the naive
triple-`eigvalsh` still in place, so the finished change is better than they show.

### 1.4 The deferred pin bump rides along

`mkakcl/dyson#5` (`cc2f48f`) — memoising `solve` per iteration, worth under 1% — has
been merged since 2026-08-04 and is unreachable while momentGW pins `73cd18d`.
`DIAGONALISATION.md` §7.2 said "do not bump the pin for this alone", and that was right.
Tier 1 is the reason the pin moves; #5 arrives free with it.

Update the `pyproject.toml` comment, `ROADMAP.md` §1.4, and
[`baseline/README.md`](baseline/README.md) for the re-record.

### 1.5 Verification

`ruff check`, `ruff format --check`, `pre-commit run --all-files`, the full dyson suite,
`python -m pytest tests/ -q`, then `python -m baseline.check` — expecting movement at
~1e-13 Ha, understood and then re-recorded.

`Spectral` and `Lehmann.moments` are shared with the unrestricted, periodic and
self-consistent paths, which the 52 baseline cases do not cover (`baseline/run.py:562`
records `fock_loop` as provenance but never varies it). Their tests must pass; consider
recording one such case so the change is judged against a number rather than a threshold.

---

## 2. Tier 2 — exact, harder, and re-costed

Post-Tier-1 targets at benzene / cc-pVTZ, where the Dyson stage is 7.3 s of a ~31 s run.

### 2.1 The recurrence — the new largest Dyson item

`_recurrence_iteration_hermitian` (`dyson/solvers/static/mblse.py:282`) is ~3.9 s, about
**12% of the run** after Tier 1 and the biggest thing left in the stage. The block
recursion is `O(max_cycle^2 * nmom)` products of `nphys x nphys` blocks, issued one at a
time: the profile shows 54 `_dgemm` calls and 158 `ndarray.dot` calls, none of them
large. This is the same shape as the Milestone 4 external-orbital loop, which went from
701 ms to 245 ms by batching and reordering for contiguity.

**Unmeasured**, and the obvious next thing to measure. Batching the recursion's products
into larger GEMMs is exact up to summation order. Start by profiling which of the ten
terms in the diagonal recursion dominate before restructuring anything.

### 2.2 The one remaining arrowhead solve — `DIAGONALISATION.md` Options 1, 2, 3

After Tier 1 there is exactly one genuine arrowhead eigendecomposition, at
`momentGW/fock.py:557`: 1.8 s, **~6% of the run** at cc-pVTZ and ~3% at cc-pVDZ. The
block-tridiagonal solves that Option 1 targets are ~0.75 s, ~2%.

| option | target after Tier 1 | ceiling on the whole run |
| --- | --- | --- |
| Option 1, Golub–Welsch on the block-tridiagonal | ~2% | **1.02x** |
| Options 2 and 3, on the arrowhead | ~6% (cc-pVTZ) | **1.06x** |

Option 1's own kill criterion in §7.3 (2x over `dsyevd` before tuning) already made it a
prototype-and-discard, and §4 measured a quarter of the arithmetic losing 5.5x to BLAS-3
in exactly this shape. Option 3 additionally needs the §7.1 prerequisite — `N(mu)` and
the density matrix decoupled from the eigenpair list, since `search_chempot` consumes 45%
of the spectrum — which is 1–2 weeks of work for at most 1.06x on this path.

**Verdict: close all three**, and record these ceilings in `DIAGONALISATION.md` so they
are not reopened. One piece is worth keeping if 2.1 or a future need revives this: only
`c[:nmo]` is ever read from that solve, and given the eigenvalues each physical block is
the null vector of an `nphys x nphys` Schur complement, so `eigvalsh` plus an
`O(n * nphys^2)` recovery would replace an `O(n^3)` solve exactly.

### 2.3 Poles the realization cannot place

**Not a performance item.** It came out of verifying §1.1 and is recorded here because
this is where the evidence is. It belongs to Milestone 3, and it is worth more than
anything else in this document.

**What was found.** The MBLSE realization emits poles whose couplings are at the level of
roundoff — norms ~1e-10, weights ~1e-20 — sitting at the eigenvalues of a numerically null
block. Their **energies are not determined by the input moments**. Perturbing the input
moments of the h2o/sto-3g CCSD particle self-energy by 1e-16 relative:

| input | lowest pole | lowest pole with weight > 1e-12 | total weight | max abs moment 7 |
| --- | --- | --- | --- | --- |
| unperturbed | −31.959026 | 1.313758 | 0.032516536 | 3.997330e+05 |
| jitter 0 | −23.769228 | 1.313758 | 0.032516536 | 3.997330e+05 |
| jitter 1 | −0.000002 | 1.313758 | 0.032516536 | 3.997330e+05 |
| jitter 2 | −40.464478 | 1.313758 | 0.032516536 | 3.997330e+05 |
| jitter 3 | **−216.282234** | 1.313758 | 0.032516536 | 3.997330e+05 |
| jitter 4 | −15.452897 | 1.313758 | 0.032516536 | 3.997330e+05 |
| jitter 5 | −17.692589 | 1.313758 | 0.032516536 | 3.997330e+05 |

The null pole ranges over three orders of magnitude under last-bit noise, while every
determined quantity is unchanged to every digit printed. Nothing reports this.

**How common.** **50 of the 60 cases** in `tests/test_mblse.py` carry at least one such
pole. This is the normal state of the realization, not an exotic failure.

**Why it matters.** Moment `n` weights a pole by `e**n`, so an undetermined energy becomes
visible at exactly the high moment orders this project exists to make trustworthy. It also
breaches a guiding rule: *"Never silently discard a moment direction without reporting its
scale, rank loss, and effect on reconstructed moments."* These directions are not
discarded — they are **kept and given an arbitrary energy**, which is worse, and unreported.
Milestone 3.3's step-down gate and 3.4's deep-state validation are the right home.

It is also a plausible mechanism for something `ROADMAP.md` Milestone 4 already records:
deep quasiparticle states moving by 5.2e-8 eV at `nmom_max = 7` under a pure reassociation
of BLAS calls, while the HOMO holds at ~1e-13 eV. **Not established** — that was measured
in momentGW's G0W0 path and this in dyson's MBLSE fixtures. Connecting the two is a
concrete piece of Milestone 3.1's error budget and worth doing before anything else here.

**It is not simply "delete the small-weight poles."** Three reasons:

1. **Small weight is not the same as spurious.** A genuinely weak satellite also has small
   weight. Separating "at the roundoff floor, therefore meaningless" from "small but real"
   needs the error budget of Milestone 3.1; a bare threshold would discard physics.
2. **Deleting a pole changes the moments.** Exact moment conservation is the premise of the
   construction. Removing a pole perturbs every moment it contributed to — ~0 at low order,
   but weighted by `e**n` at high order, which is the same amplification that caused the
   problem. Any removal must be verified against delivered moments, not assumed safe.
3. **The cause is upstream of the pole list.** These poles exist because the moment data
   supports fewer independent directions than the recurrence emits. The principled response
   is to detect that rank deficiency where it arises and either step the order down — the
   Milestone 1.3 machinery already does this for other failures — or report it, rather than
   emit a pole and let an eigendecomposition of a null block decide where it lands.

**Cheapest useful first step**, short of any of that: *report* them. Count the poles below
a scale-aware weight threshold, their energy spread, and their contribution to each
delivered moment, and put it on the solver diagnostics. That is a diagnostic, not a change
of numerics, so it needs no baseline re-record, and it would say how much of the deep-state
irreproducibility this accounts for.

**Test debt taken in the meantime.** `test_vs_exact_solver_central[h2o-sto3g-CCSD-3]` was
passing at 1e-8 by luck: it compares moments of a representation containing these poles, so
any reassociation of the arithmetic can push it over. That one parametrisation now compares
at 1e-6, with the analysis in a comment beside it and a pointer here. The other 59 keep
1e-8 — widening every case that merely *contains* null poles would have gutted the test.
**Remove the widening when the cause is fixed.**

---

## 3. Tier 3 — speedups that trade accuracy

Nothing here should start before Milestone 3, and after Tier 1 there is little left to
buy. All of it depends on knowing which poles are converged, which is what Milestone 3.2
and 3.4 are for. Milestone 4 already measured the problem: a pure reassociation of BLAS
calls moves the deep quasiparticle states by 5.2e-8 eV at `nmom_max = 7` while the HOMO
holds at ~1e-13 eV.

- **Windowed frontier-only solve** (`DIAGONALISATION.md` §7.5 Part B). Bounded by 2.2's
  1.06x, so not worth doing on performance grounds. Worth doing only if a frontier band
  is genuinely the whole answer for some calculation — and then as a mode recorded in the
  diagnostics, never a default, and not serving Milestone 3.4's spectral functions.
- **Deep-pole truncation in the realization.** Judge by delivered moment error and
  frontier QP movement, per the guiding rules, not a proxy weight threshold.
- **Mixed precision.** FP64 stays the reference; needs refinement and residual checks.

---

## 4. Ordering

1. ~~**Tier 1.1, 1.2 and 1.3.**~~ **Done** on branch `m4-lehmann-moments-gemm`; 1.1 is
   committed as `a1569d9`, 1.2 and 1.3 are uncommitted. Measured **2.97x on the Dyson
   stage and 1.45x on the run** (§0.5).
2. **Check `baseline.check` is reproducible** (§5) — it is the gate everything below rests
   on, and the dyson suite demonstrably is not. Then **§1.4**: pin bump, `baseline.check`,
   deliberate re-record, `ROADMAP.md` §1.4 and `baseline/README.md`.
3. **§2.3's cheapest first step — report the unplaceable poles.** Ahead of the rest of
   Tier 2 because it is a diagnostic, needs no re-record, and answers whether §2.3 is the
   mechanism behind Milestone 4's 5.2e-8 eV deep-state movement.
4. **Re-profile.** Milestone 4's acceptance gate requires identifying the new dominant
   stage; after Tier 1 the Dyson stage is 17–27% of a run and the correlated moment
   construction is the majority again.
5. **Measure Tier 2.1**, the recurrence — the largest remaining Dyson item and the only
   one whose size is not yet known.
6. **Close Options 1, 2 and 3** in `DIAGONALISATION.md` with the ceilings in §2.2.
7. **Tier 3 last**, after Milestone 3, gated and opt-in.

## 5. What has not been checked

- §1.1, §1.2 and §1.3 are all implemented and the dyson suite is green (1003 passed, 40
  skipped). **`python -m baseline.check` has not been run against any of it**, because the
  pin has not moved.
- The 1e-6 tolerance in §2.3's test debt is a judgement, not a measured bound on the pole
  lottery. The observed excursion was 2.8e-8; how far the lottery can actually reach was
  not established.
- ~~**Open: is `baseline.check` reproducible run to run?**~~ **Answered — yes.** Four runs
  on an unchanged tree: 52/52 unchanged every time, and almost every case bit-identical
  (`HOMO shift +0.000e+00 eV`, `eta0 frobenius rel 0.0e+00`). The only movement is on the
  four `hydrogen_pbe` cases, at **1e-14 eV** in the HOMO, plus realization residuals moving
  26–66% *relatively* — which is 6e-16 against 9e-16, exactly what `NOISE_FLOOR` exists to
  absorb.

  Two consequences. First, the gate is sound, and **sharper than its own docstring
  claims**: `DETERMINISTIC = (1e-8, 1e-12)` is documented as "the measured run-to-run
  scatter", propagated from the Clenshaw-Curtis grid-scale plateau — but HHT became the
  default on 2026-08-04, which removes that source, and the measured scatter is now ~1e-14
  eV. The comment is stale and the floor is ~6 orders more generous than the path needs.
  Worth revisiting, since a tighter gate would catch more.

  Second, and directly relevant to §1.4: Tier 1's movement (5e-13 to 5e-12 eV in the QP
  energies) sits **30–270x above** that scatter. So the re-record is *evidence*, not just
  provenance — the shifts it reports will be attributable to the change rather than lost in
  noise. That is the opposite of the situation `DIAGONALISATION.md` §7.2 described for the
  `#5` memoisation, and it means the re-record is worth reading rather than rubber-stamping.

- **The dyson suite, unlike the baseline, is not reproducible.** Parked for later. The same
  test, same code, gave 8.88e-16, 6.70e-16, 1.33e-15 and 1.33e-15 on four identical runs,
  and it persists under `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` and at fixed
  `PYTHONHASHSEED` (three runs at seed 0: 1.44e-15, 1.33e-15, 8.88e-16). That rules out
  BLAS threading and dict/set ordering, and points at allocator- and alignment-dependent
  kernel dispatch in Accelerate under ASLR — outside the code. The visible symptom is a
  rare multi-test failure in `test_mblgf.py` / `test_mblse.py`, seen in 2 runs of about 28
  with §1.2 present and 0 of 7 without; that sample is far too small to attribute to §1.2,
  and no attribution should be read into it.

  The split is now measured rather than argued: it is confined to dyson's CCSD/FCI
  fixtures — non-Hermitian, through `np.linalg.eig` — while the restricted molecular G0W0
  path, Hermitian throughout, reproduces bit-identically (bullet above). So it does not
  threaten the Tier 1 gate. What it does threaten is dyson's own suite, where any test
  comparing an ulp-decided quantity is a coin flip; that is §2.3's problem in a different
  costume, and the same fix retires both.
- Tier 2.1's size is an estimate from one profile; the batching payoff is unmeasured.
- Two molecules, one starting point (PBE), one moment order (`nmom_max = 7`) for the
  A/B/C table. Lower moment orders shrink the Dyson share — the arrowhead is `5 * nmo`
  at `nmom_max = 3` against `9 * nmo` at 7 — so Tier 1 is worth less there (1.079x at
  cc-pVDZ for 1.1 alone).
- The unrestricted, periodic, THC and self-consistent paths were not exercised, and
  neither was MPI.
- The round-trip identity was verified for the Hermitian restricted MBLSE path; the
  non-Hermitian branch of `get_auxiliaries` follows the same algebra but was not measured.
- Wall times on this machine drift between processes and appear to throttle under load;
  only within-process A/B/C ratios and shares should be trusted.
