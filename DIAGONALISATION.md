# Diagonalisation in the Dyson solve: what happens now, and what could replace it

Written 2026-08-04. Nothing here has been implemented. Every number is measured on this
machine (macOS, Accelerate, PySCF 2.14.0, `mgw-monomial`) at the pin `mkakcl/dyson@73cd18d`,
with the benzene test systems used for the Milestone 4 profiling.

## Summary

The premise is right: the block Lanczos form **is** discarded. `MBLSE.solve` assembles the
block-tridiagonal Hamiltonian into a dense array and hands it to a general dense
eigensolver, then throws away three quarters of the eigenvectors it paid for.

Three things qualify that, and they change what is worth doing:

1. **The block Lanczos solve is not where the time goes.** There is a second, larger dense
   eigendecomposition downstream — the upfolded Green's function supermatrix, which is
   *arrowhead*, not block-tridiagonal. It costs about 5x the two block Lanczos solves
   combined. If we are optimising a diagonalisation, it is that one.
2. **The obvious structural fix is measurably slower.** Feeding the block-tridiagonal
   matrix to LAPACK's banded eigensolver — the textbook move — is 5.5x *slower* than the
   dense solver at `nmo = 384`. Reason in §4.
3. **Amdahl caps the prize.** All eigendecomposition together is 7.7% of a G0W0 run at
   `nmom_max = 3` and 15.6% at `nmom_max = 7`. Making it instantaneous buys 1.08x and
   1.19x respectively.

So the most valuable option is not a faster eigensolver. It is **computing fewer
eigenpairs** — which we can now justify on numerical grounds, not just cost, because we
have already measured that the deep poles this work produces are not trustworthy at high
moment order (§5, Option 3).

**Update, same day — Option 3 has now been measured rather than argued (§5.1).** The
windowed kernel is real and very fast: shift-invert on the arrowhead gives **78x** for a
narrow frontier window at `nmo = 384`. But the end-to-end ceiling is **1.10x** at cc-pVDZ
and **1.19x** at cc-pVTZ, and it saturates: a 10x kernel already captures almost all of it,
so a 78x kernel is worth about 1% more than a 10x one. The prize for this entire document
is under 20%, and Option 3 collects most of what there is.

**One item has been acted on.** The repeated diagnostic diagonalisation (§5.1, last
subsection) turned out to be a genuine duplicate of the top iteration only, and is fixed in
`mkakcl/dyson#5`. It is worth under 1%, not the 3–5% I first estimated. momentGW still pins
`73cd18d`, so the fix is not reachable from here until the pin next moves for some other
reason — a deliberate decision, taken because re-recording 52 baseline cases is
disproportionate to a sub-1% gain that changes no numbers.

Two estimates in the first draft of this document were wrong and are corrected in place:
that one, and the claim that the arrowhead shift-invert costs `O(n)` per solve when it is
`O(n * nphys)`. Everything labelled as measured has been measured; treat the effort
estimates in §5 as judgement, and the one for Option 2 as little better than a guess.

## 1. What the code does today

Per restricted molecular G0W0, three dense eigendecompositions:

| # | Where | Structure | Dimension |
| --- | --- | --- | --- |
| 1, 2 | `MBLSE.solve`, once for the hole sector and once for the particle sector | block tridiagonal | `(max_cycle + 1) * nphys` |
| 3 | `Lehmann.diagonalise_matrix`, on the combined self-energy | arrowhead | `(2 * max_cycle + 3) * nphys` |

with `nphys = nmo` and `max_cycle = (nmom_max - 1) // 2`.

**The block Lanczos step.** `dyson/solvers/static/mblse.py:450` calls
`util.build_block_tridiagonal`, which at `dyson/util/moments.py:193` materialises the whole
thing with `np.block` — every off-tridiagonal zero included. Line 453 slices off the
physical block and line 454 calls `util.eig_lr`, which for the Hermitian case is
`np.linalg.eigh`: a general dense solver that knows nothing about the tridiagonal block
structure it has been handed.

Then line 456:

```python
couplings = np.atleast_2d(self.off_diagonal[0]) @ rotated[0][: self.nphys]
```

**Only the first `nphys` rows of the eigenvector matrix are ever read.** At `nmom_max = 7`
the subspace is `4 * nphys`, so 75% of the eigenvector matrix is computed and discarded.
The eigenvalues are all used; the eigenvectors are needed only through their first block
row. That is precisely the structure of a Gauss quadrature rule — nodes plus weights — and
it is the part of the block Lanczos form the code is not exploiting.

Two existing markers in the same function are worth noting while anyone is in here:
`# FIXME: being called twice?` at `mblse.py:426` and `# TODO inherit` inside it.

**The arrowhead step.** `Lehmann.diagonalise_matrix` at
`dyson/representations/lehmann.py:638` builds the upfolded supermatrix

```
[ f    v ]
[ v^T  diag(e) ]
```

— a dense `nphys` head, a border, and a **diagonal** tail of auxiliary poles — and line 640
sends it to a dense `eigh`. This is the largest of the three by a wide margin, and its
structure is thrown away just as completely.

## 2. How big these get

For `nphys = nmo` and the two moment orders the baseline uses:

| `nmom_max` | `max_cycle` | block-tridiagonal (x2) | arrowhead |
| --- | --- | --- | --- |
| 3 | 1 | `2 * nmo` | `5 * nmo` |
| 7 | 3 | `4 * nmo` | `9 * nmo` |

The arrowhead grows faster in both directions, which is why it dominates.

## 3. What it costs

Measured directly on synthetic matrices with the exact structure and sizes above, at
`nmom_max = 7`. Times are best of two.

| `nmo` | block-tri `n` | dense `eigh` | `eig_banded` | `eig_banded`, values only | arrowhead `n` | dense `eigh` |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 256 | 13 ms | 17 ms | 11 ms | 576 | 64 ms |
| 128 | 512 | 51 ms | 123 ms | 70 ms | 1152 | 444 ms |
| 256 | 1024 | 309 ms | 1915 ms | 903 ms | 2304 | 3794 ms |
| 384 | 1536 | 1213 ms | 6659 ms | 3340 ms | 3456 | 12152 ms |

The arrowhead solve is **~5x the two block-tridiagonal solves combined** at every size
tested, and pulling further ahead as `nmo` grows.

In a real calculation (benzene, PBE, cc-pVDZ, current `master` with the Milestone 4 work
merged), all `eigh` calls together are:

| | total | `eigh` | share |
| --- | --- | --- | --- |
| `nmom_max = 3` | 8.46 s | 0.651 s | **7.7%** |
| `nmom_max = 7` | 12.66 s | 1.978 s | **15.6%** |

That share is the ceiling on everything below. It grows with `nmom_max` and shrinks with
system size, because the eigendecompositions are `O(N^3)` while the moment construction
they sit behind is `O(N^4)`.

## 4. Why the textbook fix loses

A block-tridiagonal matrix is banded, so the obvious move is LAPACK's banded eigensolver.
Measured above, it is **5.5x slower** than the dense one at `nmo = 384`, and even
eigenvalues-only banded is 2.75x slower than a full dense solve.

Two reasons, and both are structural rather than incidental:

- **The matrix is barely banded.** A block-tridiagonal matrix with blocks of size `nphys`
  has half-bandwidth `2 * nphys - 1`. With only four blocks, that is half the matrix
  dimension. Banded algorithms win when `b << n`; here `b/n ~ 1/2`.
- **Dense LAPACK is BLAS-3 and banded LAPACK is not.** `dsyevd` uses a blocked
  tridiagonal reduction and divide-and-conquer; the banded path (`dsbtrd`) is a
  bandwidth-narrowing sweep that is largely BLAS-2 and memory-bound.

This is worth remembering beyond this document: **a flop count that exploits structure is
not a prediction of wall time.** The same lesson turned up in Milestone 4, where Cholesky
solves lost to an explicit inverse.

## 5. Options

### Option 0 — leave it

Free, and defensible. 7.7–15.6% of runtime with a hard ceiling, in code that is correct
and understood. Everything below competes against this.

### Option 1 — stop computing the discarded eigenvectors

*The block Lanczos step, done properly.*

We need every eigenvalue but only the first `nphys` rows of each eigenvector. For a scalar
tridiagonal matrix this is exactly the Gauss quadrature nodes-and-weights problem solved by
**Golub–Welsch**: run the symmetric tridiagonal QL and accumulate only the first row of the
rotation, giving `O(n^2)` instead of the `O(n^3)` of a full eigenvector accumulation. The
block generalisation is standard (Golub & Meurant, *Matrices, Moments and Quadrature*), and
it is the algorithm the moment-conserving construction is implicitly built on.

- **Expected gain**: a constant factor on items 1 and 2 only, which are a fifth of the
  eigendecomposition cost, which is at most 15.6% of the run. Realistically **under 2%
  overall**.
- **Effort**: moderate. Needs care with degenerate and clustered eigenvalues.
- **Risk**: §4 is the caution. The flops saved are real, but a hand-written reduction
  competing with `dsyevd`'s blocked BLAS-3 may not be faster in wall time even when it does
  a quarter of the arithmetic.
- **Verdict**: correct in principle, poor return. Worth doing only as part of Option 3,
  where the same machinery is needed anyway.

### Option 2 — exploit the arrowhead structure

*The one that is actually big.*

The supermatrix is a dense `nphys` head plus a diagonal tail, so it is not the classical
arrowhead (which has a scalar head) but a **diagonal-plus-rank-`nphys`** problem after the
head is diagonalised. Two established routes:

- **Secular-equation arrowhead solvers.** Jakovčević Stor, Slapničar & Barlow give an
  `O(n^2)` algorithm for real symmetric arrowhead matrices with *high relative accuracy* for
  every eigenvalue and every eigenvector component, based on shift-and-invert plus bisection
  on the secular equation. Implemented in **[Arrowhead.jl](https://github.com/ivanslapnicar/Arrowhead.jl)**,
  which also covers diagonal-plus-rank-one.
- **DPRk as a sequence of rank-1 updates.** The standard divide-and-conquer device: a
  rank-`k` update is `k` successive rank-1 updates, each an `O(n^2)` secular-equation solve.
  Here `k = nphys` and `n = 9 * nphys`, giving `~81 nphys^3` against `729 nphys^3` dense — a
  **~9x** theoretical saving on the dominant eigendecomposition.

- **Expected gain**: if 9x were realised on item 3, the eigendecomposition share would fall
  from 15.6% to ~4%, i.e. **~1.14x** on the whole calculation at `nmom_max = 7`. Less at
  lower moment order.
- **Effort**: high. This is the part of the numerical-linear-algebra literature where
  deflation, clustered roots and the accuracy of the secular solve are the entire problem.
  The Jakovčević Stor et al. accuracy result depends on evaluating one element of a shifted
  inverse in extended precision — the same discipline as the Milestone 2 coefficient work,
  which is a point in its favour: this project already has that habit and the
  `numpy.longdouble` machinery for it.
- **Risk**: real, and of a kind this roadmap cares about. A secular-equation solver that
  silently misconverges on a cluster produces plausible poles. It would need the same
  treatment eta0 got — a certificate, not a return code.
- **Verdict**: the only option with a double-digit-percent payoff, and the only one whose
  accuracy story is genuinely delicate.

### Option 3 — compute only the poles that are worth computing

*Recommended.*

Roadmap Milestone 4 already contains "use partial/shift-invert diagonalization when only
frontier poles of a large compact Hamiltonian are requested". Two facts we did not have when
that was written make it the strongest option:

1. **Shift-invert on an arrowhead matrix is cheap.** Solving `(A - sigma I) x = b` needs no
   `n x n` factorisation: eliminate the diagonal tail, and each solve costs one
   `nphys x nphys` back-substitution plus two `nphys x naux` products. The Schur complement
   is formed once per shift. Measured below.
2. **The deep poles are not trustworthy anyway.** Measured for Milestone 4: a pure
   reassociation of BLAS calls — identical arithmetic, different summation order — moves the
   deep quasiparticle states by 5.2e-8 eV at `nmom_max = 7` while the HOMO holds at ~1e-13
   eV. We are currently spending the large majority of the diagonalisation budget computing
   states whose values are not reproducible under a change that is provably a no-op.

### 5.1 Measured, not argued

**Correction to point 1 above.** The per-solve cost is `O(n * nphys)`, not `O(n)` — the head
is a dense `nphys` block, not the scalar of a classical arrowhead. It is still much cheaper
than a dense solve: at `nphys = 384`, `n = 3456`, the arrowhead solve takes **1396 us**
against **18606 us** for a dense LU back-substitution with the factorisation already paid
for, a 13x gap that widens with `n`.

**The window width decides everything.** Shift-invert Lanczos against dense `eigh`, on the
real arrowhead structure, with the frontier eigenvalues checked against the dense spectrum
every time:

| window `k` | % of spectrum | `nphys = 114`, `n = 1026` | `nphys = 384`, `n = 3456` |
| --- | --- | --- | --- |
| 8 | 0.2–0.8% | **24.6x** | **77.7x** |
| 16 | 0.5–1.6% | 14.1x | 62.0x |
| 32 | 0.9–3.1% | 9.4x | 37.3x |
| 64 | 1.9–6.2% | 2.7x | 17.7x |
| 128 | 3.7–12.5% | 1.1x | 7.3x |
| 256 | 7.4–25% | 0.9x | 3.1x |
| 768 | 22% | — | 0.4x |

The crossover is at roughly **10% of the spectrum**. Ask for a narrow frontier band and the
win is enormous; ask for a QP energy per orbital (`k ~ nmo`, which is 11% of `9 * nmo`) and
there is **no win at all**. That single fact is the whole design question, and it is a
specification question, not a linear algebra one.

A note on the obvious middle road: LAPACK's subset driver (`evr` with `subset_by_index`)
gives a **flat ~1.5x regardless of window width**, because it still pays the full
tridiagonal reduction and only saves on back-transforming eigenvectors. It never beats
shift-invert for narrow windows and never beats dense `eigh` by much for wide ones.

**The end-to-end ceiling.** Timing the real `Lehmann.diagonalise_matrix` calls inside a
complete G0W0 run at `nmom_max = 7`:

| system | total run | in the arrowhead solve | ceiling if that solve were free |
| --- | --- | --- | --- |
| benzene / cc-pVDZ, `nmo = 114` | 12.21 s | 1.16 s (9.5%) | **1.10x** |
| benzene / cc-pVTZ, `nmo = 264` | 85.76 s | 13.87 s (16.2%) | **1.19x** |

And it saturates immediately: a **10x** kernel gives 1.09x / 1.17x, a **78x** kernel gives
1.10x / 1.19x. Past roughly 10x, further speedup of this kernel is worth nothing.

**An unexpected find while instrumenting — and a corrected estimate.**
`diagonalise_matrix` is called **12 times** per G0W0, at dimensions
`[228, 342, 456, 570, 1026]` for cc-pVDZ — not once. The small ones are the per-iteration
diagnostics: with `calculate_errors=True` (momentGW's default), `moment_errors`
reconstructs the representation at every iteration, and each reconstruction diagonalises.

I first estimated those at **a third of the arrowhead time, 3–5% of the run**. That was
wrong, and the fix has since been made and measured: **it is under 1%.** The estimate
assumed the per-iteration solves were redundant. They are not — each is a distinct
iteration, reported once. The only genuine duplicate was the *top* iteration, solved twice
per solver: once by the diagnostics at the end of the recurrence, and again by `kernel` to
assign `result`. That is two solves per solver, four per G0W0, worth ~100 ms of a 12.5 s
run.

Fixed in `mkakcl/dyson#5` by memoising `solve` per iteration in `BaseMBL` — which is what
the `# FIXME: being called twice?` at `mblse.py:426` was pointing at. Verified bit-identical:
52/52 baseline cases unchanged, which also demonstrates that nothing downstream mutates the
now-shared object. **Nothing further can be removed here** without changing what the
diagnostics compute.

- **Expected gain**: up to 1.10x (cc-pVDZ) or 1.19x (cc-pVTZ), and only for narrow windows.
- **Effort**: the kernel is a day's work — the shift-invert operator used for the table
  above is about 15 lines. The *specification* is the job: what the window is, what the
  particle-number fit needs, and what the calculation may still claim to report.
- **Risk**: a windowed solve cannot produce a spectral function or deep satellites, so it
  has to be a mode, not a default, with diagnostics saying which was used. Milestone 3.4
  wants spectral functions, which this cannot serve.
- **Verdict**: still the best option in this document, and now known to be worth at most
  ~1.2x. Worth doing for the systems where a frontier band is genuinely all that is wanted;
  not worth doing on performance grounds alone.

### Option 4 — write it in a faster language

Worth stating plainly so it is not assumed: **on its own this buys nothing.** All three
eigendecompositions are already single LAPACK calls in Fortran. Python is doing no work
worth removing; the profile shows the time inside `dsyevd`, not around it.

Language only becomes relevant *as a consequence* of Options 2 or 3, where the algorithm is
a loop over `n` secular-equation solves or Lanczos iterations, and Python-level overhead per
iteration would dominate. In that case:

- **Julia** — Arrowhead.jl already exists and is the reference implementation of the
  accuracy result. Best if the goal is to evaluate the algorithm quickly. Poor fit as a
  runtime dependency of a PySCF-based Python package.
- **C or Rust extension** — the natural home for a production version, and the shape dyson
  would want if this were upstreamed.
- **numba or Cython** — cheapest path to a working prototype inside the existing package,
  and enough to find out whether the theoretical 9x survives contact with a real machine
  (§4 says: assume it might not).

Recommendation if this is pursued: prototype in whatever is fastest to write, **measure
against `dsyevd` before committing to a language**, and treat the prototype as disposable.

## 6. What I would do next

The ceiling measurement in §5.1 is done, and it reorders the list:

1. ~~**Fix the repeated diagnostic diagonalisation first.**~~ **Done** — `mkakcl/dyson#5`,
   merged as `cc2f48f`. Worth under 1%, not the 3–5% estimated; see §5.1. Not yet reachable
   from momentGW, which still pins `73cd18d`.
2. **Then decide whether a windowed mode is wanted at all** — not on performance grounds,
   which cap out at ~1.2x, but on whether there are calculations where a frontier band is
   genuinely the whole answer. If yes, the kernel is a day and the specification is a week.
3. **Treat Option 2 as research**, gated behind the same standard as eta0: a certificate on
   the computed spectrum, validated against the dense solve on the baseline systems, before
   it goes anywhere near a default. Its ~9x on a kernel that saturates at 10x means it is now
   competing for the *same* 1.2x that Option 3 already collects more cheaply.
4. **Leave Option 1 alone.** Under 2%, and Option 3 does not need it.

None of this is urgent, and the whole area is now bounded: **at most 1.10x on cc-pVDZ and
1.19x on cc-pVTZ, even with a free eigensolver.** The `O(N^3)`-versus-`O(N^4)` trend works
against it as systems grow. The Milestone 3 conditioning work is worth more — including to
Option 3, which would be far easier to specify if we knew which poles were converged.

## References

- I. Jakovčević Stor, I. Slapničar, J. L. Barlow, *Accurate eigenvalue decomposition of real
  symmetric arrowhead matrices and applications*, Linear Algebra Appl. 464 (2015) 62–89.
  [arXiv:1302.7203](https://arxiv.org/abs/1302.7203)
- [Arrowhead.jl](https://github.com/ivanslapnicar/Arrowhead.jl) — Julia implementation of the
  above, plus diagonal-plus-rank-one.
- G. H. Golub, G. Meurant, *Matrices, Moments and Quadrature with Applications* — block Gauss
  quadrature, and the nodes-and-weights structure behind Option 1.
- W. N. Gansterer et al., *Eigendecomposition of block tridiagonal matrices*
  ([arXiv:1306.0217](https://arxiv.org/abs/1306.0217)) and the twisted block factorization
  work, for Option 1's block generalisation.
- C. J. C. Scott, O. J. Backhouse, G. H. Booth, *A "moment-conserving" reformulation of GW
  theory*, J. Chem. Phys. 158, 124102 (2023).
  [arXiv:2301.09107](https://arxiv.org/abs/2301.09107) — the upfolded Hamiltonian whose
  diagonalisation this document is about.
