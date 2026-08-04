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

**Second update — §7 adds implementation plans, and one more measurement moves Option 3.**
The chemical-potential search and the density matrix both need every pole below the
chemical potential, which is **45% of the spectrum** on benzene/cc-pVTZ — not a frontier
window. Since shift-invert loses to dense above ~10% of the spectrum, a windowed solve
returns nothing on the default path until `N(mu)` and the density matrix are obtained some
other way. See §7.1; it is the single most important paragraph in this document.

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
   genuinely the whole answer. ~~If yes, the kernel is a day and the specification is a
   week.~~ **Superseded by §7.1**: a window cannot serve the default path at all until the
   electron count and the density matrix stop being computed from the eigenpair list. Read
   §7.5 for what that costs.
3. **Treat Option 2 as research**, gated behind the same standard as eta0: a certificate on
   the computed spectrum, validated against the dense solve on the baseline systems, before
   it goes anywhere near a default. Its ~9x on a kernel that saturates at 10x means it is now
   competing for the *same* 1.2x that Option 3 already collects more cheaply.
4. **Leave Option 1 alone.** Under 2%, and Option 3 does not need it.

None of this is urgent, and the whole area is now bounded: **at most 1.10x on cc-pVDZ and
1.19x on cc-pVTZ, even with a free eigensolver.** The `O(N^3)`-versus-`O(N^4)` trend works
against it as systems grow. The Milestone 3 conditioning work is worth more — including to
Option 3, which would be far easier to specify if we knew which poles were converged.

Per-change plans, with kill criteria, are in §7. The ordering there supersedes this list.

## 7. Implementation plans

Added 2026-08-04, after a further measurement that changes Option 3 materially (§7.1).
Each plan states where the change goes, how it is verified, and — the part usually left
out — **what result would make us stop**.

Common to all of them: dyson changes land on `mkakcl/dyson` and reach momentGW only through
the pin, so every one of these carries a pin bump and a baseline re-record at the end. That
ceremony is ~30 minutes and is the reason none of these is worth doing for its own sake in
isolation.

### 7.1 The finding that reorders everything

**A frontier window cannot serve the default code path.** Two consumers need the whole
occupied half of the upfolded spectrum, not a window:

- `search_chempot` (`momentGW/fock.py:21`) walks eigenpairs upward from the bottom of the
  spectrum accumulating physical weight `2 |v_phys,i|^2` until it reaches `nelec`. It needs
  every pole below the chemical potential.
- `make_rdm1` (`momentGW/gw.py`) is `gf.occupied().moment(0) * 2`, a sum over every occupied
  pole's physical-block eigenvector.

Measured on benzene/cc-pVTZ at `nmom_max = 7`: **1077 of 2376 poles lie below the chemical
potential, or 45% of the spectrum**, and `search_chempot` consumes all 1077 of them.

Put that next to the crossover in §5.1 — shift-invert loses to dense above roughly 10% of
the spectrum — and Option 3 as originally described **returns nothing on the default path**.
The 78x is real, and unreachable, unless the electron count and the density matrix come from
somewhere other than enumerated eigenpairs.

That is possible, and it is the interesting part of Option 3 rather than an obstacle to it:
`N(mu) = Tr[P_phys theta(mu - A)]` is a spectral projector, evaluable by contour integration
of the resolvent, and the resolvent of an arrowhead matrix is exactly the `O(n * nphys)`
solve already prototyped. The same contour gives the density matrix. This is Milestone 5
technology pointed at a different target, and it is a much larger job than "call `eigsh`
with a window".

### 7.2 Diagnostic re-diagonalisation — done, pin deferred

Implemented and merged as `mkakcl/dyson#5` (`cc2f48f`); see §5.1 for the measured size.

**Remaining step.** Nothing, until the pin next moves. When it does:

1. Bump `dyson @ git+...@<sha>` in `pyproject.toml`, reinstall into `mgw-monomial`.
2. `python -m baseline.check` — expect **52/52 unchanged**, since this changes no numbers.
   It was already verified against this build before merge.
3. Re-record in the same commit, as every pin move requires. Provenance-only.

**Do not bump the pin for this alone.** Re-recording 52 cases to collect under 1% is
disproportionate, and the fix is not going anywhere.

### 7.3 Option 1 — first block row only

**Goal.** Stop computing the 75% of the block-tridiagonal eigenvector matrix that
`_solve` discards.

**Where.** `dyson/solvers/static/mblse.py`, the `util.eig_lr(subspace, ...)` call inside
`_solve`, and its counterpart in `mblgf.py`. Nothing in momentGW changes.

**Steps.**

1. **Prototype outside the package first**, on the exact sizes from §3. Reduce the block
   tridiagonal to scalar tridiagonal, take eigenvalues from `dsterf`, and accumulate only
   the first `nphys` rows of the transformation through the QL sweeps — the Golub–Welsch
   structure. Compare against `numpy.linalg.eigh` for both time and agreement.
2. Only if it wins: implement behind an option, defaulting off, with the dense path retained
   as the reference.
3. Tests: eigenvalues and first-block-row couplings agree with the dense route to machine
   precision on the existing MBLSE/MBLGF fixtures, including the degenerate and clustered
   cases in `tests/test_mbl_realization.py`.
4. Pin bump and re-record. This one **will** move numbers at the last bit, so expect the
   deep-state movement described in Milestone 4 and check the frontier holds.

**Kill criterion — expect to use it.** If step 1 is not at least **2x faster than
`dsyevd`** on the real sizes before any tuning, stop and delete the prototype. §4 is the
precedent: a quarter of the arithmetic lost 5.5x to BLAS-3 in exactly this shape. There is
no LAPACK driver for "all eigenvalues, first `k` rows of eigenvectors", so this is a
hand-written sweep competing with a blocked, tuned library routine.

**Payoff if it works.** Under 2% of a calculation. **Effort**: 2–3 days, most of it step 1.
This is a prototype-and-probably-discard, and should be costed as one.

### 7.4 Option 2 — arrowhead / DPRk secular solver

**Goal.** Replace the dense `eigh` on the upfolded supermatrix with a structure-exploiting
solver, ~9x on the largest single eigendecomposition.

**Where.** `dyson/representations/lehmann.py:638`, inside `diagonalise_matrix`.

**Steps.**

1. **Reproduce the reference first.** Port or call Arrowhead.jl's algorithm on random
   arrowhead matrices and confirm the published accuracy claim before touching this problem.
   If the reference implementation cannot be reproduced, stop here.
2. **Two-stage reduction.** Diagonalise the `nphys` head (`nphys^3`, negligible), transform
   the border, leaving diagonal-plus-rank-`nphys`. Handle it as `nphys` successive rank-1
   updates, each an `O(n^2)` secular solve.
3. **Certificate, not a return code.** Per-eigenpair residual `||A x - lambda x||`, the
   orthogonality `||X^T X - I||`, and an explicit failure when a cluster cannot be resolved.
   This is the eta0 standard and it is not optional here: a secular solver that misconverges
   on a cluster returns plausible poles, and this project's whole premise is not accepting
   those.
4. Validate against the dense solve on all 52 baseline systems before it is allowed to be a
   default; ship it opt-in first, exactly as HHT was.
5. Pin bump and re-record.

**Kill criteria.** Stop if step 1 fails; if the measured speedup on the real sizes is below
**3x** (below that it is not worth the accuracy risk, and §4 says measure before believing);
or if clustered eigenvalues need extended precision in the inner loop and the result is no
longer faster once that is included.

**Payoff.** Competes for the same ~1.2x ceiling that Option 3 targets more cheaply, so it is
only worth starting if Option 3 is ruled out. **Effort**: weeks, and I have no basis for a
tighter figure — treat it as research with an open end.

### 7.5 Option 3 — windowed solve, and what it now requires

**Goal.** Compute only the poles that are wanted, and only the poles that are trustworthy.

**Where.** `dyson/representations/lehmann.py` for the solver mode; `momentGW/fock.py` and
`momentGW/gw.py` for the two consumers identified in §7.1.

This is now a two-part job, and the first part is the real one.

**Part A — decouple the electron count and density matrix from the eigenpair list.**

1. Implement the arrowhead resolvent as a first-class object — the 15-line operator
   prototyped for §5.1, plus its Schur-complement setup.
2. Evaluate `N(mu)` and the occupied-block density matrix by contour integration of that
   resolvent, and validate both against the current eigen-decomposition route on the
   baseline systems to the accuracy the particle-number gate already demands (`1e-6`
   on `conv_tol_nelec`).
3. Only when `N(mu)` and the RDM no longer require an eigenpair list does a windowed solve
   become possible at all.

**Part B — the windowed solve itself.**

4. Add a mode that returns a requested window plus a buffer, using shift-invert Lanczos with
   the Part A resolvent as `OPinv`.
5. Specify what a calculation in this mode may report: frontier QP energies yes; spectral
   functions and deep satellites no. The mode must be recorded in the diagnostics, and the
   off-diagonal-coupling check Milestone 4 already asks for belongs here.
6. Pin bump and re-record.

**Kill criteria.** Stop if Part A's contour cannot hit the particle-number tolerance at a
cost below the dense solve it replaces — that is the whole premise. Stop if the required
window, once the buffer is honest, exceeds ~10% of the spectrum, since §5.1 shows
shift-invert loses there.

**Payoff.** Bounded by §5.1 at 1.10x (cc-pVDZ) to 1.19x (cc-pVTZ), and only for calculations
that genuinely want a frontier band. **Effort**: Part A is 1–2 weeks and is where the risk
is; Part B is a few days on top. My earlier "1 day kernel plus a week of specification" was
written before §7.1 and understated this by a lot.

### 7.6 Suggested order, and the honest recommendation

1. Nothing, until a pin bump is wanted for another reason — then §7.2 rides along free.
2. If someone wants to spend two days on it: §7.3, expecting to discard it. It is
   self-contained and answers a question that will otherwise keep being asked.
3. §7.5 Part A **only if it is wanted for its own sake** — a cheap spectral projector for
   `N(mu)` and the density matrix is useful well beyond this document, and would also serve
   Milestone 3's error budget. Justifying it by the 1.19x alone does not work.
4. §7.4 last, and only if 3 is ruled out.

The recommendation from §6 stands and is strengthened by §7.1: the whole area is capped near
1.2x, the one option that looked like a clean win needs a substantial prerequisite before it
returns anything on the default path, and Milestone 3 is worth more than all of it.

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
