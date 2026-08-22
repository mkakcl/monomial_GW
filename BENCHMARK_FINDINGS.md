# Benchmark findings — molecular G0W0 against exact references

**Written 2026-08-22.** The group's benchmark harness ran this code (fork tip 7bb941c and
the BoothGroup baseline 462f568, dyson 0f37b54) on molecular, restricted G0W0 —
GW100-scale systems, cc-pVDZ against DF-matched exact oracles and def2-TZVPP against
published values — over four weeks and ~13,000 calculations. This note records what a user
of this package should know. Conventions: errors are method − reference, in meV; screening
and reference are named per number.

## Convergence in `nmom_max`

- **Quote results at `nmom_max` ≤ 15.** The error against an exact oracle reaches a
  conditioning minimum near 15 and then *diverges*: CO2 (cc-pVDZ, dRPA@PBE) is 15.7 meV at
  nmom_max=21 and 113.3 meV at 31; N2 is better at 15 than at 31. A "stopped moving by
  10 meV" plateau criterion cannot distinguish this turnaround from convergence, and the
  code exposes no delivered-order diagnostic that would reveal it from the result object.
- **nmom_max=11 is a floor, not convergence, at production scale**: over all 102 GW100
  systems at def2-TZVPP the median |HOMO(9) − HOMO(11)| is still 23.6 meV (dRPA@PBE;
  8.4 meV under dTDA@HF, which converges ~3× faster in the moment order).
- **Even orders carry no information**: nmom_max = 2n and 2n−1 give bit-identical HOMO and
  LUMO with identical pole counts across the whole 1..12 ladder (pole counts grow only on
  odd steps — the block-iteration signature, moments = 2·n_iter + 2). Sweep odd orders
  only.
- **One build serves every order.** The nth self-energy moment does not depend on
  `nmom_max`: truncating a stored nmom_max=11 moment set and re-solving reproduces a
  native lower-order run to 0.000e+00 meV. A ladder therefore costs one moment build —
  but the archive must keep `se_static` alongside the hole/particle moments or the
  spectrum cannot be re-diagonalised later.
- At converged order the residual against a DF-matched exact oracle is real and
  system-dependent: −81 meV on water, −412 meV on lithium-fluoride (cc-pVDZ, dRPA@PBE),
  the latter a pole-assignment difference that no moment order repairs — confirmed by an
  independent second oracle.

## Quadrature and compression defaults

- **`npoints` matters only for dRPA** — `tda.py` builds its moments by explicit recursion
  and contains no quadrature, so npoints is bit-inert under `polarizability="dtda"`.
- Under dRPA the default `npoints=48` is −60.6 meV wrong on magnesium-monoxide (converges
  at 96, flat to 0.2 meV after), and **lithium-fluoride has no valid npoints datum at
  all**: its HOMO wanders over a 41 meV span with no monotone trend out to 1024 points,
  and moves 5–38 meV across SCF-reference rebuilds at fixed npoints. Since the cost is
  flat in npoints (2.2–2.8 s/cell from 8 to 256 points), there is nothing saved by running
  low: 256 was read as the safe-for-all setting.
- **`compression='ia'` (tol 1e-10) applies silently unless explicitly set to `None`.** The
  benchmark's GW100 production runs required `compression=None`; anyone comparing against
  another code should state which was in force.

## Fork vs baseline

- The fork reproduces the BoothGroup baseline to 3.3 µeV worst-case (guanine) while
  running faster (fitted nao^1.97 vs nao^2.17 — a four-point fit; re-fit before quoting
  hard). Two attribution corrections from the benchmark's follow-up: the residual is an
  *algorithm* difference (the fork's HHT/Zolotarev η⁰ default versus the baseline's
  Clenshaw–Curtis at npoints=48), not floating-point reassociation; and the fork's
  recorded dyson provenance did not match its own pyproject pin in that campaign — worth
  re-checking in any fork-vs-baseline study, since both halves (momentGW and dyson) move
  together.
- The measured Dyson-stage share is 35–46% of a restricted molecular G0W0 cell and *rises*
  with basis at fixed molecule — the O(N³)-vs-O(N⁴) argument for ignoring it holds only
  for growing the molecule.
- `MBLSE.reconstruct_moments` (dyson) silently drops the two newest moments from its own
  error check (`MLSE` carries the same defect; `MBLGF` is correct). Fixed on the pinned
  mkakcl/dyson; the defect is live for anyone using upstream dyson's version.

## Open items the benchmark could not close

- Against published G0W0@PBE values at matched def2-TZVPP basis the deviation depends
  strongly on the subset: median 93 meV on the benchmark's nine-system development set but
  ~146 meV over all 102 GW100 systems (only 37 of 102 within 100 meV), with
  lithium-fluoride at −750 meV and magnesium-monoxide at −1061 meV. Shown *not* to be the
  fork (fork ≡ baseline to µeV) and not the auxiliary basis. The decisive unrun test is a
  two-cell Clenshaw–Curtis-vs-HHT pair on those two systems.
- Reporting practice that would have prevented several misreadings: publish the
  quasiparticle weight next to every energy (below Z ≈ 0.7 the energy alone stops meaning
  much — several apparently flat convergence curves were satellites), and screen on the
  per-state weight, not the multiplicity-summed multiplet weight.
