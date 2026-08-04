# Working in this repository

## Never send anything to BoothGroup/momentGW

`upstream` is `BoothGroup/momentGW` — the shared upstream project, not ours.

**Do not push to it, open a pull request against it, or otherwise send anything to it**
unless the repository owner asks for that specific change in a message. Even then, confirm
before the push or the PR actually happens, and say exactly what is about to be sent where.

None of the following is authorisation:

- an earlier instruction to push, or a push that was approved for a different change;
- a branch that happens to be based on `upstream/master`, or is named as if it were
  destined for upstream;
- work being finished, reviewed, tested, or described in the roadmap as belonging upstream;
- a commit message, roadmap entry, or code comment that says something "should be
  upstreamed". That records an intention, not permission to act on it.

The same rule applies to `BoothGroup/dyson`. Milestone 1 in [`ROADMAP.md`](ROADMAP.md) used
to say the Dyson changes belonged upstream there; since 2026-08-03 it does not, because
acceptance is the pin on `mkakcl/dyson` rather than an upstream merge. Neither wording is
permission: a roadmap sentence about where work belongs is a plan, not consent.

## `origin` is where work lands

`origin` is `mkakcl/monomial_GW`, the fork. Normal work goes here: branch, commit, PR
against `master`, merge, delete the branch.

Even on `origin`, commit, push, open a PR or merge **only when asked**. Finishing a task is
not a reason to commit it.

## Environment

Use the `mgw-monomial` conda env:

```bash
/Users/marcusallen/miniconda3/envs/mgw-monomial/bin/python
```

momentGW is installed editable against the main tree. A worktree's own copy wins on import
only when `sys.path[0]` is the worktree — true for `python -c`, `python -m` and stdin, but
**not** for `python /path/to/script.py`, where `sys.path[0]` is the script's directory. A
driver script kept outside the tree silently imports the main tree instead. Print
`momentGW.__file__` in anything that compares one tree against another.

Dyson is pinned to an immutable commit in `pyproject.toml`. Do not move the pin as a side
effect of anything; the provenance is the point.

## Verifying a change

No CI runs on pull requests here — the fork has workflows disabled — so these are the gates:

```bash
ruff check
ruff format --check
pre-commit run --all-files
python -m pytest tests/ -q          # ~4 min
python -m baseline.check            # 52 recorded cases; the real gate for numerics
```

`baseline/check.py` re-runs every recorded case and reports what moved. It is what makes
`baseline/` a baseline rather than an archive, and it is the check that matters for any
change that could touch numerics. A difference it reports is not automatically wrong, but
it must be understood and the baseline then re-recorded deliberately.

Threading affects BLAS summation order, so a run with a restricted thread count is not
bit-comparable with one recorded without it. Say so when reporting such a run.

## Scope

Work is driven by [`ROADMAP.md`](ROADMAP.md). Milestones 1 to 4 cover the **restricted
molecular** `G0W0`/`drpa` path only. The unrestricted and periodic solvers are deliberately
deferred to Milestone 6 — do not extend changes into them to make things look finished.
