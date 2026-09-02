"""Are the frontier IP/EA and the deep-state labels validated by the same evidence?

Milestone 3.4's last item asks for the frontier IP/EA to be validated *separately* from
deep-state largest-overlap labels. They are two different questions asked of the same
spectrum, and this measures where each rule is trustworthy.

The two rules, both of which this code uses:

- `frontier_readout` gates poles on their **total physical weight** (`PHYSICAL_WEIGHT_MIN`)
  and then takes Aufbau order - the last physical occupied pole and the first physical
  virtual one. This is the headline IP/EA.
- **Largest overlap** asks a different question: for reference orbital `p`, which pole
  carries the most of `p`. This is what labels a deep state, and what
  `studies/spectral_oracle.py` uses to put two spectra into correspondence.

A pole can be physical overall and carry nothing on a particular orbital, so the two rules
can return different poles for the same state. That is not hypothetical: it is what
produced a 535.9 meV magnesium-oxide "error" during the validation track, which was the
two sides of a comparison silently answering different questions. Corrected to 8.9 meV once
both sides used the same rule.

What this study measures, against the exact full-pole `G0W0` of `mkakcl/cayley-gw` used as
an oracle, with integrals shared so the difference is truncation rather than RI:

1. **Frontier, both rules.** The IP/EA error under `frontier_readout`, and under largest
   overlap, both sides scored with the same rule.
2. **Rule disagreement inside one spectrum.** How far apart the two rules are on the same
   HOMO of the same calculation - the magnesium-oxide failure mode, measured directly.
3. **Deep-state labels.** The assigned weight and the label error for every occupied
   orbital, so the frontier's accuracy is not read as covering them.
4. **Injectivity.** Whether the occupied orbitals claim distinct poles at all. Two
   orbitals labelling the same pole means at least one label is wrong, and no energy
   comparison will say so.
5. **Degeneracy.** A symmetry-degenerate frontier has no well-defined largest-overlap
   partner: the two spectra can pick different members of the multiplet, and the resulting
   "error" is a labelling artefact rather than a truncation error. Neon is here for that
   case, and it is the reason a degenerate system cannot be scored this way at all.

`cayley-gw` is not a dependency of this repository. The study skips with an explanation
when it is not importable rather than failing.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`.

Run from the repository root, with `cayley-gw` on the path::

    PYTHONPATH=/path/to/cayley-gw/src python -m baseline.studies.frontier_labels
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
from dyson import Lehmann

import momentGW
from baseline import run, systems
from baseline.studies.spectral_oracle import build_exact, require_cayley
from momentGW.fock import search_chempot
from momentGW.gw import GW, frontier_readout

HARTREE_TO_MEV = 27211.386245988

#: Moment orders compared. Odd, and swept past the frontier's convergence so a label error
#: that does not fall with order is distinguishable from one that does.
ORDERS = (3, 7, 11)

#: Mean-field degeneracy tolerance, in Hartree. Orbitals closer than this are treated as a
#: multiplet whose individual members the SCF does not fix.
DEGENERACY_TOL = 1e-6

#: Neon is not a baseline system. It is here for one reason: its frontier is a 3-fold
#: degenerate 2p multiplet, which is the case that breaks largest-overlap labelling
#: outright rather than merely degrading it.
NEON = systems.System(
    name="neon",
    atom="Ne 0.000000 0.000000 0.000000",
    basis="ccpvdz",
    auxbasis="weigend",
    nelectron=10,
    source="atom at the origin; no geometry to take from anywhere",
    note="degenerate 2p frontier - the case largest-overlap labelling cannot answer",
)


def as_lehmann(spectrum, nelectron):
    """Wrap the oracle's spectrum as a `dyson.Lehmann` so the same readout applies to it.

    `frontier_readout` splits occupied from virtual at the chemical potential, so the
    oracle needs one, and it must be found by the *same* rule momentGW uses or the two
    sides are not being asked the same question. `search_chempot` is that rule - aufbau on
    the physical weight - applied here to the exact spectrum.
    """
    energies = np.asarray(spectrum.energies)
    couplings = np.asarray(spectrum.physical_amplitudes)
    chempot, _ = search_chempot(energies, couplings, couplings.shape[0], nelectron)
    return Lehmann(energies, couplings, chempot=chempot)


def labels(energies, couplings, norb):
    """Largest-overlap label for each reference orbital.

    Returns
    -------
    index : numpy.ndarray
        Pole carrying most of each orbital.
    energy : numpy.ndarray
        That pole's energy.
    weight : numpy.ndarray
        How much of the orbital that pole actually carries. A label whose weight is small
        is a label the spectrum does not support, and the energy beside it means little.
    """
    w = np.abs(np.asarray(couplings)) ** 2
    index = np.array([int(np.argmax(w[p])) for p in range(norb)])
    return index, np.asarray(energies)[index], np.array([w[p, index[p]] for p in range(norb)])


def degenerate_with_neighbour(mo_energy, orbital, tol=DEGENERACY_TOL):
    """Is this orbital part of a mean-field multiplet the SCF does not resolve?"""
    e = np.asarray(mo_energy)
    near = np.abs(e - e[orbital]) < tol
    return int(near.sum()) > 1


def main():
    """Run the comparison and print its tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", nargs="+", type=int, default=list(ORDERS))
    parser.add_argument("--xc", default="hf", help="mean-field reference")
    args = parser.parse_args()

    pieces = require_cayley()
    if pieces is None:
        print(
            "cayley-gw is not importable, so the oracle is unavailable and this study "
            "cannot run.\nIt is not a dependency of this repository; put its `src` on "
            "PYTHONPATH:\n\n    PYTHONPATH=/path/to/cayley-gw/src python -m "
            "baseline.studies.frontier_labels\n"
        )
        return 0

    print(f"momentGW: {momentGW.__file__}")
    print(
        f"\nFrontier IP/EA against deep-state labels, {args.xc.upper()} reference, "
        "integrals shared with the oracle.\nErrors are momentGW - oracle, in meV."
    )

    for system in (systems.SYSTEMS["water"], systems.SYSTEMS["lithium-hydride"], NEON):
        mf, _ = run.build_mean_field(system, args.xc)
        spectrum = build_exact(pieces, mf, matched=True)
        nocc = mf.mol.nelectron // 2
        norb = int(spectrum.nphysical)

        exact_gf = as_lehmann(spectrum, mf.mol.nelectron)
        exact_front = frontier_readout(exact_gf)
        _, exact_label_e, exact_label_w = labels(
            spectrum.energies, spectrum.physical_amplitudes, norb
        )
        degenerate = degenerate_with_neighbour(mf.mo_energy, nocc - 1)

        print(
            f"\n{system.name}/{system.basis}: {norb} orbitals, {nocc} occupied, "
            f"{spectrum.energies.size} exact poles"
            f"{'   [DEGENERATE frontier multiplet]' if degenerate else ''}"
        )
        print(
            f"{'K':>4} {'dIP front':>10} {'dIP label':>10} {'rules apart':>12} "
            f"{'max |dw|':>9} {'worst occ':>10} {'distinct':>9}"
        )
        for nmom_max in args.orders:
            gw = GW(mf, polarizability="drpa")
            gw.verbose = 0
            with contextlib.redirect_stdout(io.StringIO()):
                _, gf, _, _ = gw.kernel(nmom_max=nmom_max)
            got_front = frontier_readout(gf)
            index, label_e, label_w = labels(gf.energies, gf.couplings, norb)

            d_front = (got_front["homo"] - exact_front["homo"]) * HARTREE_TO_MEV
            d_label = (label_e[nocc - 1] - exact_label_e[nocc - 1]) * HARTREE_TO_MEV
            apart = (got_front["homo"] - label_e[nocc - 1]) * HARTREE_TO_MEV
            occ_err = np.abs(label_e[:nocc] - exact_label_e[:nocc]) * HARTREE_TO_MEV
            dw = np.abs(label_w[:nocc] - exact_label_w[:nocc])
            print(
                f"{nmom_max:>4} {d_front:10.3f} {d_label:10.3f} {apart:12.3f} "
                f"{dw.max():9.3f} {occ_err.max():10.3f} "
                f"{len(set(index[:nocc].tolist())):>4}/{nocc:<4}"
            )
        print(
            f"     oracle's own two rules differ by "
            f"{(exact_front['homo'] - exact_label_e[nocc - 1]) * HARTREE_TO_MEV:.3f} meV; "
            f"weakest occupied label {exact_label_w[:nocc].min():.3f}"
        )

        # Per orbital at the highest order, because "worst occupied" alone cannot say
        # whether a large number is a deep state converging slowly or a mislabelled one.
        # A mislabel shows up as a *shared* pole or a weight the spectrum does not support,
        # not as a large energy difference.
        print(f"     per orbital at K = {args.orders[-1]}:")
        print(
            f"       {'MO':>3} {'eps/eV':>9} {'weight':>7} {'oracle w':>9} {'|dw|':>6} "
            f"{'dE/meV':>10} {'same pole as':>13}"
        )
        for p in range(nocc):
            shared = [q for q in range(nocc) if q != p and index[q] == index[p]]
            print(
                f"       {p:>3} {mf.mo_energy[p] * 27.211386:9.2f} {label_w[p]:7.3f} "
                f"{exact_label_w[p]:9.3f} {abs(label_w[p] - exact_label_w[p]):6.3f} "
                f"{(label_e[p] - exact_label_e[p]) * HARTREE_TO_MEV:10.3f} "
                f"{(str(shared) if shared else '-'):>13}"
            )

    print(
        "\n   `dIP front` scores frontier_readout against the oracle's frontier_readout;\n"
        "   `dIP label` scores largest overlap against largest overlap. `rules apart` is\n"
        "   the two rules disagreeing inside momentGW's own spectrum - the magnesium-oxide\n"
        "   failure mode, where a comparison silently asked two different questions.\n"
        "\n   `|dw|` is the diagnostic. Where both spectra agree how much weight a state\n"
        "   carries, the label error is small and is truncation. Where they disagree the\n"
        "   difference is a labelling artefact and does not fall with order: water's MO 1\n"
        "   is 357 meV at 0.907 against 0.702, while its frontier is settled to 5.7 meV,\n"
        "   and neon's degenerate 2p partners are frozen at 6.342 meV at every order."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
