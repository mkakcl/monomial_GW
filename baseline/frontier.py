"""Read frontier quasiparticle energies out of a correlated charged spectrum.

The Green's function returned by a moment-constrained G0W0 calculation is a Lehmann
representation: pole energies plus, for each pole, its projection onto the reference
molecular orbitals. Two different readouts are taken from it, and they are not the same
quantity.

`aufbau_frontier` finds the HOMO and LUMO of the *correlated* spectrum by electron
counting, with degenerate poles grouped into multiplets. It does not use the reference
orbital ordering, so it stays correct through a level crossing.

`orbital_table` reports, for each *reference* MO, the multiplet carrying most of that
orbital's spectral weight. This is what makes "how did this orbital move when the moment
order changed" a well-defined question: poles are not comparable across a sweep by index,
because the index changes.

Notes
-----
The algorithm and the two thresholds are taken from `harness/mgw_harness/spectrum.py` in
`mkakcl/molecular-mGW-testing`, so that the numbers recorded here can be compared directly
against that harness's results for the same systems. The reasoning behind
`PHYSICAL_WEIGHT_MIN` in particular is recorded there at length: a moment-truncated
spectrum carries a dense forest of low-weight satellites between the HOMO and the true
LUMO, and a permissive threshold selects one of them as the frontier orbital.
"""

import numpy as np

#: Conversion from Hartree to electronvolt.
EV = 27.211386245988

#: Poles closer together than this are treated as one multiplet. Degeneracy is exact in
#: theory and numerical in practice; grouping is what makes a summed weight physical.
DEGENERACY_ATOL = 1.0e-6

#: A multiplet carrying less total weight than this is a satellite, not a quasiparticle,
#: and cannot be a frontier orbital.
PHYSICAL_WEIGHT_MIN = 1.0e-1

#: Thresholds probed to report how stable the frontier is. The default must sit inside
#: this range, not at its edge, or the report cannot say whether the plateau extends above
#: the value in force.
STABILITY_PROBES = (1e-3, 1e-2, 1e-1, 0.3, 0.5)


class SpectrumError(ValueError):
    """The spectrum does not support the readout being asked for."""


def orbital_weights(couplings):
    """Get the per-orbital spectral weight of each pole.

    Parameters
    ----------
    couplings : numpy.ndarray
        Lehmann couplings, with shape `(nmo, npoles)`.

    Returns
    -------
    weights : numpy.ndarray
        Spectral weight of each pole on each orbital, with shape `(nmo, npoles)`.
    """
    return np.abs(np.asarray(couplings)) ** 2


def _as_sorted(energies, weights):
    """Sort the poles by energy, validating the shapes."""
    e = np.asarray(energies, dtype=float).ravel()
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[1] != e.size:
        raise SpectrumError(f"weights must be (nmo, npoles); got {w.shape} for {e.size} poles")
    if e.size == 0:
        raise SpectrumError("empty spectrum")
    if np.any(np.diff(e) < 0.0):
        order = np.argsort(e, kind="stable")
        e, w = e[order], w[:, order]
    return e, w


def multiplets(energies, weights, atol=DEGENERACY_ATOL):
    """Group degenerate poles and sum their orbital-resolved weights.

    Parameters
    ----------
    energies : numpy.ndarray
        Pole energies.
    weights : numpy.ndarray
        Per-orbital spectral weights, with shape `(nmo, npoles)`.
    atol : float, optional
        Poles closer than this are one multiplet. Default value is
        `DEGENERACY_ATOL`.

    Returns
    -------
    grouped : dict
        Multiplet energies, sizes, orbital-resolved weights and total
        (physical) weights.
    """
    e, w = _as_sorted(energies, weights)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(e) > atol) + 1)).astype(np.int64)
    sizes = np.append(starts[1:], e.size) - starts
    grouped = np.add.reduceat(w, starts, axis=1)
    return {
        "starts": starts,
        "sizes": sizes,
        "energies": np.add.reduceat(e, starts) / sizes,
        "orbital_weights": grouped,
        "physical_weights": np.sum(grouped, axis=0),
    }


def orbital_table(energies, weights, mo_energy=None, mo_occ=None):
    """Get the multiplet carrying most of each reference orbital's spectral weight.

    Parameters
    ----------
    energies : numpy.ndarray
        Pole energies.
    weights : numpy.ndarray
        Per-orbital spectral weights, with shape `(nmo, npoles)`.
    mo_energy : numpy.ndarray, optional
        Reference orbital energies, recorded alongside if given. Default
        value is `None`.
    mo_occ : numpy.ndarray, optional
        Reference orbital occupancies, recorded alongside if given.
        Default value is `None`.

    Returns
    -------
    table : dict
        Per-orbital quasiparticle energies, weights and multiplicities.
    """
    grouped = multiplets(energies, weights)
    contributions = grouped["orbital_weights"]
    best = np.argmax(contributions, axis=1)
    rows = np.arange(contributions.shape[0])
    nmo = contributions.shape[0]

    table = {
        "mo_index": list(range(nmo)),
        "qp_energy_ha": [float(x) for x in grouped["energies"][best]],
        "qp_weight": [float(x) for x in contributions[rows, best]],
        "multiplicity": [int(x) for x in grouped["sizes"][best]],
    }
    if mo_energy is not None:
        table["reference_energy_ha"] = [float(x) for x in np.asarray(mo_energy).ravel()[:nmo]]
    if mo_occ is not None:
        occ = np.asarray(mo_occ).ravel()[:nmo]
        table["occupation"] = [float(x) for x in occ]
        filled = np.flatnonzero(occ > 0.0)
        empty = np.flatnonzero(occ == 0.0)
        if filled.size and empty.size:
            table["reference_homo_index"] = int(filled[-1])
            table["reference_lumo_index"] = int(empty[0])
    return table


def aufbau_frontier(energies, weights, nelectron, weight_min=None, probe=True):
    """Get the HOMO and LUMO of the correlated spectrum by Aufbau electron counting.

    Restricted and closed-shell, so each unit of spectral weight holds two electrons.
    The multiplets are walked from the bottom until the reference electron count is
    reached; the chemical potential sits between that multiplet and the next, and the
    frontier orbitals are the nearest *physical* multiplets on either side of it.

    Parameters
    ----------
    energies : numpy.ndarray
        Pole energies.
    weights : numpy.ndarray
        Per-orbital spectral weights, with shape `(nmo, npoles)`.
    nelectron : int
        Reference number of electrons.
    weight_min : float, optional
        Minimum total weight for a multiplet to be a quasiparticle. If
        `None`, use `PHYSICAL_WEIGHT_MIN`. Default value is `None`.
    probe : bool, optional
        Whether to report which probe thresholds give the same frontier.
        Default value is `True`.

    Returns
    -------
    frontier : dict
        Frontier energies, weights, the chemical potential implied by the
        electron count, and the electron-count error.

    Raises
    ------
    SpectrumError
        If the spectrum cannot support the readout, rather than returning
        a number derived from an incomplete pole set.
    """
    grouped = multiplets(energies, weights)
    e, physical_weights = grouped["energies"], grouped["physical_weights"]
    electrons = 2.0 * physical_weights

    if float(np.sum(electrons)) < nelectron:
        raise SpectrumError(
            f"the correlated spectrum carries {float(np.sum(electrons)):.3f} electrons, fewer "
            f"than the reference's {nelectron} -- the pole set is incomplete"
        )

    cumulative = np.cumsum(electrons)
    index = int(np.searchsorted(cumulative, nelectron))
    if index >= e.size:
        raise SpectrumError("could not place the chemical potential inside the spectrum")
    below = float(cumulative[index - 1]) if index > 0 else 0.0
    above = float(cumulative[index])
    take_below = abs(below - nelectron) < abs(above - nelectron)
    occupied_end = index - 1 if take_below else index
    if not 0 <= occupied_end < e.size - 1:
        raise SpectrumError("could not place the chemical potential inside the spectrum")

    chempot = 0.5 * float(e[occupied_end] + e[occupied_end + 1])
    electron_error = nelectron - (below if take_below else above)

    weight_min = PHYSICAL_WEIGHT_MIN if weight_min is None else weight_min
    physical = physical_weights > weight_min
    occupied = physical & (e < chempot)
    virtual = physical & (e >= chempot)
    if not occupied.any() or not virtual.any():
        raise SpectrumError("no physical multiplet on one side of the chemical potential")
    homo_index = int(np.flatnonzero(occupied)[-1])
    lumo_index = int(np.flatnonzero(virtual)[0])

    def record(i):
        """Describe one frontier multiplet."""
        contributions = grouped["orbital_weights"][:, i]
        dominant = int(np.argmax(contributions))
        return {
            "energy_ha": float(e[i]),
            "energy_ev": float(e[i] * EV),
            # The multiplet total, summed over poles and over orbitals, so a degenerate
            # multiplet exceeds one. The conventional quasiparticle weight Z is per state
            # and bounded by one -- that is `weight_per_state`.
            "weight": float(physical_weights[i]),
            "weight_per_state": float(physical_weights[i] / grouped["sizes"][i]),
            "multiplicity": int(grouped["sizes"][i]),
            "dominant_mo_index": dominant,
            "dominant_mo_weight": float(contributions[dominant]),
        }

    stable_over = None
    if probe:
        stable_over = _stability(energies, weights, nelectron, e[homo_index], e[lumo_index])

    return {
        "definition": (
            "physical correlated-spectrum multiplets split at the Aufbau chemical potential; "
            "independent of the reference MO ordering, so it stays correct through a crossing"
        ),
        "degeneracy_atol_ha": DEGENERACY_ATOL,
        "physical_weight_min": weight_min,
        "chemical_potential_ha": chempot,
        "electron_count_error": float(electron_error),
        "homo": record(homo_index),
        "lumo": record(lumo_index),
        "gap_ha": float(e[lumo_index] - e[homo_index]),
        "n_poles": int(np.asarray(energies).size),
        "n_multiplets": int(e.size),
        "total_weight": float(np.sum(physical_weights)),
        "stable_over": stable_over,
    }


def _stability(energies, weights, nelectron, homo_energy, lumo_energy):
    """Report which probe thresholds give this same frontier.

    A short list means the frontier sits on a cliff and the number should not be trusted
    without looking at the spectrum; a long one means the threshold is doing no work.
    """
    out = []
    for threshold in STABILITY_PROBES:
        try:
            found = aufbau_frontier(energies, weights, nelectron, weight_min=threshold, probe=False)
        except SpectrumError:
            continue
        if (
            abs(found["homo"]["energy_ha"] - homo_energy) < 1e-9
            and abs(found["lumo"]["energy_ha"] - lumo_energy) < 1e-9
        ):
            out.append(threshold)
    return out


def readouts(energies, couplings, nelectron, mo_energy=None, mo_occ=None):
    """Get the full frontier and per-orbital readout of a correlated spectrum.

    Parameters
    ----------
    energies : numpy.ndarray
        Pole energies.
    couplings : numpy.ndarray
        Lehmann couplings, with shape `(nmo, npoles)`.
    nelectron : int
        Reference number of electrons.
    mo_energy : numpy.ndarray, optional
        Reference orbital energies. Default value is `None`.
    mo_occ : numpy.ndarray, optional
        Reference orbital occupancies. Default value is `None`.

    Returns
    -------
    results : dict
        Headline frontier energies, the frontier record, and the
        per-orbital table.
    """
    weights = orbital_weights(couplings)
    front = aufbau_frontier(energies, weights, nelectron)
    table = orbital_table(energies, weights, mo_energy, mo_occ)
    homo_ha, lumo_ha = front["homo"]["energy_ha"], front["lumo"]["energy_ha"]
    return {
        "homo_ha": homo_ha,
        "lumo_ha": lumo_ha,
        "gap_ev": front["gap_ha"] * EV,
        "ip_ev": -homo_ha * EV,
        "ea_ev": -lumo_ha * EV,
        "homo_index": table.get("reference_homo_index"),
        "frontier": front,
        "orbital_table": table,
        "qp_energies_ha": table["qp_energy_ha"],
        "qp_weights": table["qp_weight"],
    }
