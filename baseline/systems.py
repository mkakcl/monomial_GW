"""Systems and mean-field references for the Milestone 0 baseline.

Four molecules, chosen to cover the range the restricted molecular G0W0/dRPA path has to
survive: a two-electron system where the moment expansion is exactly convergent at low
order, two ordinary closed-shell molecules, and one small-gap system where the
particle-hole denominators entering the dRPA integrand come close to zero.

Geometries are the GW100 structures, taken from the `gw100` set of
`mkakcl/molecular-mGW-testing` (`datasets/sets/gw100/systems.json`, built from
`gmtkn@047c548` and `gw100_data@bafa6ab`, cross-checked against the published
`setten/GW100` structures to 0 Angstrom). Using those exact structures, basis and
auxiliary basis means every case here is directly comparable to the results that harness
has already recorded for the same systems, rather than being a private set of numbers.

The one exception is `hydrogen-631g`, which is not a GW100 case. It reproduces the
worked example in the repository README, and exists as an external anchor: an independent
value for this pipeline that was recorded before any of this roadmap's work began.
"""

import dataclasses

#: Published G0W0@PBE HOMO energies for the GW100 systems here, in eV, as redistributed by
#: `molecular-mGW-testing` (`datasets/sets/gw100/reference.json`, series
#: `g0w0_pbe_turbomole_qzvp`). Recorded for orientation only. They are complete-basis
#: TURBOMOLE values, so they are not a target for a cc-pVDZ calculation; the baseline's job
#: is to be stable under our own changes, not to reproduce these.
PUBLISHED_G0W0_PBE_HOMO_EV = {
    "hydrogen": -15.812,
    "water": -11.972,
    "lithium-hydride": -6.553,
    "ozone": -11.401,
}


@dataclasses.dataclass(frozen=True)
class System:
    """A molecule and the basis it is run in.

    Parameters
    ----------
    name : str
        Identifier, used in case identifiers and filenames.
    atom : str
        Geometry in PySCF's format, in Angstrom.
    basis : str
        Orbital basis.
    auxbasis : str or None
        Density-fitting auxiliary basis. `None` uses PySCF's default for
        the orbital basis.
    nelectron : int
        Number of electrons, asserted against the built molecule.
    source : str
        Where the geometry came from.
    note : str
        Why this system is in the set.
    """

    name: str
    atom: str
    basis: str
    auxbasis: str | None
    nelectron: int
    source: str
    note: str

    def to_dict(self):
        """Get a JSON-serialisable record of the system.

        Returns
        -------
        record : dict
            The system definition.
        """
        return dataclasses.asdict(self)


SYSTEMS = {
    system.name: system
    for system in [
        System(
            name="hydrogen",
            atom="H 0.000000 0.000000 0.000000; H 0.000000 0.000000 0.741440",
            basis="ccpvdz",
            auxbasis="weigend",
            nelectron=2,
            source="GW100 (HCP92 experimental structure) via molecular-mGW-testing gw100 set",
            note=(
                "Two electrons and nine RPA poles, so the moment expansion is converged by "
                "order 7 -- moment-truncation error can be driven out of the comparison here."
            ),
        ),
        System(
            name="water",
            atom=(
                "O 0.000000 0.000000 0.000000; "
                "H 0.757100 0.000000 0.586100; "
                "H -0.757100 0.000000 0.586100"
            ),
            basis="ccpvdz",
            auxbasis="weigend",
            nelectron=10,
            source="GW100 (HCP92 experimental structure) via molecular-mGW-testing gw100 set",
            note=(
                "The ordinary well-behaved case. Also the control half of the "
                "compressed/uncompressed pair: it has fewer auxiliary functions than "
                "particle-hole pairs, so the compression has no null directions to remove "
                "and is exactly a no-op."
            ),
        ),
        System(
            name="lithium-hydride",
            atom="Li 0.000000 0.000000 0.000000; H 0.000000 0.000000 1.594900",
            basis="ccpvdz",
            auxbasis="weigend",
            nelectron=4,
            source="GW100 (HCP92 experimental structure) via molecular-mGW-testing gw100 set",
            note=(
                "Strongly ionic, with a diffuse and weakly bound LUMO -- the published "
                "G0W0@PBE and CCSD(T) HOMO energies differ by 1.4 eV, more than any other "
                "system here. It has more auxiliary functions than particle-hole pairs, so "
                "it is the half of the compressed/uncompressed pair where the compression "
                "actually removes something: 60 auxiliaries down to 34."
            ),
        ),
        System(
            name="ozone",
            atom=(
                "O 0.000000 0.000000 0.000000; "
                "O 1.086900 0.000000 0.660000; "
                "O -1.086900 0.000000 0.660000"
            ),
            basis="ccpvdz",
            auxbasis="weigend",
            nelectron=24,
            source="GW100 (HCP92 experimental structure) via molecular-mGW-testing gw100 set",
            note=(
                "The small-gap member of the set: a low-lying LUMO gives the smallest "
                "particle-hole gap in this set, which is what the dRPA integrand and the "
                "realization both have to cope with. The measured gap is recorded per case."
            ),
        ),
        System(
            name="hydrogen-631g",
            atom="H 0.000000 0.000000 0.000000; H 0.000000 0.000000 0.740000",
            basis="6-31g",
            auxbasis=None,
            nelectron=2,
            source="repository README worked example (not a GW100 geometry)",
            note=(
                "External anchor. G0W0@HF at nmom_max=1 reproduces the value recorded in the "
                "README and in molecular-mGW-testing: HOMO -16.047 eV, LUMO 6.535 eV."
            ),
        ),
    ]
}

#: Mean-field starting points. Both are run for every GW100 system: momentGW's own
#: regression tests use a Hartree-Fock starting point, while every published G0W0 reference
#: value and every result recorded by `molecular-mGW-testing` uses PBE. Keeping both means
#: a later change can be shown to be starting-point independent, and keeps starting-point
#: uncertainty separate from numerical error.
STARTING_POINTS = ("hf", "pbe")

#: SCF convergence tolerance, matching the `molecular-mGW-testing` reference builder.
SCF_CONV_TOL = 1e-10

#: Moment orders. Odd only: the MBLSE construction conserves `2 * iteration + 2` moments,
#: so an even `nmom_max` supplies a moment the recurrence has no block for.
MOMENT_ORDERS = (1, 3, 5, 7)
