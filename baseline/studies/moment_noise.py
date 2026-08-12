"""Measure the self-energy moment error at production settings, per order.

[`HANKEL_PENCIL.md`](../../HANKEL_PENCIL.md) §4 states a noise ceiling for the
one-shot Hankel pencil, and §6.A makes this measurement the gate in front of
every proposal there.  The noise rows in that document are *synthetic*
i.i.d. Gaussian perturbations of exact moments; this measures the real thing,
so the two can be read against each other.

The structure of the dRPA path decides how to measure it.  `build_dd_moments`
(`rpa.py:382`) takes the zeroth moment from `build_zeroth_dd_moment`, sets
`moments[1] = Lia * d` exactly, and builds every higher order by an exact
algebraic recursion.  **The whole path therefore has one numerical error
source: eta0.**  There is nothing to converge in the higher moments and no
quadrature to refine -- `npoints` is read only by the legacy Clenshaw-Curtis
route (`base.py:67`), and the default `eta0_method` is `"hht"`.

So the error is measured against an exact reference: `dense_eta0` is a dense
eigendecomposition of Mtilde, which is the same quantity HHT approximates.
Running the production pipeline from each and differencing gives the real
per-order relative error in the self-energy moments, at the shipped
`eta0_tol=1e-14`.

This complements [`eta0_amplification.py`](eta0_amplification.py) rather than
repeating it.  That study sweeps the HHT pole count to show that the dd-moment
recurrence does not amplify an eta0 perturbation, and reports the self-energy
error as a maximum over orders.  This one fixes the settings at what actually
ships and resolves the error by order, which is the form §4 needs.

Two limits on what this bounds.  The reference is a dense eigendecomposition in
the same float64 arithmetic, so it bounds the error *relative to exact eta0*,
not the total distance from the exact self-energy moments -- basis, SCF
convergence and density fitting are common to both sides and cancel.  And
compression is off, so the auxiliary compression error is excluded
deliberately; `--compression` puts it back.

A study, not part of the recorded baseline set: it is re-run when the claim it
supports is in question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the
printed `momentGW.__file__` before believing a comparison::

    python -m baseline.studies.moment_noise
    python -m baseline.studies.moment_noise --systems water_hf --nmom-max 11
    python -m baseline.studies.moment_noise --compression ia
"""

import argparse
import contextlib
import io

import numpy as np
from pyscf import dft, gto, scf

import momentGW
from momentGW.gw import GW
from momentGW.rpa import dRPA

#: The same three systems as `eta0_amplification`, so the two studies are read together.
#: Ozone is the small-gap case, where the certified interval is worst conditioned.
SYSTEMS = {
    "water_hf": (
        "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        "cc-pvdz",
        "hf",
    ),
    "lih_hf": ("Li 0 0 0; H 0 0 1.595", "cc-pvdz", "hf"),
    "ozone_pbe": (
        "O 0.000000 0.000000 0.000000; O 1.086900 0.000000 0.660000; O -1.086900 0.000000 0.660000",
        "cc-pvdz",
        "pbe",
    ),
}


def dense_eta0(d, Lia):
    """Dense eigendecomposition reference for the projected zeroth moment.

    The same reference `eta0_amplification` uses, reproduced here so the two
    studies cannot drift apart on the definition of "exact".

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences.
    Lia : numpy.ndarray
        The ``(aux, W occ, W vir)`` integral array.

    Returns
    -------
    numpy.ndarray
        RI-projected zeroth moment, shape ``(naux, nov)``.
    """
    W = (Lia * np.sqrt(d)[None]).T
    mtilde = np.diag(d**2) + 4.0 * W @ W.T
    evals, evecs = np.linalg.eigh(mtilde)
    return (np.sqrt(d)[:, None] * (evecs @ ((evals**-0.5)[:, None] * (evecs.T @ W)))).T


def relative_error(predicted, reference):
    """Relative Frobenius error per moment order.

    Parameters
    ----------
    predicted : numpy.ndarray
        Moments indexed by order along the leading axis.
    reference : numpy.ndarray
        Reference moments, same shape.

    Returns
    -------
    numpy.ndarray
        One relative error per order.  A zero reference order gives zero when the
        difference is also zero and infinity otherwise, rather than a silent nan.
    """
    errors = []
    for order in range(reference.shape[0]):
        difference = float(np.linalg.norm(np.ravel(predicted[order] - reference[order])))
        scale = float(np.linalg.norm(np.ravel(reference[order])))
        errors.append(difference / scale if scale > 0.0 else (0.0 if difference == 0.0 else np.inf))
    return np.array(errors)


def run_system(name, atom, basis, xc, nmom_max, compression):
    """Measure the production self-energy moment error for one system.

    Parameters
    ----------
    name : str
        System identifier for the report.
    atom : str
        Geometry.
    basis : str
        Basis set.
    xc : str
        Exchange-correlation functional, or `"hf"`.
    nmom_max : int
        Maximum moment order.
    compression : str
        Auxiliary compression sectors, or `""` for none.
    """
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    if xc == "hf":
        mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    else:
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    gw = GW(mf)
    gw.compression = compression
    with contextlib.redirect_stdout(io.StringIO()):
        integrals = gw.ao2mo()
        rpa = dRPA(gw, nmom_max, integrals)
        rpa._build_d()

        # Reference: exact eta0, then the production pipeline unchanged.
        reference0 = dense_eta0(rpa.d, integrals.Lia)
        dd_reference = rpa.build_dd_moments(integral=reference0)
        hole_reference, particle_reference = rpa.build_se_moments(moments_dd=dd_reference)

        # Production: whatever the shipped defaults select, with nothing overridden.
        production0 = rpa.build_zeroth_dd_moment()
        dd_production = rpa.build_dd_moments(integral=production0)
        hole_production, particle_production = rpa.build_se_moments(moments_dd=dd_production)

    scale = np.max(np.abs(reference0))
    eta0_error = float(np.max(np.abs(production0 - reference0)) / scale)

    print(
        f"\n== {name}  (nov = {rpa.d.size}, naux = {integrals.Lia.shape[0]}, "
        f"eta0_method = {gw.eta0_method!r}, eta0_tol = {gw.eta0_tol:.0e}, "
        f"compression = {compression or 'off'}) =="
    )
    print(f"  eta0 max relative error against the dense reference: {eta0_error:.2e}")
    for label, predicted, reference in (
        ("dd", dd_production, dd_reference),
        ("hole", hole_production, hole_reference),
        ("particle", particle_production, particle_reference),
    ):
        errors = relative_error(np.asarray(predicted), np.asarray(reference))
        print(
            f"  {label:9s} "
            + " ".join(f"n{n}={e:.1e}" for n, e in enumerate(errors))
            + f"   max {np.max(errors):.2e}"
        )


def main():
    """Run the requested sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--systems",
        nargs="+",
        default=sorted(SYSTEMS),
        choices=sorted(SYSTEMS),
        help="systems to measure",
    )
    parser.add_argument("--nmom-max", type=int, default=7, help="maximum moment order")
    parser.add_argument(
        "--compression",
        default="",
        help="auxiliary compression sectors, e.g. 'ia'; default is off",
    )
    args = parser.parse_args()

    print(f"momentGW imported from: {momentGW.__file__}")
    for name in args.systems:
        atom, basis, xc = SYSTEMS[name]
        run_system(name, atom, basis, xc, args.nmom_max, args.compression)


if __name__ == "__main__":
    main()
