"""Validate the dRPA construction against an external code and an explicit-pole oracle.

The scientific validation track asks for "comparisons against PySCF TD-dRPA or an
explicit-pole oracle on small systems". This does both, because they answer different
questions and neither is sufficient alone:

**PySCF TD-dRPA** validates the *physics*. `pyscf.tdscf.dRPA` is an independent
implementation with its own conventions, so agreement means the matrix this code builds
really is the direct RPA problem - the spin factor, the absence of exchange, and the
particle-hole metric are all right. What it cannot check is the moments, which PySCF does
not compute.

**The explicit-pole oracle** validates the *recursion*. With `D` the particle-hole energy
differences, `V` the RI integrals, and

    Mtilde = D^1/2 (D + 4 V^T V) D^1/2 = S Omega^2 S^T

the recursion's own step is `eta_n = eta_(n-2) M` with `M = D^2 + 4 V^T V D`, and `M` is
similar to `Mtilde` under `D^1/2`. Substituting collapses every moment to a single sum over
the explicit excitations:

    eta_n = V D^1/2 S Omega^(n-1) S^T D^1/2

which is a closed form in the poles rather than an iteration, so it shares no arithmetic
with the recursion beyond the integrals both start from. Two checks come free: `n = 1` must
collapse to `V D` because `S` is orthogonal, and `n = 0` to the zeroth moment Milestone 2
certified independently.

The oracle is dense in the particle-hole space and so is limited to small systems, which is
what the roadmap asks of it.

A study, not part of the recorded baseline set: it is re-run when the claim it supports is
in question, not by `baseline.check`. The fast case is also pinned in
`tests/test_rpa_oracle.py`, so a regression is caught by the suite rather than only here.

Run from the repository root so the intended tree is imported::

    python -m baseline.studies.rpa_oracle
    python -m baseline.studies.rpa_oracle --systems water_hf --orders 1 5 11
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
from pyscf import dft, gto, tdscf

import momentGW
from momentGW.gw import GW
from momentGW.rpa import dRPA

#: Small enough that the particle-hole space can be diagonalised densely.
SYSTEMS = {
    "h2_hf": ("H 0 0 0; H 0 0 0.74", "6-31g", "hf"),
    "water_hf": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "sto3g", "hf"),
    "water_pbe": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "sto3g", "pbe"),
    "lih_hf": ("Li 0 0 0; H 0 0 1.595", "sto3g", "hf"),
}

#: Orders compared. Swept well past what the baseline records, because a maintained
#: comparison earns its keep where the recursion has had room to drift.
ORDERS = (0, 1, 3, 7, 11, 15)


def build(name, nmom_max):
    """Build the moments and the pieces the oracle needs.

    Returns
    -------
    tuple
        The mean field, the recursion's moments, the energy differences and the integrals.
    """
    atom, basis, xc = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    # `RKS` with `xc="hf"` rather than `RHF`, for both references: it is the same
    # Hartree-Fock mean field, and `pyscf.tdscf.dRPA` is only reachable from a KS object.
    mf = dft.RKS(mol, xc=xc).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    gw = GW(mf, polarizability="drpa")
    with contextlib.redirect_stdout(io.StringIO()):
        integrals = gw.ao2mo()
        rpa = dRPA(gw, nmom_max, integrals)
        rpa._build_d()
        moments = np.asarray(rpa.build_dd_moments())
    return mf, moments, rpa.d, integrals.Lia


def rpa_spectrum(d, integrals):
    """Diagonalise the dRPA problem densely.

    Returns
    -------
    tuple
        Excitation energies and the eigenvectors of `Mtilde`.
    """
    sqrt_d = np.sqrt(d)
    mtilde = np.diag(d**2) + 4.0 * (sqrt_d[:, None] * (integrals.T @ integrals) * sqrt_d[None, :])
    eigenvalues, vectors = np.linalg.eigh(mtilde)
    return np.sqrt(np.clip(eigenvalues, 0.0, None)), vectors


def oracle_moment(order, d, integrals, omega, vectors):
    """Build the density-density moment as a sum over explicit RPA excitations."""
    sqrt_d = np.sqrt(d)
    left = (integrals * sqrt_d[None]) @ vectors
    return (left * omega[None] ** (order - 1)) @ (vectors.T * sqrt_d[None])


def compare_to_pyscf(mf, omega):
    """Compare the excitation energies against `pyscf.tdscf.dRPA`."""
    td = tdscf.dRPA(mf)
    td.nstates = omega.size
    with contextlib.redirect_stdout(io.StringIO()):
        td.kernel()
    theirs = np.sort(np.asarray(td.e).real)
    count = min(omega.size, theirs.size)
    return count, float(np.abs(omega[:count] - theirs[:count]).max())


def main():
    """Run both comparisons and print their tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=sorted(SYSTEMS), choices=sorted(SYSTEMS))
    parser.add_argument("--orders", nargs="+", type=int, default=list(ORDERS))
    args = parser.parse_args()

    print(f"momentGW: {momentGW.__file__}")
    nmom_max = max(args.orders)

    print("\n1. excitation energies against pyscf.tdscf.dRPA - an independent implementation")
    print(f"{'system':>12} {'nov':>5} {'states':>7} {'max |diff| / eV':>17}")
    for name in args.systems:
        mf, _, d, integrals = build(name, nmom_max)
        omega, _ = rpa_spectrum(d, integrals)
        count, difference = compare_to_pyscf(mf, omega)
        print(f"{name:>12} {d.size:>5} {count:>7} {difference * 27.211386245988:17.3e}")

    print("\n2. density-density moments against the explicit-pole oracle")
    print(f"{'system':>12} " + " ".join(f"{'n=' + str(n):>10}" for n in args.orders))
    for name in args.systems:
        _, moments, d, integrals = build(name, nmom_max)
        omega, vectors = rpa_spectrum(d, integrals)
        cells = []
        for order in args.orders:
            reference = oracle_moment(order, d, integrals, omega, vectors)
            got = np.asarray(moments[order])
            norm = np.linalg.norm(reference)
            cells.append(float(np.linalg.norm(got - reference) / norm) if norm > 0 else np.nan)
        print(f"{name:>12} " + " ".join(f"{value:10.2e}" for value in cells))

    print(
        "\n   n=1 is `V D` exactly, and n=0 is the zeroth moment Milestone 2 certified,\n"
        "   so those two columns check the oracle as much as the recursion."
    )


if __name__ == "__main__":
    raise SystemExit(main())
