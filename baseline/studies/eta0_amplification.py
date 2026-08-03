"""Measure how an eta0 perturbation amplifies through the moment recurrences.

For each system, the reference eta0 is a dense eigendecomposition of Mtilde.
Perturbed variants are HHT rational approximations at fixed pole counts, so
each carries a *known* measured scalar error delta.  Both are pushed through
`build_dd_moments`, `build_se_moments`, and `solve_dyson`, and the per-order
relative Frobenius differences and frontier QP shifts are reported against
delta.

This is the measurement the ROADMAP 2.4 `eta0_tol` default rests on
(measured 2026-07-31): the dd-moment recurrence does not amplify an
eta0 perturbation through order 6, frontier QP energies move by roughly
30-300x the scalar error in eV, and below a scalar error of ~1e-13 the
float64 kernel arithmetic floor takes over.  A study, not part of the
recorded baseline set: it is re-run when the claim it supports is in
question, not by `baseline.check`.

Run from the repository root so the intended tree is imported, and read the
printed `momentGW.__file__` before believing a comparison::

    python -m baseline.studies.eta0_amplification
"""

import contextlib
import io
import sys

import numpy as np
from pyscf import dft, gto, scf

import momentGW
from momentGW import eta0 as eta0_lib
from momentGW.gw import GW
from momentGW.rpa import dRPA

HARTREE2EV = 27.211386245988

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

NMOM_MAX = 7
POLE_COUNTS = [3, 4, 6, 8, 10, 12, 16, 20, 24]


def dense_eta0(d, Lia):
    """Dense eigendecomposition reference for the projected zeroth moment.

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences.
    Lia : numpy.ndarray
        The ``(aux, W occ, W vir)`` integral array.

    Returns
    -------
    eta0 : numpy.ndarray
        RI-projected zeroth moment, shape ``(naux, nov)``.
    """
    W = (Lia * np.sqrt(d)[None]).T
    mtilde = np.diag(d**2) + 4.0 * W @ W.T
    evals, evecs = np.linalg.eigh(mtilde)
    return (np.sqrt(d)[:, None] * (evecs @ ((evals**-0.5)[:, None] * (evecs.T @ W)))).T


def frontier(gw, gf):
    """Frontier HOMO/LUMO energies from largest-overlap assignment.

    Parameters
    ----------
    gw : momentGW.gw.GW
        GW object.
    gf : dyson.Lehmann
        Green's function.

    Returns
    -------
    homo : float
        Frontier occupied energy.
    lumo : float
        Frontier virtual energy.
    """
    qp = gw._gf_to_mo_energy(gf)
    homo = qp[gw.mo_occ > 0].max()
    lumo = qp[gw.mo_occ == 0].min()
    return homo, lumo


def run_system(name, atom, basis, xc):
    """Measure amplification for one system.

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
    with contextlib.redirect_stdout(io.StringIO()):
        integrals = gw.ao2mo()
        rpa = dRPA(gw, NMOM_MAX, integrals)
        rpa._build_d()
        d = rpa.d
        Lia = integrals.Lia

        se_static = gw.build_se_static(integrals)

        # Reference pipeline from the dense eta0
        ref0 = dense_eta0(d, Lia)
        mom_ref = rpa.build_dd_moments(integral=ref0)
        th_ref, tp_ref = rpa.build_se_moments(moments_dd=mom_ref)
        gf_ref, _ = gw.solve_dyson(th_ref, tp_ref, se_static, integrals=integrals)
        homo_ref, lumo_ref = frontier(gw, gf_ref)

        # Certified interval, exactly as the kernel builds it (serial)
        sqrt_d = np.sqrt(d)
        w_frob_sq = float(np.sum(d * np.sum(Lia * Lia, axis=0)))
        w_one = float(np.max(np.sum(np.abs(Lia) * sqrt_d[None], axis=1)))
        w_inf = float(np.max(sqrt_d * np.sum(np.abs(Lia), axis=0)))
        lmin, lmax = eta0_lib.certified_interval(d, 4.0 * min(w_frob_sq, w_one * w_inf))

    print(
        f"\n== {name}  (nov = {d.size}, naux = {Lia.shape[0]}, kappa_cert = {lmax / lmin:.3e}) =="
    )
    print(
        f"  reference HOMO = {homo_ref * HARTREE2EV:+.6f} eV, "
        f"LUMO = {lumo_ref * HARTREE2EV:+.6f} eV"
    )
    header = "  {:>3s} {:>9s} {:>9s}".format("N", "delta", "d(eta0)")
    header += "".join(f"   mom{n}" for n in range(0, NMOM_MAX + 1, 2))
    header += "     dSE(h)    dSE(p)   dHOMO(eV)   dLUMO(eV)"
    print(header)

    ref_scale = np.max(np.abs(ref0))
    for n_poles in POLE_COUNTS:
        with contextlib.redirect_stdout(io.StringIO()):
            shifts, weights = eta0_lib.hht_coefficients(lmin, lmax, n_poles)
            delta = eta0_lib.scalar_error(shifts, weights, lmin, lmax)
            pert0, _ = rpa._hht_apply(shifts, weights, d, Lia)
            d_eta0 = np.max(np.abs(pert0 - ref0)) / ref_scale

            mom_p = rpa.build_dd_moments(integral=pert0)
            th_p, tp_p = rpa.build_se_moments(moments_dd=mom_p)
            gf_p, _ = gw.solve_dyson(th_p, tp_p, se_static, integrals=integrals)
            homo_p, lumo_p = frontier(gw, gf_p)

        row = f"  {n_poles:>3d} {delta:9.2e} {d_eta0:9.2e}"
        for n in range(0, NMOM_MAX + 1, 2):
            rel = np.linalg.norm(mom_p[n] - mom_ref[n]) / max(np.linalg.norm(mom_ref[n]), 1e-300)
            row += f" {rel:7.1e}"
        th_rel = max(
            np.linalg.norm(th_p[n] - th_ref[n]) / max(np.linalg.norm(th_ref[n]), 1e-300)
            for n in range(NMOM_MAX + 1)
        )
        tp_rel = max(
            np.linalg.norm(tp_p[n] - tp_ref[n]) / max(np.linalg.norm(tp_ref[n]), 1e-300)
            for n in range(NMOM_MAX + 1)
        )
        row += f"  {th_rel:9.2e} {tp_rel:9.2e}"
        row += f" {abs(homo_p - homo_ref) * HARTREE2EV:11.3e}"
        row += f" {abs(lumo_p - lumo_ref) * HARTREE2EV:11.3e}"
        print(row)
        sys.stdout.flush()


if __name__ == "__main__":
    print(f"momentGW imported from: {momentGW.__file__}")
    for name, (atom, basis, xc) in SYSTEMS.items():
        run_system(name, atom, basis, xc)
