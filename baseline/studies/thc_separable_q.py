"""Decide how a THC dRPA should treat the particle-hole denominator, by measurement.

The HHT route in `momentGW.rpa` reduced the zeroth dRPA moment to a pole sum whose only
system-sized object is, per pole shift ``s``,

    Q^(1)(s) = V^T [ D / (D^2 + s) ] V ,          V = the RI or THC vertex,

and the higher density-density moments need the same object at higher powers,

    Q^(m+1)(s) = V^T [ D^(m+1) / (D^2 + s) ] V .

``D`` is not separable in the occupied and virtual indices, so a THC vertex cannot contract
against it directly.  Every attempt in `~/Desktop/Old THC` handled that by splitting the
particle-hole pairs into `D < z` and `D > z` regions at each quadrature point, using a
truncated series on one side and a Laplace/cosine integral on the other, with correction
terms across the boundary.  The splitting is where that work lost its accuracy.

The route measured here removes the splitting.  Write

    D / (D^2 + s)  ~  sum_k c_k(s) exp(-t_k D) ,
    exp(-t_k D) = exp(+t_k (e_i - mu)) exp(-t_k (e_a - mu)) ,

so each grid point contributes a Hadamard product of two auxiliary-space matrices and the
expensive factor depends on ``t_k`` alone -- not on the shift, and not on the moment order.
The grid is built once and every pole is a linear combination of it.  `exponential_grid`
measures how large that grid has to be, on the true set of particle-hole gaps.

That leaves the higher powers.  Three ways to reach them, and `route_comparison` measures
all three against an exact evaluation of the same pole sum:

**A, polynomial division.**  ``D^(2r+1)/(D^2+s) = sum_k (-s)^k D^(2r-1-2k) + (-s)^r D/(D^2+s)``
is exact per pair, and every term but the last is a shift-independent ``Dcal^(j) = V^T D^j V``
that THC dTDA already builds.  Two auxiliary-space builds per grid point, plus one set of
``Dcal`` for the whole calculation -- by far the cheapest.

**B, binomial in the orbital energies.**  ``D^m = sum_p C(m,p) (e_a - mu)^(m-p) (mu - e_i)^p``
is exact and separable term by term, so ``D^m exp(-tD)`` is reachable at ``(m+1)`` builds per
grid point.  This is the structure `get_D_exp_list` in the old code already had.

**C, fit the target directly.**  Fit ``D^(m+1)/(D^2+s)`` on the same grid and keep two builds
per grid point for every order.

Measured 2026-08-25 on this machine (macOS, Accelerate, PySCF 2.14.0, `mgw-monomial`):

* The shared grid reaches 1e-12 relative on ``Q^(1)`` for every pole at ``n_t = 60``.
* **A degrades by about 1.5 decades per moment order** -- 2e-5 by order 8 on benzene -- so
  it is not usable past the low orders, however cheap it is.
* **B holds 1e-12 at every order and does not degrade with order.**  It is the design.
* **C saturates**: adding grid points stops helping, because a growing function cannot be
  represented by decaying exponentials without cancellation that float64 will not carry.
  Rejected on measurement.

So the old kernel was right and the machinery around it was not: keep the binomial, replace
the region splitting and the oscillatory inner quadrature with one fitted shared grid.

A study, not part of the recorded baseline set: it is re-run when the claims it supports are
in question, not by `baseline.check`.  Run from the repository root so the intended tree is
imported, and read the printed `momentGW.__file__` before believing a comparison::

    python -m baseline.studies.thc_separable_q
"""

import numpy as np
from pyscf import df, gto, scf
from pyscf.lib import unpack_tril
from scipy.special import binom

import momentGW
from momentGW import eta0

#: Extra system beyond the baseline set: large enough for the particle-hole dimension to
#: dominate the auxiliary one, which is where the cost model turns over.
BENZENE = (
    "C 0.000000 1.396000 0.000000; C 1.209000 0.698000 0.000000; "
    "C 1.209000 -0.698000 0.000000; C 0.000000 -1.396000 0.000000; "
    "C -1.209000 -0.698000 0.000000; C -1.209000 0.698000 0.000000; "
    "H 0.000000 2.479000 0.000000; H 2.147000 1.240000 0.000000; "
    "H 2.147000 -1.240000 0.000000; H 0.000000 -2.479000 0.000000; "
    "H -2.147000 -1.240000 0.000000; H -2.147000 1.240000 0.000000"
)

SYSTEMS = {
    "water": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "ccpvdz", "weigend"),
    "ozone": (
        "O 0.000000 0.000000 0.000000; O 1.086900 0.000000 0.660000; O -1.086900 0.000000 0.660000",
        "ccpvdz",
        "weigend",
    ),
    "benzene": (BENZENE, "ccpvdz", "weigend"),
}

#: Highest density-density moment order reported.
NMOM_MAX = 9

#: Requested eta0 tolerance. The production default in `momentGW.base`, so the pole count
#: here is the one a real calculation would carry.
ETA0_TOL = 1e-14

#: Grid sizes swept.
GRID_SIZES = (20, 30, 40, 48, 60, 80)


class Case:
    """A system reduced to the objects the measurements need.

    The vertex is the exact RI one, not a THC factorisation: what is under test is the
    treatment of the particle-hole denominator, so the THC fitting error is deliberately
    kept out of the comparison.

    Parameters
    ----------
    atom : str
        Geometry in PySCF's format.
    basis : str
        Orbital basis.
    auxbasis : str
        Density-fitting auxiliary basis.
    """

    def __init__(self, atom, basis, auxbasis):
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
        mf.conv_tol = 1e-11
        mf.kernel()

        nocc = mol.nelectron // 2
        energy = mf.mo_energy

        # Referencing the orbital energies to the chemical potential is not cosmetic: it
        # is what keeps both separable exponential factors bounded by one, and what makes
        # the binomial expansion of D^m a sum of like-signed terms.
        self.mu = 0.5 * (energy[nocc - 1] + energy[nocc])
        self.e_occ = energy[:nocc] - self.mu
        self.e_vir = energy[nocc:] - self.mu
        self.d = (self.e_vir[None, :] - self.e_occ[:, None]).ravel()

        with_df = df.DF(mol, auxbasis=auxbasis)
        with_df.build()
        Lpq = unpack_tril(np.vstack(list(with_df.loop())))
        self.Lia = np.einsum(
            "Lpq,pi,qa->Lia", Lpq, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
        ).reshape(Lpq.shape[0], -1)

        self.nao = mol.nao_nr()
        self.naux, self.nov = self.Lia.shape

        coupling = 4.0 * float(np.sum(self.d * np.sum(self.Lia * self.Lia, axis=0)))
        self.lmin, self.lmax = eta0.certified_interval(self.d, coupling)
        self.shifts, self.weights, self.delta, _ = eta0.select_poles(self.lmin, self.lmax, ETA0_TOL)

    def contract(self, f):
        """Contract a scalar function of the gaps into the auxiliary space.

        Parameters
        ----------
        f : numpy.ndarray
            Values on the particle-hole gaps, shape ``(nov,)``.

        Returns
        -------
        matrix : numpy.ndarray
            ``V^T f(D) V``, shape ``(naux, naux)``.
        """
        return (self.Lia * f[None]) @ self.Lia.T

    def grid(self, n_t):
        """Get the shared exponential grid and its values on the gaps.

        Parameters
        ----------
        n_t : int
            Number of grid points.

        Returns
        -------
        t : numpy.ndarray
            Grid points.
        basis : numpy.ndarray
            ``exp(-t_k D)``, shape ``(nov, n_t)``.
        """
        t = np.geomspace(1e-2 / self.d.max(), 30.0 / self.d.min(), n_t)
        return t, np.exp(-np.outer(self.d, t))

    def fit(self, basis, target):
        """Fit a scalar function on the grid with relative-error weighting.

        Weighting by ``1/target`` makes the residual relative, which is what the moment
        error responds to; fitting on the true gaps rather than an enclosing interval
        makes the reported error the realised one for this system.

        Parameters
        ----------
        basis : numpy.ndarray
            ``exp(-t_k D)``, shape ``(nov, n_t)``.
        target : numpy.ndarray
            Values to reproduce, shape ``(nov,)``.

        Returns
        -------
        coeffs : numpy.ndarray
            Fitted coefficients, shape ``(n_t,)``.
        """
        weight = 1.0 / target
        coeffs, *_ = np.linalg.lstsq(basis * weight[:, None], target * weight, rcond=None)
        return coeffs


def exponential_grid(case, sizes=GRID_SIZES):
    """Measure a shared exponential grid against every pole at once.

    Parameters
    ----------
    case : Case
        The system.
    sizes : tuple of int, optional
        Grid sizes to sweep.  Default value is `GRID_SIZES`.

    Returns
    -------
    rows : list of tuple
        ``(size, worst relative error on g, worst relative error on Q^(1),
        worst coefficient amplification)`` over the pole set.
    """

    rows = []
    for n_t in sizes:
        _, basis = case.grid(n_t)
        worst_g = worst_q = worst_amp = 0.0
        for s in case.shifts:
            g = case.d / (case.d**2 + s)
            coeffs = case.fit(basis, g)
            fit = basis @ coeffs
            worst_g = max(worst_g, float(np.max(np.abs(fit - g) / g)))
            worst_amp = max(worst_amp, float(np.sum(np.abs(coeffs)) / np.max(np.abs(g))))
            exact = case.contract(g)
            worst_q = max(
                worst_q,
                float(np.max(np.abs(case.contract(fit) - exact)) / np.max(np.abs(exact))),
            )
        rows.append((n_t, worst_g, worst_q, worst_amp))

    return rows


def exact_moments(case, nmom_max=NMOM_MAX):
    """Evaluate the pole sum with no separable approximation anywhere.

    Parameters
    ----------
    case : Case
        The system.
    nmom_max : int, optional
        Highest moment order.  Default value is `NMOM_MAX`.

    Returns
    -------
    moments : numpy.ndarray
        ``N^(m)`` for even ``m``, shape ``(nmom_max + 1, naux, naux)``; odd orders are zero
        because they are reached by recursion rather than by the integral.
    """

    moments = np.zeros((nmom_max + 1, case.naux, case.naux))
    eye = np.eye(case.naux)
    for s, w in zip(case.shifts, case.weights):
        inv = np.linalg.inv(eye + 4.0 * case.contract(case.d / (case.d**2 + s)))
        for m in range(0, nmom_max + 1, 2):
            moments[m] += w * (inv @ case.contract(case.d ** (m + 1) / (case.d**2 + s)))
    return moments * (2.0 / np.pi)


def route_comparison(case, n_t, nmom_max=NMOM_MAX):
    """Measure the three routes to the higher powers against the exact pole sum.

    Parameters
    ----------
    case : Case
        The system.
    n_t : int
        Size of the shared exponential grid.
    nmom_max : int, optional
        Highest moment order.  Default value is `NMOM_MAX`.

    Returns
    -------
    rows : list of tuple
        ``(m, route A error, route B error, route C error)``, relative to the exact
        pole sum in the max norm.
    """

    exact = exact_moments(case, nmom_max)
    _, basis = case.grid(n_t)
    eye = np.eye(case.naux)
    d = case.d

    dcal = {j: case.contract(d**j) for j in range(1, nmom_max + 2, 2)}

    routes = {key: np.zeros_like(exact) for key in "ABC"}
    for s, w in zip(case.shifts, case.weights):
        g = d / (d**2 + s)
        gfit = basis @ case.fit(basis, g)
        q1 = case.contract(gfit)
        inv = np.linalg.inv(eye + 4.0 * q1)

        for m in range(0, nmom_max + 1, 2):
            r = m // 2
            terms = [((-s) ** k) * dcal[m - 1 - 2 * k] for k in range(r)]
            q_a = ((-s) ** r) * q1 + (sum(terms) if terms else 0.0)

            # The binomial route as it would actually be evaluated: D^m never formed,
            # only like-signed products of the shifted orbital energies.
            q_b = np.zeros_like(q1)
            for p in range(m + 1):
                power = (
                    binom(m, p)
                    * np.power.outer((-case.e_occ), p)[:, None]
                    * np.power.outer(case.e_vir, m - p)[None, :]
                ).ravel()
                q_b += case.contract(power * gfit)

            target = d ** (m + 1) / (d**2 + s)
            q_c = case.contract(basis @ case.fit(basis, target))

            for key, q in zip("ABC", (q_a, q_b, q_c)):
                routes[key][m] += w * (inv @ q)

    rows = []
    for m in range(0, nmom_max + 1, 2):
        scale = np.max(np.abs(exact[m]))
        rows.append(
            (m, *[np.max(np.abs(routes[k][m] * (2 / np.pi) - exact[m])) / scale for k in "ABC"])
        )
    return rows


def separable_cost(nao, n_t, rank_ratio, n_shifts, nmom_max=NMOM_MAX):
    """Leading multiplicative cost of the separable route, in units of ``nao**3``.

    Three terms, all cubic in the orbital count:

    * the grid build -- ``nmom_max + 1`` weighted outer products of side `naux_thc` over
      the occupied and the virtual index, at every grid point.  This is the term that
      makes the route worth having and the one the rank squares into;
    * combining them binomially into ``D^m exp(-t D)``, once per grid point and order;
    * projecting onto the shifts, and the auxiliary-space moment recursion, which is the
      only cubic-in-`naux_thc` term.

    Parameters
    ----------
    nao : int
        Number of orbitals.
    n_t : int
        Size of the shared exponential grid.
    rank_ratio : float
        THC rank as a multiple of `nao`.
    n_shifts : int
        Number of pole shifts.
    nmom_max : int, optional
        Highest moment order.  Default value is `NMOM_MAX`.

    Returns
    -------
    cost : float
        Predicted multiply-accumulates.
    """

    naux = rank_ratio * nao
    grid = n_t * (nmom_max + 1) * naux**2 * nao
    combine = n_t * ((nmom_max + 1) ** 2 / 4.0) * naux**2
    project = n_shifts * ((nmom_max + 1) / 2.0) * n_t * naux**2
    recursion = ((nmom_max + 1) ** 2 / 2.0) * naux**3

    return grid + combine + project + recursion


def ri_cost(naux, nov, n_shifts, nmom_max=NMOM_MAX):
    """Leading multiplicative cost of the density-fitted route it has to beat.

    Per pole, `momentGW.rpa.dRPA._hht_apply` forms one ``(naux, nov)`` Gram and solves
    with `nov` right-hand sides -- three passes of ``naux**2 nov``.  Per moment order,
    `build_dd_moments` takes two more.

    Parameters
    ----------
    naux : int
        Density-fitting auxiliary dimension.
    nov : int
        Number of particle-hole pairs.
    n_shifts : int
        Number of pole shifts.
    nmom_max : int, optional
        Highest moment order.  Default value is `NMOM_MAX`.

    Returns
    -------
    cost : float
        Predicted multiply-accumulates.
    """

    return (3.0 * n_shifts + 2.0 * nmom_max) * naux**2 * nov


def crossover(case, n_t, rank_ratio, nmom_max=NMOM_MAX):
    """Get the number of orbitals at which the separable route becomes cheaper.

    Holds the auxiliary-to-orbital and pair-to-orbital ratios of `case` fixed.  Both
    routes are cubic in `nao` except the RI Gram, which is quartic, so the crossover is
    where the cubic constants balance the quartic one.

    Parameters
    ----------
    case : Case
        The system, used for its shape ratios.
    n_t : int
        Size of the shared exponential grid.
    rank_ratio : float
        THC rank as a multiple of the number of orbitals.
    nmom_max : int, optional
        Highest moment order.  Default value is `NMOM_MAX`.

    Returns
    -------
    nao : float
        Orbital count at the crossover.
    """

    n_shifts = case.shifts.size
    aux_per_orb = case.naux / case.nao
    pairs_per_orb_sq = case.nov / case.nao**2

    # Cubic coefficient of the separable route, with the sub-leading terms dropped: they
    # carry no factor of `nao` and vanish against the others at the sizes in question.
    separable = n_t * (nmom_max + 1) * rank_ratio**2 + ((nmom_max + 1) ** 2 / 2.0) * (rank_ratio**3)
    ri = (3.0 * n_shifts + 2.0 * nmom_max) * aux_per_orb**2 * pairs_per_orb_sq

    return separable / ri


def main():
    """Run the measurements over the system set and print the tables."""

    print(f"momentGW: {momentGW.__file__}")

    for name, (atom, basis, auxbasis) in SYSTEMS.items():
        case = Case(atom, basis, auxbasis)
        print(f"\n=== {name} / {basis} ===")
        print(
            f"nao {case.nao}  naux(RI) {case.naux}  nov {case.nov}   "
            f"D in [{case.d.min():.4f}, {case.d.max():.4f}] "
            f"(ratio {case.d.max() / case.d.min():.1f})"
        )
        print(
            f"certified interval [{case.lmin:.4e}, {case.lmax:.4e}] "
            f"(cond {case.lmax / case.lmin:.3e}); "
            f"{case.shifts.size} poles at scalar error {case.delta:.3e}"
        )

        print("\n  shared exponential grid, worst over the pole set")
        print("    n_t   rel. err g   rel. err Q^(1)   amplification")
        for n_t, err_g, err_q, amp in exponential_grid(case):
            print(f"   {n_t:4d}   {err_g:10.3e}   {err_q:14.3e}   {amp:13.3e}")

        for n_t in (40, 60):
            print(f"\n  route comparison on N^(m), n_t = {n_t}")
            print("     m   A: division   B: binomial   C: direct fit")
            for m, err_a, err_b, err_c in route_comparison(case, n_t):
                print(f"    {m:2d}   {err_a:11.3e}   {err_b:11.3e}   {err_c:13.3e}")

        print("\n  predicted flop ratio (route B / RI), and crossover orbital count")
        print("    THC rank      n_t=24            n_t=33            n_t=42")
        for rank_ratio in (5.0, 8.0, 10.0, 12.0):
            cells = []
            for n_t in (24, 33, 42):
                ratio = separable_cost(case.nao, n_t, rank_ratio, case.shifts.size) / ri_cost(
                    case.naux, case.nov, case.shifts.size
                )
                cells.append(f"{ratio:6.1f} (nao {crossover(case, n_t, rank_ratio):5.0f})")
            print(f"   {rank_ratio:5.1f} x nao   " + "  ".join(cells))


if __name__ == "__main__":
    main()
