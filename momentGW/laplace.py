r"""Separable representation of the particle-hole resolvent.

`momentGW.eta0` reduced the zeroth dRPA moment to a sum over certified poles, whose only
system-sized object is, per pole shift :math:`s`,

.. math::
    Q^{(1)}(s) = V^T \frac{D}{D^2 + s} V ,

with :math:`D_{ia} = \epsilon_a - \epsilon_i` and :math:`V` the vertex.  The higher
density-density moments need the same object at higher powers of :math:`D`.  A
density-fitted vertex can contract against that directly; a tensor-hypercontracted one
cannot, because :math:`X_{i\mu} X_{a\mu}` separates the occupied and virtual indices and
:math:`D` couples them.

This module supplies the missing separability.  On a grid of positive exponents
:math:`t_k`,

.. math::
    \frac{D}{D^2 + s} \approx \sum_k c_k(s) e^{-t_k D} ,
    \qquad
    e^{-t_k D} = e^{t_k (\epsilon_i - \mu)} e^{-t_k (\epsilon_a - \mu)} ,

so each term factorises, and **the grid does not depend on the pole or on the moment
order** -- only the coefficients do.  Whatever a solver builds per grid point is therefore
built once and reused by every pole and every order.

Referencing the orbital energies to the chemical potential :math:`\mu` is not cosmetic.
It puts both exponential factors in :math:`[0, 1]`, so neither overflows however wide the
orbital range, and it makes the binomial expansion of :math:`D^m` a sum of like-signed
terms.

What the measured residual certifies
------------------------------------
The fit is made against the *true* particle-hole gaps with relative-error weighting, so
the reported residual :math:`\varepsilon = \max_{ia} |\delta(D_{ia})| / g(D_{ia})` is
realised rather than a bound over an enclosing interval.  That number then transfers
directly to the operator, at every moment order:

.. math::
    \Delta Q^{(m+1)} = V \,\mathrm{diag}(D^m \delta)\, V^T ,
    \qquad
    |u^T \Delta Q^{(m+1)} u| \le \varepsilon \sum_{ia} D^m_{ia} g(D_{ia}) (V^T u)_{ia}^2
    = \varepsilon \, u^T Q^{(m+1)} u ,

because :math:`D^m g(D) > 0` makes :math:`Q^{(m+1)}` positive semidefinite.  Hence
:math:`\|\Delta Q^{(m+1)}\|_2 \le \varepsilon \|Q^{(m+1)}\|_2` for every :math:`m` -- one
scalar residual certifies the whole moment set, and it does not degrade with order.

Why the grid is selected rather than laid out
---------------------------------------------
The exponential basis on a real gap distribution has a numerical rank near 30 in float64,
and a geometric grid reaches it only by oversampling: measured on ozone, the least-squares
effective rank saturates at 27 to 34 whether the grid holds 48 points or 120.  Choosing
the points by pivoted QR on the basis instead reaches the same floor with far fewer:

===========  =====================  =====================
`n_points`   geometric (benzene)    pivot-selected
===========  =====================  =====================
36           4.4e-08                1.4e-10
42           4.5e-09                **1.0e-12**
60           1.9e-11                2.9e-12
===========  =====================  =====================

Since a solver's cost is linear in the number of grid points, that is a direct saving.
The floor itself is `TOL_FLOOR`: below a few times 1e-13 the fit is limited by float64,
not by the grid, and more points do not help.
"""

import numpy as np
import scipy.linalg

from momentGW import logging

#: Smallest residual the float64 least squares can honour.  Measured across the baseline
#: systems, the best achievable relative residual is 5e-13 to 6e-12 and does not improve
#: with more grid points; asking for less would be asking for a number the arithmetic
#: cannot deliver.
TOL_FLOOR = 1e-12

#: Candidate exponents span ``[LO / max(D), HI / min(D)]``.  The lower end resolves the
#: largest gaps, where the integrand decays fastest; the upper end has to reach far enough
#: past ``1 / min(D)`` for the slowest-decaying pair to be represented.  The selection
#: prunes whatever is redundant, so these are deliberately generous.
CANDIDATE_LO = 1e-3
CANDIDATE_HI = 60.0

#: Size of the candidate set the selection prunes.  Only its resolution matters -- the
#: pivoted QR runs on an ``(nov, N_CANDIDATE)`` matrix once per calculation.
N_CANDIDATE = 400

#: Singular-value cutoff for the least squares.  The default `numpy` cutoff truncates
#: directions the fit still needs: on ozone it leaves 3.6e-11 where 1e-15 leaves 7.2e-12,
#: at an amplification cost of under a factor of two.
RCOND = 1e-15

#: Largest grid the search may reach.  Well above the numerical rank of the basis, so
#: hitting it means the request was below `TOL_FLOOR` in all but name.
N_POINTS_MAX = 96


def candidate_exponents(d, n=N_CANDIDATE):
    """Get the candidate exponent set for a distribution of particle-hole gaps.

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences, all of them (not an MPI slice).  Must be
        finite and strictly positive.
    n : int, optional
        Number of candidates.  Default value is `N_CANDIDATE`.

    Returns
    -------
    t : numpy.ndarray
        Candidate exponents, ascending.
    """

    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return np.zeros((0,))
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("particle-hole gaps must be finite and strictly positive")

    return np.geomspace(CANDIDATE_LO / d.max(), CANDIDATE_HI / d.min(), n)


def select_exponents(d, n_points, candidates=None):
    """Select exponents by pivoted QR on the exponential basis.

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences, all of them.
    n_points : int
        Number of exponents to keep.
    candidates : numpy.ndarray, optional
        Candidate set.  `None` builds one with `candidate_exponents`.

    Returns
    -------
    t : numpy.ndarray
        Selected exponents, ascending.
    """

    if candidates is None:
        candidates = candidate_exponents(d)

    return np.sort(
        candidates[selection_order(d, candidates)[: min(int(n_points), candidates.size)]]
    )


def selection_order(d, candidates):
    """Rank the candidate exponents by how much independent information each adds.

    Split out from `select_exponents` because the ordering does not depend on how many
    points are wanted: a search over grid sizes ranks the candidates once and slices,
    rather than repeating an ``(nov, N_CANDIDATE)`` QR at every size it tries.

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences, all of them.
    candidates : numpy.ndarray
        Candidate exponents.

    Returns
    -------
    order : numpy.ndarray
        Indices into `candidates`, most informative first.
    """

    # Column-normalising first makes the pivoting select for independence rather than for
    # magnitude: the columns at small `t` are all close to the constant vector and would
    # otherwise dominate by norm alone.
    basis = np.exp(-np.outer(d, candidates))
    norms = np.linalg.norm(basis, axis=0)
    norms[norms == 0.0] = 1.0
    _, _, pivots = scipy.linalg.qr(basis / norms, mode="economic", pivoting=True)

    return pivots


def fit(d, shifts, t, rcond=RCOND):
    """Fit the resolvent for every shift on a shared exponent set.

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences, all of them.
    shifts : numpy.ndarray
        Pole shifts, as returned by `momentGW.eta0.select_poles`.
    t : numpy.ndarray
        Exponents.
    rcond : float, optional
        Singular-value cutoff.  Default value is `RCOND`.

    Returns
    -------
    coeffs : numpy.ndarray
        Coefficients, shape ``(n_shifts, n_points)``.
    residual : float
        Largest relative residual over every gap and every shift.
    amplification : float
        Largest ``sum |c| / max |g|`` over the shifts.  How far the fit relies on
        cancellation, and so how much of the residual is roundoff rather than
        approximation.
    """

    d = np.asarray(d, dtype=float)
    shifts = np.atleast_1d(np.asarray(shifts, dtype=float))
    t = np.asarray(t, dtype=float)

    basis = np.exp(-np.outer(d, t))
    coeffs = np.zeros((shifts.size, t.size))
    residual = 0.0
    amplification = 0.0

    for j, shift in enumerate(shifts):
        g = d / (d * d + shift)
        weight = 1.0 / g
        coeffs[j], *_ = np.linalg.lstsq(basis * weight[:, None], g * weight, rcond=rcond)
        residual = max(residual, float(np.max(np.abs(basis @ coeffs[j] - g) / g)))
        amplification = max(amplification, float(np.sum(np.abs(coeffs[j])) / np.max(g)))

    return coeffs, residual, amplification


def build(d, shifts, tol, n_points=None, n_points_max=N_POINTS_MAX):
    """Build a separable representation of the resolvent meeting a residual tolerance.

    The grid is grown against the *measured* residual, then walked back down to the
    smallest size that still meets the tolerance -- the same shape as
    `momentGW.eta0.select_poles`, and for the same reason: the achievable size depends on
    the gap distribution, which is not known before the calculation.

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences, all of them (not an MPI slice), referenced to
        the chemical potential by the caller or not -- the gaps are the same either way.
    shifts : numpy.ndarray
        Pole shifts.
    tol : float
        Requested relative residual.  Must be at least `TOL_FLOOR`.
    n_points : int, optional
        Fixed grid size.  If given, no search is performed: the fit and its measured
        residual are returned, and a tolerance miss is left for the caller to report,
        since an explicit size is an instruction.  Default value is `None`.
    n_points_max : int, optional
        Largest grid the search may reach.  Default value is `N_POINTS_MAX`.

    Returns
    -------
    t : numpy.ndarray
        Exponents.
    coeffs : numpy.ndarray
        Coefficients, shape ``(n_shifts, n_points)``.
    diagnostics : dict
        Measured residual, amplification, grid size and endpoints.
    """

    if not tol >= TOL_FLOOR:
        raise ValueError(
            f"Requested separable tolerance {tol:.3e} is below the measured float64 floor "
            f"{TOL_FLOOR:.3e}: the fit is solved in float64 and does not improve with more "
            "grid points below that, so a tolerance there would not describe the arithmetic "
            "that uses it"
        )

    candidates = candidate_exponents(d)
    order = selection_order(d, candidates)

    def attempt(n):
        exponents = np.sort(candidates[order[: min(int(n), candidates.size)]])
        coefficients, residual, amplification = fit(d, shifts, exponents)
        return exponents, coefficients, residual, amplification

    if n_points is not None:
        t, coeffs, residual, amplification = attempt(n_points)
    else:
        n = 24
        t, coeffs, residual, amplification = attempt(n)
        while residual > tol and n < n_points_max:
            n = min(n + 6, n_points_max)
            t, coeffs, residual, amplification = attempt(n)
        if residual > tol:
            raise RuntimeError(
                f"Separable fit cannot meet tolerance {tol:.3e} on a gap range "
                f"[{np.min(d):.6e}, {np.max(d):.6e}] within {n_points_max} exponents: "
                f"achieved {residual:.3e}"
            )
        # Walk back down: the step of six can overshoot by five
        while t.size > 1:
            trial = attempt(t.size - 1)
            if trial[2] > tol:
                break
            t, coeffs, residual, amplification = trial

    diagnostics = {
        "n_points": int(t.size),
        "residual": float(residual),
        "tol": float(tol),
        "amplification": float(amplification),
        "n_shifts": int(np.atleast_1d(shifts).size),
        "gap_range": (float(np.min(d)), float(np.max(d))),
        "exponent_range": (float(t[0]), float(t[-1])) if t.size else (0.0, 0.0),
        "n_points_fixed": n_points is not None,
    }

    return t, coeffs, diagnostics


def report(diagnostics):
    """Write the separable-fit diagnostics through the logging layer.

    Parameters
    ----------
    diagnostics : dict
        As returned by `build`.
    """

    residual, tol = diagnostics["residual"], diagnostics["tol"]
    style = logging.rate(residual, tol, tol * 100.0)
    logging.write(
        f"Separable resolvent:  {diagnostics['n_points']} exponents over "
        f"[{diagnostics['exponent_range'][0]:.2e}, {diagnostics['exponent_range'][1]:.2e}] "
        f"for {diagnostics['n_shifts']} shifts"
    )
    logging.write(
        f"Measured residual:  [{style}]{residual:.3e}[/] against tolerance {tol:.3e} "
        f"(amplification {diagnostics['amplification']:.2e})"
    )
    if diagnostics["n_points_fixed"] and residual > tol:
        logging.warn(
            f"Fixed grid of {diagnostics['n_points']} exponents achieves residual "
            f"[bad]{residual:.3e}[/] against tolerance {tol:.3e}"
        )
