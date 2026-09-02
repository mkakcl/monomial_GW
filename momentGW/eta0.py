r"""Certified rational approximation of the inverse square root for the zeroth dRPA moment.

The zeroth moment of the dRPA density-density response, projected onto the RI
auxiliary space, is

.. math:: \eta_0 V = D^{1/2} \tilde{M}^{-1/2} W,
          \qquad \tilde{M} = D^2 + c W W^T, \qquad W = D^{1/2} V,

where `D` is the diagonal of particle-hole energy differences, `V` the RI
vectors, and `c = 4` for restricted spatial orbitals.  This module supplies the
scalar layer for evaluating :math:`\tilde{M}^{-1/2}` through a rational
approximation

.. math:: x^{-1/2} \approx \sum_{j=1}^{N} \frac{w_j}{x + s_j},
          \qquad x \in [\lambda_{\min}, \lambda_{\max}],

using the conformal-map quadrature of Hale, Higham and Trefethen (SIAM J.
Numer. Anal. 46, 2505 (2008), method 2).  Everything the projected kernel in
`momentGW.rpa` needs is here: pole and weight generation in extended
precision, a rigorous spectral enclosure for :math:`\tilde{M}`, the measured
scalar error over that enclosure, and pole-count selection against a requested
tolerance with an explicit failure when the tolerance cannot be met.

Two properties are load-bearing:

- The spectral interval is a theorem-backed enclosure built from `D` and norm
  bounds on `W`, not from Ritz values, so the scalar error measured on it
  bounds the relative spectral error of the operator approximation:
  :math:`r(\tilde{M}) = \tilde{M}^{-1/2} (I + E)` with
  :math:`\|E\|_2 \le \delta`.
- The error certificate is the *measured* supremum of
  :math:`|1 - \sqrt{x}\, r(x)|` over the certified interval, evaluated in
  extended precision -- the asymptotic convergence rate only seeds the search
  for the pole count.
"""

import numpy as np

from momentGW import logging

# Extended-precision scalar type used for coefficient generation and error
# measurement.  On this platform `numpy.longdouble` is 80-bit extended
# (eps ~ 1e-19); where it silently aliases float64 the certificates lose
# nothing structural, only their measurement precision, and the kernel-level
# comparisons against the legacy route still gate the result.
_LD = np.longdouble
_PI = _LD("3.14159265358979323846264338327950288")

#: Relative padding applied outward to both endpoints of the certified
#: interval.  The enclosure itself is exact in real arithmetic; the padding
#: covers floating-point rounding in forming `d**2` and the norm bounds.  It
#: is deliberately generous: the pole count depends only logarithmically on
#: the interval, so padding by 1e-8 costs nothing measurable, while padding
#: by a handful of ulps would require an argument about every upstream
#: rounding.
_INTERVAL_PAD = 1e-8

#: Smallest scalar tolerance the float64 kernel can honour.  The rational sum
#: is evaluated in float64 by the projected kernel, so certificates below a
#: few eps are not representable in the arithmetic that uses them.
TOL_FLOOR = 4 * np.finfo(np.float64).eps


def _agm(a, b):
    """Compute the arithmetic-geometric mean in extended precision.

    Parameters
    ----------
    a : numpy.longdouble
        First operand.
    b : numpy.longdouble
        Second operand.

    Returns
    -------
    agm : numpy.longdouble
        Arithmetic-geometric mean of `a` and `b`.
    """
    a = _LD(a)
    b = _LD(b)
    eps = np.finfo(_LD).eps
    for _ in range(64):
        if abs(a - b) <= 4 * eps * abs(a):
            break
        a, b = (a + b) / 2, np.sqrt(a * b)
    return (a + b) / 2


def _sncndn(u, emmc):
    """Compute the Jacobi elliptic functions for complementary parameter `emmc`.

    Descending-Landen recurrence in extended precision, vectorised over the
    argument.  The functions returned are those of parameter ``m = 1 - emmc``;
    taking the *complementary* parameter as the input avoids ever forming
    ``1 - m`` in floating point, which is the unstable step this replaces.

    Parameters
    ----------
    u : numpy.ndarray
        Arguments.
    emmc : numpy.longdouble
        Complementary parameter ``1 - m``.  Must satisfy ``0 <= emmc <= 1``.

    Returns
    -------
    sn : numpy.ndarray
        Jacobi ``sn(u | m)``.
    cn : numpy.ndarray
        Jacobi ``cn(u | m)``.
    dn : numpy.ndarray
        Jacobi ``dn(u | m)``.
    """
    u = np.asarray(u, dtype=_LD)
    emc = _LD(emmc)
    if not 0 <= emc <= 1:
        raise ValueError(f"Complementary parameter must be in [0, 1], got {emc}")

    if emc == 0:
        # m = 1: hyperbolic limit
        cn = 1 / np.cosh(u)
        return np.tanh(u), cn, cn.copy()
    if emc == 1:
        # m = 0: trigonometric limit
        return np.sin(u), np.cos(u), np.ones_like(u)

    # Descending Landen (Gauss) transformation.  The convergence tolerance is
    # set so the squared truncation error sits below extended-precision eps.
    tol = _LD(3e-10)
    a = _LD(1)
    em = []
    en = []
    c = None
    for _ in range(32):
        em.append(a)
        emc = np.sqrt(emc)
        en.append(emc)
        c = (a + emc) / 2
        if abs(a - emc) <= tol * a:
            break
        emc = emc * a
        a = c

    uu = c * u
    sn = np.sin(uu)
    cn = np.cos(uu)
    dn = np.ones_like(u)

    # Ascend back through the recurrence.  Lanes with sn == 0 keep the
    # trigonometric values, matching the scalar algorithm.
    nonzero = sn != 0
    safe_sn = np.where(nonzero, sn, 1)
    a_v = np.where(nonzero, cn / safe_sn, 0)
    c_v = c * a_v
    d_v = dn.copy()
    for i in range(len(em) - 1, -1, -1):
        b = em[i]
        a_v = a_v * c_v
        c_v = c_v * d_v
        d_v = (en[i] + a_v) / (b + a_v)
        a_v = c_v / b
    a_fin = 1 / np.sqrt(c_v * c_v + 1)
    sn_fin = np.where(sn >= 0, a_fin, -a_fin)
    cn_fin = c_v * sn_fin

    sn = np.where(nonzero, sn_fin, sn)
    cn = np.where(nonzero, cn_fin, cn)
    dn = np.where(nonzero, d_v, dn)
    return sn, cn, dn


def hht_coefficients(lmin, lmax, n_poles):
    r"""Generate the poles and weights of the HHT rational approximation.

    The `n_poles`-point midpoint rule on the conformal map
    ``t = sqrt(lmin) sc(u, k')`` of the integral
    :math:`x^{-1/2} = (2 / \pi) \int_0^\infty (t^2 + x)^{-1} dt` gives

    .. math:: x^{-1/2} \approx \sum_j \frac{w_j}{x + s_j}

    with all shifts and weights strictly positive.  The coefficients are
    generated in extended precision and cast to float64; the elliptic
    functions are evaluated through the complementary parameter
    ``k^2 = lmin / lmax`` directly, so no ``1 - k^2`` cancellation occurs
    anywhere in the construction.

    Parameters
    ----------
    lmin : float
        Lower endpoint of the spectral interval.  Must be positive.
    lmax : float
        Upper endpoint of the spectral interval.  Must satisfy
        ``lmax >= lmin``.
    n_poles : int
        Number of poles.

    Returns
    -------
    shifts : numpy.ndarray
        Pole shifts ``s_j``, float64, strictly positive.
    weights : numpy.ndarray
        Weights ``w_j``, float64, strictly positive.
    """
    if not (np.isfinite(lmin) and np.isfinite(lmax)):
        raise ValueError(f"Spectral interval must be finite, got [{lmin}, {lmax}]")
    if lmin <= 0:
        raise ValueError(f"Spectral interval must be positive, got lmin = {lmin}")
    if lmax < lmin:
        raise ValueError(f"Spectral interval is empty: [{lmin}, {lmax}]")
    if n_poles < 1:
        raise ValueError(f"At least one pole is required, got {n_poles}")

    lmin_ld = _LD(lmin)
    lmax_ld = _LD(lmax)
    k2 = lmin_ld / lmax_ld
    k = np.sqrt(k2)

    # K(k') by the AGM, again through the complementary route:
    # K(k') = pi / (2 agm(1, k)).
    kp_int = _PI / (2 * _agm(_LD(1), k))

    j = np.arange(1, n_poles + 1, dtype=_LD)
    u = (j - _LD(0.5)) * kp_int / _LD(n_poles)
    sn, cn, dn = _sncndn(u, k2)
    sc = sn / cn

    shifts = lmin_ld * sc * sc
    weights = (2 * kp_int * np.sqrt(lmin_ld) / (_PI * _LD(n_poles))) * dn / (cn * cn)

    if not (np.all(np.isfinite(shifts)) and np.all(np.isfinite(weights))):
        raise RuntimeError(
            f"Non-finite HHT coefficients for interval [{lmin}, {lmax}] with {n_poles} poles"
        )
    if not (np.all(shifts > 0) and np.all(weights > 0)):
        raise RuntimeError(
            f"Non-positive HHT coefficients for interval [{lmin}, {lmax}] with {n_poles} poles"
        )

    return shifts.astype(np.float64), weights.astype(np.float64)


def scalar_error(shifts, weights, lmin, lmax, n_grid=None):
    """Measure the scalar relative error of the rational approximation.

    Evaluates ``max |1 - sqrt(x) r(x)|`` over the certified interval on a
    logarithmically spaced grid in extended precision.  This is the quantity
    the certificate reports: it bounds the relative spectral error of the
    operator approximation over any matrix whose spectrum the interval
    encloses.

    Parameters
    ----------
    shifts : numpy.ndarray
        Pole shifts.
    weights : numpy.ndarray
        Weights.
    lmin : float
        Lower endpoint of the certified interval.
    lmax : float
        Upper endpoint of the certified interval.
    n_grid : int, optional
        Number of grid points.  Default resolves the error equioscillation
        with a wide margin: ``max(4001, 64 * n_poles)``.

    Returns
    -------
    error : float
        Measured supremum of the relative error over the grid.
    """
    shifts = np.asarray(shifts, dtype=_LD)
    weights = np.asarray(weights, dtype=_LD)
    if n_grid is None:
        n_grid = max(4001, 64 * shifts.size)

    x = np.exp(np.linspace(np.log(_LD(lmin)), np.log(_LD(lmax)), n_grid, dtype=_LD))
    r = np.zeros_like(x)
    for s, w in zip(shifts, weights):
        r += w / (x + s)
    return float(np.max(np.abs(1 - np.sqrt(x) * r)))


def certified_interval(d, coupling_bound):
    """Build a rigorous spectral enclosure for ``Mtilde = D^2 + c W W^T``.

    The enclosure is ``lambda_min = min(d)^2`` (since ``c W W^T`` is positive
    semidefinite) and ``lambda_max = max(d^2) + b`` where `coupling_bound`
    ``b >= c ||W W^T||_2`` is a norm bound supplied by the caller.  Both
    endpoints are padded outward for floating-point rounding.

    Parameters
    ----------
    d : numpy.ndarray
        Particle-hole energy differences, all of them (not an MPI slice).
        Must be finite and strictly positive.
    coupling_bound : float
        Upper bound on ``c ||W W^T||_2``.  Must be finite and non-negative.

    Returns
    -------
    lmin : float
        Certified lower endpoint.
    lmax : float
        Certified upper endpoint.
    """
    d = np.asarray(d)
    if d.size == 0:
        raise ValueError("Cannot certify a spectral interval for an empty particle-hole space")
    if not np.all(np.isfinite(d)):
        raise ValueError("Particle-hole gaps must be finite")
    if not np.all(d > 0):
        raise ValueError(
            "Particle-hole gaps must be strictly positive: the dRPA inverse square root "
            f"is not defined for a gapless system (min gap = {np.min(d):.6e})"
        )
    if not (np.isfinite(coupling_bound) and coupling_bound >= 0):
        raise ValueError(f"Coupling norm bound must be finite and non-negative: {coupling_bound}")

    lmin = float(np.min(d)) ** 2 * (1 - _INTERVAL_PAD)
    lmax = (float(np.max(d)) ** 2 + float(coupling_bound)) * (1 + _INTERVAL_PAD)
    return lmin, lmax


#: How far an eta0 relative error travels to the frontier, in eV per unit relative error.
#: **An observed maximum, not a proven bound.** Milestone 2.4 recorded 30-300x and the
#: budget used 300; re-measuring above the float64 floor (`delta > 1e-12`) over the same
#: three systems gives 350x on water/HF, 582x on ozone/PBE and 1008x on LiH/HF, so 300
#: understated it by 3.4x. This is the measured maximum rounded up. The frontier response
#: is not smooth in the perturbation - see `studies/eta0_target.py`, where it is
#: non-monotonic and reaches 8630x at loose tolerances - so nothing here should be read as
#: a guarantee. Set above the largest value observed (1076x, LiH/HF at a 1e-2 request in
#: `studies/eta0_target.py`) rather than at it, because the response is not monotonic in
#: the request and a constant sitting exactly on the observed maximum has no margin.
ETA0_FRONTIER_AMPLIFICATION = 1200.0

#: How far an eta0 relative error travels to the dd moments, per unit relative error.
#: Milestone 2.4's finding that the recurrence does not amplify, made quantitative: above
#: the float64 floor the largest ratio over water/HF, LiH/HF and ozone/PBE is 0.97, so 1.0
#: is a bound with a little room. Unlike the frontier figure this one is well behaved,
#: which is why a requested *moment* tolerance can be inverted and a requested frontier
#: accuracy cannot.
MOMENT_AMPLIFICATION = 1.0

#: Floor on a derived eta0 tolerance. Below this the rational approximation is asking for
#: an accuracy the double-precision arithmetic underneath it cannot deliver, so a target
#: that implies one is clamped and said out loud rather than silently missed.
ETA0_TOL_FLOOR = 1e-15


def eta0_tol_for_moment_tol(moment_tol):
    """Turn a requested dd-moment accuracy into the eta0 tolerance that delivers it.

    Milestone 3.1 asks for the eta0 tolerance and pole count to be selected from the
    accuracy actually wanted rather than set by hand. Only the tolerance needs deriving:
    the pole count is already chosen against it by `select_poles`, so fixing one fixes the
    other.

    The conversion is `MOMENT_AMPLIFICATION` inverted. The recurrence carries an eta0
    perturbation to the moments at a factor of at most 0.97 as measured, so this inversion
    is sound in a way the frontier one is not: see `eta0_tol_for_qp_tol`, which needs a
    constant 1200x larger and an empirical rather than a derived justification.

    Parameters
    ----------
    moment_tol : float
        Requested relative accuracy of the dd moments. Must be positive.

    Returns
    -------
    eta0_tol : float
        Scalar relative tolerance for the rational approximation.
    clamped : bool
        Whether the floor was hit, meaning the requested accuracy is not deliverable
        through this term and the caller was given the tightest tolerance available
        instead.

    Raises
    ------
    ValueError
        If `moment_tol` is not positive. A non-positive accuracy has no tolerance.
    """
    if not moment_tol > 0.0:
        raise ValueError(f"moment_tol must be positive, got {moment_tol!r}")
    derived = moment_tol / MOMENT_AMPLIFICATION
    if derived < ETA0_TOL_FLOOR:
        return ETA0_TOL_FLOOR, True
    return derived, False


def eta0_tol_for_qp_tol(qp_tol):
    """Turn a requested frontier accuracy in eV into an eta0 tolerance.

    The frontier counterpart of `eta0_tol_for_moment_tol`, and a weaker thing. The moment
    conversion inverts a factor the recurrence is *measured not to exceed*; this one
    inverts an observed maximum over three systems, of a response that is not monotonic in
    the request. It delivered the requested accuracy in all 18 cases swept in
    `studies/eta0_target.py`, and that is the whole of its justification.

    Two consequences, both of which the caller is entitled to know about. It is expensive:
    the constant is 1200x, so a requested frontier accuracy buys a far tighter eta0
    tolerance and several more poles than the same number requested of the moments. And it
    is checkable after the fact but not before - `eta0_diagnostics["frontier_bound_ev"]`
    reports the achieved scalar error carried through the same constant, and `dRPA` warns
    when that exceeds what was asked for.

    Parameters
    ----------
    qp_tol : float
        Requested frontier accuracy in eV. Must be positive.

    Returns
    -------
    eta0_tol : float
        Scalar relative tolerance for the rational approximation.
    clamped : bool
        Whether the floor was hit.

    Raises
    ------
    ValueError
        If `qp_tol` is not positive.
    """
    if not qp_tol > 0.0:
        raise ValueError(f"qp_tol must be positive, got {qp_tol!r}")
    derived = qp_tol / ETA0_FRONTIER_AMPLIFICATION
    if derived < ETA0_TOL_FLOOR:
        return ETA0_TOL_FLOOR, True
    return derived, False


def estimate_n_poles(lmin, lmax, tol):
    """Estimate the pole count needed for a scalar tolerance.

    Uses the conservative asymptotic rate ``error ~ exp(-a N)`` with
    ``a = 2 pi^2 / (log(lmax / lmin) + 6)``, which measured decay exponents
    exceed across condition numbers 1e2 to 1e16.  This only seeds the search
    in `select_poles`; the measured scalar error decides.

    Parameters
    ----------
    lmin : float
        Lower endpoint of the certified interval.
    lmax : float
        Upper endpoint of the certified interval.
    tol : float
        Requested scalar relative error.

    Returns
    -------
    n_poles : int
        Estimated pole count.
    """
    rate = 2 * np.pi**2 / (np.log(lmax / lmin) + 6)
    return max(1, int(np.ceil(np.log(10.0 / tol) / rate)))


def select_poles(lmin, lmax, tol, n_poles=None, n_poles_max=100):
    """Select poles and weights meeting a scalar tolerance on a certified interval.

    The asymptotic rate estimates a starting pole count, which is then walked
    down or up against the *measured* scalar error until the smallest count
    satisfying the tolerance is found.  Failure is explicit in both
    directions: a tolerance below what float64 kernel arithmetic can honour
    raises immediately, and a tolerance that `n_poles_max` poles cannot meet
    raises with the error actually achieved.

    Parameters
    ----------
    lmin : float
        Lower endpoint of the certified interval.
    lmax : float
        Upper endpoint of the certified interval.
    tol : float
        Requested scalar relative error.  Must be at least `TOL_FLOOR`.
    n_poles : int, optional
        Fixed pole count.  If given, no selection is performed: the
        coefficients and their measured error are returned, and a tolerance
        miss is reported by the caller rather than raised, since an explicit
        pole count is an instruction.  Default value is `None`.
    n_poles_max : int, optional
        Largest pole count the search may reach.  Default value is `100`.

    Returns
    -------
    shifts : numpy.ndarray
        Pole shifts, float64.
    weights : numpy.ndarray
        Weights, float64.
    error : float
        Measured scalar relative error over the certified interval.
    n_estimate : int
        Pole count the asymptotic rate predicted (for diagnostics).
    """
    if not tol >= TOL_FLOOR:
        raise ValueError(
            f"Requested eta0 tolerance {tol:.3e} is below the float64 representable floor "
            f"{TOL_FLOOR:.3e}: the projected kernel evaluates the rational sum in float64, "
            "so a certificate below a few machine epsilon would not describe the arithmetic "
            "that uses it"
        )

    n_estimate = estimate_n_poles(lmin, lmax, tol)

    if n_poles is not None:
        shifts, weights = hht_coefficients(lmin, lmax, n_poles)
        error = scalar_error(shifts, weights, lmin, lmax)
        return shifts, weights, error, n_estimate

    n = min(n_estimate, n_poles_max)
    shifts, weights = hht_coefficients(lmin, lmax, n)
    error = scalar_error(shifts, weights, lmin, lmax)

    if error <= tol:
        # Walk down to the smallest count that still meets the tolerance
        while n > 1:
            shifts_try, weights_try = hht_coefficients(lmin, lmax, n - 1)
            error_try = scalar_error(shifts_try, weights_try, lmin, lmax)
            if error_try > tol:
                break
            n -= 1
            shifts, weights, error = shifts_try, weights_try, error_try
    else:
        # Walk up until the tolerance is met or the cap is hit
        while error > tol and n < n_poles_max:
            n += 1
            shifts, weights = hht_coefficients(lmin, lmax, n)
            error = scalar_error(shifts, weights, lmin, lmax)
        if error > tol:
            raise RuntimeError(
                f"HHT pole selection cannot meet tolerance {tol:.3e} on the interval "
                f"[{lmin:.6e}, {lmax:.6e}] (condition number {lmax / lmin:.3e}) within "
                f"{n_poles_max} poles: achieved {error:.3e}"
            )

    return shifts, weights, error, n_estimate


def report(diagnostics):
    """Write the eta0 diagnostics through the logging layer.

    Parameters
    ----------
    diagnostics : dict
        Diagnostics dictionary assembled by the projected kernel in
        `momentGW.rpa.dRPA`.
    """
    lmin, lmax = diagnostics["interval"]
    cond = diagnostics["condition_number"]
    error = diagnostics["scalar_error"]
    tol = diagnostics["tol"]
    res = diagnostics["cholesky_residuals"]["max"]

    logging.write(
        f"Certified interval:  [{lmin:.6e}, {lmax:.6e}] "
        f"(condition = [{logging.rate(cond, 1e8, 1e12)}]{cond:.3e}[/])"
    )
    logging.write(
        f"Poles:  {diagnostics['n_poles']} "
        f"(rate estimate = {diagnostics['n_poles_estimate']}, "
        f"legacy npoints = {diagnostics['npoints_legacy']})"
    )
    style = logging.rate(error, tol, tol * 1e2)
    logging.write(f"Scalar error:  [{style}]{error:.3e}[/] (tol = {tol:.3e})")
    logging.write(f"Max Cholesky residual:  [{logging.rate(res, 1e-14, 1e-10)}]{res:.3e}[/]")
    if diagnostics.get("refinement") is not None:
        diff = diagnostics["refinement"]["max_abs_diff"]
        logging.write(
            f"Refinement check (+4 poles):  [{logging.rate(diff, 1e-12, 1e-8)}]{diff:.3e}[/]"
        )
