r"""Separable assembly of the dRPA resolvent moments under tensor hypercontraction.

The moment-conserving dRPA reduces to a sum over the certified poles of `momentGW.eta0`,
whose only system-sized objects are

.. math::
    Q^{(m+1)}(s) = V^T \frac{D^{m+1}}{D^2 + s} V ,
    \qquad m = 0, 2, \ldots, n_\mathrm{max} - 1 ,

with :math:`D_{ia} = \epsilon_a - \epsilon_i` the particle-hole gaps.  A density-fitted
vertex contracts against that scalar function of the gaps directly.  A
tensor-hypercontracted one, :math:`V_{P,ia} \to X_{iP} X_{aP}`, cannot: the vertex
separates over the pair index and the denominator does not.  Every previous attempt in
this project handled the mismatch by partitioning the pair space and expanding each region
differently; `THC.md` §2 records where that broke, and why the partition was an artefact
of an oscillatory inner quadrature rather than of the factorisation.

`momentGW.laplace` removes it at :math:`m = 0`, representing
:math:`D/(D^2+s) \approx \sum_k c_k(s) e^{-t_k D}` on a grid of exponents shared by every
pole.  This module supplies the higher powers and the assembly.  The power is recovered
*exactly* rather than fitted, by the binomial expansion of :math:`D^m` in the
chemical-potential referenced orbital energies :math:`\bar\epsilon = \epsilon - \mu`:

.. math::
    Q_X^{(m+1)}(s) = \sum_k c_k(s) \sum_{p=0}^{m} \binom{m}{p}\,
        O_p(t_k) \circ U_{m-p}(t_k) ,

.. math::
    [O_p(t)]_{PQ} = \sum_i X_{iP} X_{iQ}\, (-\bar\epsilon_i)^p\, e^{t \bar\epsilon_i} ,
    \qquad
    [U_q(t)]_{PQ} = \sum_a X_{aP} X_{aQ}\, \bar\epsilon_a^{\,q}\, e^{-t \bar\epsilon_a} ,

with :math:`\circ` the elementwise product.  Referencing to :math:`\mu` is load-bearing
twice over: it bounds both exponential factors by one, so nothing overflows however wide
the orbital range, and it makes :math:`-\bar\epsilon_i` and :math:`\bar\epsilon_a` both
positive, so the binomial is a sum of like-signed terms and introduces no cancellation of
its own.  The residual `laplace` measures therefore carries to every order unchanged.

Two cheaper routes were measured and rejected (`THC.md` §3.3): polynomial division of
:math:`D^{2r+1}/(D^2+s)` loses about one and a half decades per moment order to
cancellation against its :math:`(-s)^k` factors, and fitting :math:`D^{m+1}/(D^2+s)`
directly saturates, because a growing function cannot be built out of decaying
exponentials in float64.  The binomial route is flat in the moment order at the residual
`laplace` reports, and it is the kernel the earlier implementations already contained.

Why this is written as a stream
-------------------------------
The exponent loop has to be outermost.  That is the whole point of a *shared* grid: the
:math:`(n_\mathrm{max}+1)` collocation contractions each exponent costs then serve every
pole and every moment order, instead of being repeated per pole.  The pole loop cannot be
folded inside it either, because the solve that consumes the result,
:math:`(\mathbf{I} + 4 Q_X^{(1)} \mathbf{Z})^{-1}`, is nonlinear in :math:`Q_X^{(1)}`.

Assembling every :math:`Q_X^{(m+1)}(s_j)` before the first solve therefore costs
``n_shift * (nmom_max/2 + 1) * naux**2`` doubles.  That is formally :math:`O(N^2)`, but
with a constant near a hundred: 0.9 GB at benzene and 11.6 GB at a C16 alkane, which is
not a memory budget.  Every step of the assembly is elementwise in ``(P, Q)``, and the
solve takes its right-hand side column-wise, so the whole thing blocks over the second
auxiliary index.  `q_blocks` streams ``(cols, q)`` at a working set of
``(n_grid + n_shift) * n_order * naux * ncol``, which `block_size` fixes against a byte
budget rather than against a shape.  `build_q` is the unblocked convenience for tests and
small systems, and carries the constant above.

Scaling
-------
Operations ``O(n_grid * nmom_max * naux**2 * nmo)`` for the sector builds and
``O(n_grid * n_shift * nmom_max * naux**2)`` for the coefficient contraction; memory
``O((n_grid + n_shift) * nmom_max * naux * ncol)`` for the stream, plus whatever the
caller retains of the result.  Nothing here is quartic in operations or cubic in memory,
which is the requirement in `CLAUDE.md` this module exists to meet.
"""

import numpy as np
import scipy.linalg
from scipy.special import binom

from momentGW import eta0, laplace, logging
from momentGW.thc import dTDA as THCdTDA

#: Byte budget for the working set of one column block.  Only the block width is derived
#: from it, so it trades wall clock against footprint continuously rather than changing
#: what is computed; 256 MB is small enough to sit under any machine this runs on and
#: large enough that the blocked dimension stays wide enough for BLAS-3.
BLOCK_BYTES = 256 * 1024**2


def block_size(naux, n_grid, n_shift, n_order, n_power, budget=BLOCK_BYTES):
    """Choose the width of the block over the second auxiliary index.

    The working set of one block is the grid of assembled ``R_m(t_k)``, the pole
    contraction of it, and the two sets of sector products, all of which are
    proportional to the block width.

    Parameters
    ----------
    naux : int
        Auxiliary (THC rank) dimension.
    n_grid : int
        Number of exponents.
    n_shift : int
        Number of pole shifts.
    n_order : int
        Number of moment orders requested.
    n_power : int
        Number of energy powers held per sector, ``max(orders) + 1``.
    budget : int, optional
        Working-set budget in bytes.  Default value is `BLOCK_BYTES`.

    Returns
    -------
    block : int
        Widest block whose working set fits the budget, clamped to ``[1, naux]``.

    Notes
    -----
    A single column is the floor, so a budget below one column's working set is not
    honoured -- there is nothing narrower to fall back to.  That happens only for a rank
    large enough that ``naux * nmom_max * (n_grid + n_shift)`` doubles is itself over
    budget, which at the default is a THC rank past forty thousand.
    """

    per_column = 8 * naux * ((n_grid + n_shift) * n_order + 2 * n_power)
    return int(min(naux, max(1, budget // max(per_column, 1))))


def _sector(coll, energies, t, n_power, cols, out):
    r"""Build the energy-weighted collocation products of one sector at one exponent.

    Computes, for ``p`` up to ``n_power - 1``,

    .. math::
        [\,\cdot\,]_{PQ} = \sum_x X_{xP} X_{xQ}\, \varepsilon_x^p\, e^{-t \varepsilon_x} ,

    which is :math:`O_p` for the occupied sector and :math:`U_p` for the virtual one --
    the same expression, because `energies` carries :math:`-\bar\epsilon_i` for the
    occupied sector and :math:`\bar\epsilon_a` for the virtual one, both positive.

    Parameters
    ----------
    coll : numpy.ndarray
        Collocation array for the sector, shape ``(naux, n)``.
    energies : numpy.ndarray
        Positive sector energies, shape ``(n,)``.
    t : float
        Exponent.
    n_power : int
        Number of powers to build.
    cols : slice
        Block of the second auxiliary index.
    out : numpy.ndarray
        Output, shape ``(n_power, naux, cols.stop - cols.start)``.  Written in place.

    Returns
    -------
    out : numpy.ndarray
        The array that was passed in.

    Notes
    -----
    Operations ``O(n_power * naux * n * ncol)``, memory ``O(naux * n)`` beyond `out`.
    """

    scratch = np.empty_like(coll)
    sub = np.ascontiguousarray(coll[cols])
    weights = np.exp(-t * energies)

    for p in range(n_power):
        np.multiply(coll, weights[None, :], out=scratch)
        np.dot(scratch, sub.T, out=out[p])
        weights = weights * energies

    return out


def _energies(e_occ, e_vir, mu=None):
    """Split the orbital energies about the chemical potential and check the contract.

    Parameters
    ----------
    e_occ : numpy.ndarray
        Occupied orbital energies.
    e_vir : numpy.ndarray
        Virtual orbital energies.
    mu : float, optional
        Chemical potential.  Default value is the frontier midpoint.

    Returns
    -------
    occ : numpy.ndarray
        ``mu - e_occ``, strictly positive.
    vir : numpy.ndarray
        ``e_vir - mu``, strictly positive.
    mu : float
        The potential used.
    """

    e_occ = np.asarray(e_occ, dtype=float)
    e_vir = np.asarray(e_vir, dtype=float)
    if e_occ.size == 0 or e_vir.size == 0:
        raise ValueError("The separable assembly needs both an occupied and a virtual sector")

    if mu is None:
        mu = 0.5 * (float(np.max(e_occ)) + float(np.min(e_vir)))

    occ, vir = mu - e_occ, e_vir - mu
    if not (np.all(occ > 0) and np.all(vir > 0)):
        raise ValueError(
            f"The chemical potential {mu:.6e} does not separate the occupied and virtual "
            "orbital energies.  Both separable factors are bounded by one only if it does, "
            "and the binomial expansion of D^m is a sum of like-signed terms only if it does"
        )

    return occ, vir, float(mu)


def q_blocks(coll_occ, coll_vir, e_occ, e_vir, t, coeffs, orders, mu=None, block=None):
    r"""Stream the separable resolvent moments over blocks of the second auxiliary index.

    Yields the auxiliary-space objects

    .. math::
        [Q_X^{(m+1)}(s_j)]_{PQ} \approx \sum_{ia} X_{iP} X_{iQ} X_{aP} X_{aQ}\,
            D_{ia}^m \sum_k c_k(s_j) e^{-t_k D_{ia}} ,

    one column block at a time, in ascending column order.  The Cholesky factor of the
    Coulomb matrix is absent by design: the push-through identity moves it outside the
    pole solve (`THC.md` §5, `THC_method.tex` §"Removing the Cholesky factor"), so the
    solver never needs it.

    Parameters
    ----------
    coll_occ : numpy.ndarray
        Occupied collocation array, ``(naux, nocc)``.
    coll_vir : numpy.ndarray
        Virtual collocation array, ``(naux, nvir)``.
    e_occ : numpy.ndarray
        Occupied orbital energies, ``(nocc,)``.
    e_vir : numpy.ndarray
        Virtual orbital energies, ``(nvir,)``.
    t : numpy.ndarray
        Exponents, as returned by `momentGW.laplace.build`.
    coeffs : numpy.ndarray
        Fit coefficients, ``(n_shift, n_grid)``, as returned by `momentGW.laplace.build`.
    orders : sequence of int
        Moment orders ``m``.  The dRPA needs the even ones; nothing here requires that.
    mu : float, optional
        Chemical potential.  Default value is the frontier midpoint.
    block : int, optional
        Block width.  Default value is that of `block_size`.

    Yields
    ------
    cols : slice
        Block of the second auxiliary index.
    q : numpy.ndarray
        Shape ``(n_shift, len(orders), naux, cols.stop - cols.start)``.

    Notes
    -----
    Operations ``O(n_grid * n_power * naux**2 * nmo)`` for the sector builds plus
    ``O(n_grid * n_shift * n_order * naux**2)`` for the contraction against the
    coefficients, with the binomial assembly itself subleading at
    ``O(n_grid * n_power * n_order * naux**2)``; memory
    ``O((n_grid + n_shift) * n_order * naux * block)``.
    """

    # Contiguity is not cosmetic here: every inner call is a GEMM writing into a slice
    # of a preallocated block, and a strided operand would force a copy per exponent.
    coll_occ = np.ascontiguousarray(coll_occ, dtype=float)
    coll_vir = np.ascontiguousarray(coll_vir, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
    orders = np.atleast_1d(np.asarray(orders, dtype=int))

    if np.any(orders < 0):
        raise ValueError("Moment orders must be non-negative")
    if coeffs.shape[1] != t.size:
        raise ValueError(
            f"Coefficients are for {coeffs.shape[1]} exponents but {t.size} were given"
        )
    if coll_occ.shape[0] != coll_vir.shape[0]:
        raise ValueError(
            f"Collocation arrays disagree on the auxiliary dimension: "
            f"{coll_occ.shape[0]} against {coll_vir.shape[0]}"
        )
    if coll_occ.shape[1] != np.size(e_occ) or coll_vir.shape[1] != np.size(e_vir):
        raise ValueError(
            f"Collocation arrays and orbital energies disagree: "
            f"{coll_occ.shape[1]} occupied columns against {np.size(e_occ)} energies, "
            f"{coll_vir.shape[1]} virtual columns against {np.size(e_vir)} energies"
        )

    occ, vir, _ = _energies(e_occ, e_vir, mu=mu)
    naux = coll_occ.shape[0]
    n_grid, n_shift, n_order = t.size, coeffs.shape[0], orders.size
    n_power = int(orders.max()) + 1

    if block is None:
        block = block_size(naux, n_grid, n_shift, n_order, n_power)
    block = int(min(max(block, 1), naux))

    # Binomial coefficients are the same for every block and every exponent
    weights = [binom(m, np.arange(m + 1)) for m in orders]

    for start in range(0, naux, block):
        cols = slice(start, min(start + block, naux))
        ncol = cols.stop - cols.start

        o = np.empty((n_power, naux, ncol))
        u = np.empty((n_power, naux, ncol))
        grid = np.empty((n_grid, n_order, naux, ncol))

        for k, exponent in enumerate(t):
            _sector(coll_occ, occ, exponent, n_power, cols, o)
            _sector(coll_vir, vir, exponent, n_power, cols, u)

            for mi, m in enumerate(orders):
                r = grid[k, mi]
                np.multiply(o[0], u[m], out=r)
                for p in range(1, m + 1):
                    r += weights[mi][p] * o[p] * u[m - p]

        # One contraction against the coefficients for the whole block: the grid is
        # shared, so this is where every pole gets its own object and nowhere else
        q = coeffs @ grid.reshape(n_grid, -1)

        yield cols, q.reshape(n_shift, n_order, naux, ncol)


def build_q(coll_occ, coll_vir, e_occ, e_vir, t, coeffs, orders, mu=None, block=None):
    """Assemble the separable resolvent moments in full.

    The convenience form of `q_blocks`, for tests and for systems small enough that the
    whole set fits.  It holds ``n_shift * n_order * naux**2`` doubles, which is the
    footprint `q_blocks` exists to avoid; a solver should stream.

    Parameters
    ----------
    coll_occ : numpy.ndarray
        Occupied collocation array, ``(naux, nocc)``.
    coll_vir : numpy.ndarray
        Virtual collocation array, ``(naux, nvir)``.
    e_occ : numpy.ndarray
        Occupied orbital energies, ``(nocc,)``.
    e_vir : numpy.ndarray
        Virtual orbital energies, ``(nvir,)``.
    t : numpy.ndarray
        Exponents.
    coeffs : numpy.ndarray
        Fit coefficients, ``(n_shift, n_grid)``.
    orders : sequence of int
        Moment orders.
    mu : float, optional
        Chemical potential.  Default value is the frontier midpoint.
    block : int, optional
        Block width.  Default value is that of `block_size`.

    Returns
    -------
    q : numpy.ndarray
        Shape ``(n_shift, len(orders), naux, naux)``.
    """

    naux = np.shape(coll_occ)[0]
    n_shift = np.atleast_2d(coeffs).shape[0]
    n_order = np.atleast_1d(orders).size

    q = np.empty((n_shift, n_order, naux, naux))
    for cols, chunk in q_blocks(
        coll_occ, coll_vir, e_occ, e_vir, t, coeffs, orders, mu=mu, block=block
    ):
        q[:, :, :, cols] = chunk

    return q


def polynomial_moments(coll_occ, coll_vir, e_occ, e_vir, orders, mu=None, block=None):
    r"""Build the shift-independent auxiliary-space images of the gaps.

    These are :math:`\mathcal{D}_X^{(m)} = X \mathbf{D}^m X^T`.  A zero exponent turns the
    separable assembly into the plain binomial expansion of :math:`D^m`, so this is
    `build_q` with the resolvent fit replaced by the identity -- one code path, and it
    inherits the blocking.

    Parameters
    ----------
    coll_occ : numpy.ndarray
        Occupied collocation array, ``(naux, nocc)``.
    coll_vir : numpy.ndarray
        Virtual collocation array, ``(naux, nvir)``.
    e_occ : numpy.ndarray
        Occupied orbital energies.
    e_vir : numpy.ndarray
        Virtual orbital energies.
    orders : sequence of int
        Moment orders.
    mu : float, optional
        Chemical potential.  Default value is the frontier midpoint.
    block : int, optional
        Block width.  Default value is that of `block_size`.

    Returns
    -------
    d_x : numpy.ndarray
        Shape ``(len(orders), naux, naux)``.

    Notes
    -----
    Operations ``O(nmom_max * naux**2 * nmo)``, memory ``O(nmom_max * naux**2)`` for the
    result plus the block `build_q` streams it in.
    """

    return build_q(
        coll_occ,
        coll_vir,
        e_occ,
        e_vir,
        np.zeros(1),
        np.ones((1, 1)),
        orders,
        mu=mu,
        block=block,
    )[0]


def screened_moments(
    coll_occ, coll_vir, e_occ, e_vir, cou, weights, t, coeffs, orders, mu=None, block=None, out=None
):
    r"""Accumulate the screened auxiliary-space moments over the pole set.

    These are

    .. math::
        \mathcal{N}_X^{(m)} = \sum_j w_j
            \left(\mathbf{I} + 4 \mathcal{Q}_X^{(1)}(s_j) \mathbf{Z}\right)^{-1}
            \mathcal{Q}_X^{(m+1)}(s_j) ,

    the form the push-through identity leaves once the Cholesky factor of
    :math:`\mathbf{Z}` is moved outside the solve.  The price is a general LU in place of a
    Cholesky, at the same cost.

    Parameters
    ----------
    coll_occ : numpy.ndarray
        Occupied collocation array, ``(naux, nocc)``.
    coll_vir : numpy.ndarray
        Virtual collocation array, ``(naux, nvir)``.
    e_occ : numpy.ndarray
        Occupied orbital energies.
    e_vir : numpy.ndarray
        Virtual orbital energies.
    cou : numpy.ndarray
        The ``(naux, naux)`` Coulomb array, :math:`\mathbf{Z}`.
    weights : numpy.ndarray
        Pole weights.  The shifts enter only through `coeffs`.
    t : numpy.ndarray
        Exponents of the separable representation.
    coeffs : numpy.ndarray
        Fit coefficients, ``(n_shift, n_grid)``.
    orders : sequence of int
        Even moment orders, ascending and starting at zero.
    mu : float, optional
        Chemical potential.  Default value is the frontier midpoint.
    block : int, optional
        Block width.  Default value is that of `block_size`.
    out : numpy.ndarray, optional
        Destination, ``(len(orders), naux, naux)``, zeroed by the caller.  May be a
        strided view.  `None` allocates one.

    Returns
    -------
    n_x : numpy.ndarray
        The array written into.

    Notes
    -----
    Two passes, because the solve is nonlinear in :math:`\mathcal{Q}_X^{(1)}` and so
    cannot be folded into the shared-grid loop.  The first builds
    :math:`\mathcal{Q}_X^{(1)}(s_j)` for every pole, factorises
    :math:`\mathbf{I} + 4\mathcal{Q}_X^{(1)}(s_j)\mathbf{Z}` and overwrites the slot it
    came from, so the peak is ``n_shift * naux**2`` rather than twice that.  The second
    streams the higher orders past the stored factors a column block at a time, which is
    what bounds that stream by a byte budget instead of by ``n_shift * nmom_max *
    naux**2`` -- 3.4 GB at benzene, 41 GB at a C16 alkane.

    Holding every factorisation at once is the one place this carries a constant larger
    than the result: ``n_shift * naux**2`` against the ``(nmom_max + 1) * naux**2`` of the
    answer, so about twice.  The alternative is to re-sweep the exponent grid per pole,
    which multiplies the dominant cost by the pole count; the memory is quadratic either
    way, so the cheaper one wins.

    Operations ``O(n_shift * naux**3 + n_grid * nmom_max * naux**2 * nmo)``, memory
    ``O(n_shift * naux**2)``.
    """

    orders = list(np.atleast_1d(np.asarray(orders, dtype=int)))
    if orders[0] != 0:
        raise ValueError(
            f"The screened moments are generated from the zeroth order, which supplies the "
            f"factorisation every other order solves against; got orders {orders}"
        )

    naux = np.shape(coll_occ)[0]
    args = (coll_occ, coll_vir, e_occ, e_vir, t, coeffs)
    kwargs = dict(mu=mu, block=block)

    if out is None:
        out = np.zeros((len(orders), naux, naux))

    # Pass one: the zeroth order, and the factorisation every other order needs
    factors = build_q(*args, (0,), **kwargs)[:, 0]
    pivots = []
    for j, weight in enumerate(weights):
        matrix = np.eye(naux)
        matrix += 4.0 * np.dot(factors[j], cou)
        lu, piv = scipy.linalg.lu_factor(matrix, overwrite_a=True)
        out[0] += weight * scipy.linalg.lu_solve((lu, piv), factors[j])
        factors[j] = lu
        pivots.append(piv)

    # Pass two: the higher orders, streamed past those factors
    higher = orders[1:]
    if not higher:
        return out

    n_higher = len(higher)
    for cols, chunk in q_blocks(*args, higher, **kwargs):
        ncol = cols.stop - cols.start
        for j, weight in enumerate(weights):
            rhs = np.ascontiguousarray(chunk[j].transpose(1, 0, 2).reshape(naux, -1))
            solved = scipy.linalg.lu_solve((factors[j], pivots[j]), rhs)
            out[1:, :, cols] += weight * solved.reshape(naux, n_higher, ncol).transpose(1, 0, 2)

    return out


def combine(a_x, cou):
    r"""Run the odd/even recursion that generates every moment from the two families.

    .. math::
        \zeta^{(m)} = A_X^{(m)} + 4 \!\!\sum_{\substack{k = 1,3,5,\dots \\ k \le m-1}}\!\!
            A_X^{(m-1-k)}\, \mathbf{Z}\, \zeta^{(k)}

    Parity keeps the two families apart without a special case: for even `m` every
    ``m - 1 - k`` with odd `k` is even, and for odd `m` every one of them is odd.  So the
    odd moments are built entirely from the polynomial family and the even ones entirely
    from the screened family, which is why only those two have to be assembled.

    Parameters
    ----------
    a_x : numpy.ndarray
        ``(nmom_max + 1, naux, naux)``, the polynomial family in the odd slots and the
        screened family in the even ones.
    cou : numpy.ndarray
        The ``(naux, naux)`` Coulomb array.  It appears between every pair of factors
        because the push-through identity left both families without their Cholesky
        factors.

    Returns
    -------
    zeta : numpy.ndarray
        ``(nmom_max + 1, naux, naux)``, in the convention
        `momentGW.thc.dTDA.build_se_moments` consumes.

    Notes
    -----
    Operations ``O(nmom_max**2 * naux**3)``, memory ``O(nmom_max * naux**2)``.  Never
    touches the particle-hole index.
    """

    nmom_max = a_x.shape[0] - 1
    zeta = np.empty_like(a_x)

    zeta[0] = a_x[0]
    if nmom_max >= 1:
        zeta[1] = a_x[1]

    for m in range(2, nmom_max + 1):
        zeta[m] = a_x[m]
        for k in range(1, m, 2):
            zeta[m] += 4.0 * np.linalg.multi_dot((a_x[m - 1 - k], cou, zeta[k]))

    return zeta


class dRPA(THCdTDA):
    r"""Compute the self-energy moments using dRPA and numerical integration with tensor
    hypercontraction.

    Produces the same ``(nmom_max + 1, naux, naux)`` object
    `momentGW.thc.dTDA.build_se_moments` already consumes, so nothing downstream changes:
    the convention there applies the two outer factors of the Coulomb matrix at
    contraction time, which is exactly the convention the push-through identity leaves
    this solver in.

    Parameters
    ----------
    gw : BaseGW
        GW object.
    nmom_max : int
        Maximum moment number to calculate.
    integrals : BaseIntegrals
        Integrals object.
    mo_energy : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital energies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw.mo_energy`.
    mo_occ : numpy.ndarray or tuple of numpy.ndarray, optional
        Molecular orbital occupancies. If a tuple is passed, the first
        element corresponds to the Green's function basis and the second to
        the screened Coulomb interaction. Default value is that of
        `gw.mo_occ`.

    Notes
    -----
    Two families of auxiliary-space objects generate every moment: `polynomial_moments`
    for the odd orders and `screened_moments` for the even ones, combined by `combine`.
    The Cholesky factor of :math:`\mathbf{Z}` that substituting :math:`V \to L^T X` would
    appear to need is removed by the push-through identity
    :math:`(\mathbf{I}+\mathbf{AB})^{-1}\mathbf{A} = \mathbf{A}(\mathbf{I}+\mathbf{BA})^{-1}`.

    Operations are ``O(n_grid * nmom_max * naux**2 * nmo)`` for the separable assembly,
    ``O(n_shift * naux**3)`` for the factorisations and ``O(nmom_max**2 * naux**3)`` for
    the recursion; memory is ``O((n_shift + nmom_max) * naux**2)``.  Cubic and quadratic,
    which is the requirement in `CLAUDE.md` the whole path exists to meet.

    See Also
    --------
    momentGW.laplace : the separable representation of the resolvent.
    momentGW.rpa.dRPA : the density-fitted solver this reproduces.
    """

    def kernel(self, exact=False):
        """Run the polarizability calculation to compute moments of the self-energy.

        Parameters
        ----------
        exact : bool, optional
            Must be `False`.  There is no exact tensor-hypercontracted route: the
            comparison an exact calculation would provide is against the density-fitted
            `dRPA`, not against a variant of this one.  Default value is `False`.

        Returns
        -------
        moments_occ : numpy.ndarray
            Moments of the occupied self-energy for each spin channel.
        moments_vir : numpy.ndarray
            Moments of the virtual self-energy for each spin channel.
        """

        if exact:
            raise NotImplementedError(
                "There is no exact tensor-hypercontracted dRPA: compare against "
                "polarizability='drpa-exact' with the density-fitted integrals instead"
            )

        return super().kernel()

    def _sectors(self):
        """Split the screened-Coulomb space into its occupied and virtual halves.

        Returns
        -------
        e_occ : numpy.ndarray
            Occupied orbital energies.
        e_vir : numpy.ndarray
            Virtual orbital energies.
        d : numpy.ndarray
            Particle-hole gaps, all of them.
        """

        e_occ = self.mo_energy_w[self.mo_occ_w > 0]
        e_vir = self.mo_energy_w[self.mo_occ_w == 0]
        return e_occ, e_vir, (e_vir[None, :] - e_occ[:, None]).ravel()

    def _select_poles(self, d, d_first):
        """Select the certified pole set for the rational inverse square root.

        Parameters
        ----------
        d : numpy.ndarray
            Particle-hole gaps.
        d_first : numpy.ndarray
            The auxiliary-space image of the gaps, ``X D X^T``.

        Returns
        -------
        shifts : numpy.ndarray
            Pole shifts.
        weights : numpy.ndarray
            Weights.

        Notes
        -----
        The coupling bound is ``4 ||W||_F^2`` with ``W = D^{1/2} V``, and for a
        hypercontracted vertex ``V = L^T X`` that trace is exactly
        ``4 sum_PQ Z_PQ [X D X^T]_PQ`` -- ``O(naux^2)``, with no pair index anywhere.  The
        density-fitted route also tries ``||W||_1 ||W||_inf`` and keeps the smaller, which
        is not available here: those norms need the vertex with its pair index, which is
        the ``O(N^3)`` array this path exists to avoid.  It costs nothing worth having --
        the enclosure is dominated by ``max(d^2)``, and measured across the baseline
        systems, replacing the trace with the exact largest eigenvalue moves the pole
        count by one.
        """

        coupling = 4.0 * float(np.sum(self.cou * d_first))
        lmin, lmax = eta0.certified_interval(d, coupling)

        tol = self.gw.eta0_tol
        n_poles = self.gw.eta0_n_poles
        shifts, weights, delta, _ = eta0.select_poles(lmin, lmax, tol, n_poles=n_poles)

        if n_poles is not None and delta > tol:
            logging.warn(
                f"Fixed eta0_n_poles = {n_poles} achieves scalar error "
                f"[bad]{delta:.3e}[/] against tolerance {tol:.3e}"
            )

        logging.write(
            f"Certified poles:  {shifts.size} over "
            f"[{shifts.min():.3e}, {shifts.max():.3e}], scalar error {delta:.3e} "
            f"against tolerance {tol:.3e}"
        )

        return shifts, weights

    @logging.with_timer("Density-density moments")
    @logging.with_status("Constructing density-density moments")
    def build_dd_moments(self):
        r"""Build the moments of the density-density response using tensor-hypercontraction.

        Returns
        -------
        moments : numpy.ndarray
            Moments of the density-density response, ``(nmom_max + 1, naux, naux)``.

        Notes
        -----
        See the class docstring, and `combine` for the recursion the two families feed.
        Every stage is ``O(N^3)`` in operations and ``O(N^2)`` in memory, with no
        particle-hole index formed anywhere.
        """

        # The whole design targets the certified pole route: the partition every earlier
        # attempt needed was an artefact of an oscillatory quadrature, and there is no
        # hypercontracted Clenshaw-Curtis path to fall back to.  Inheriting the option
        # silently would report a grid that nothing reads.
        if self.gw.eta0_method != "hht":
            raise NotImplementedError(
                f"polarizability='thc-drpa' requires eta0_method='hht', got "
                f"{self.gw.eta0_method!r}: the separable route is derived against the "
                "certified pole set, not against a quadrature"
            )

        e_occ, e_vir, d = self._sectors()
        even = list(range(0, self.nmom_max + 1, 2))
        odd = list(range(1, self.nmom_max + 1, 2))

        a_x = np.zeros((self.nmom_max + 1, self.naux, self.naux))

        # The shift-independent family.  Order one is built whether or not the recursion
        # needs it, because the coupling bound the pole selection wants is a contraction
        # of exactly that object against the Coulomb matrix.
        polynomial = polynomial_moments(self.Li, self.La, e_occ, e_vir, odd or [1])
        if odd:
            a_x[1::2] = polynomial

        shifts, weights = self._select_poles(d, polynomial[0])
        del polynomial

        # The exponent grid every pole shares
        t, coeffs, diagnostics = laplace.build(
            d,
            shifts,
            self.gw.thc_laplace_tol,
            n_points=self.gw.thc_laplace_n_points,
        )
        laplace.report(diagnostics)

        # The screened family, accumulated into the even slots in place
        screened_moments(
            self.Li, self.La, e_occ, e_vir, self.cou, weights, t, coeffs, even, out=a_x[0::2]
        )

        return combine(a_x, self.cou)
