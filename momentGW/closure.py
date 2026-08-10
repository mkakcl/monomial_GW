r"""Terminal closures of a moment-conserving realization, as a truncation indicator.

A finite moment sequence does not determine a measure. The block Lanczos recurrence in
Dyson leaves that freedom in its last Jacobi block, and the natural truncation - drop the
next coupling - is the block **Gauss** rule, exact to moment order `2K - 1` for `K` blocks.
Sacrificing the top order buys a node pinned at a chosen energy: the **Gauss-Radau** rule
modifies only the last diagonal block and conserves orders `0` to `2K - 2`.

Both are admissible realizations of the same moments. They differ only in what they assume
about the moments nobody supplied, so the spread between the answers they give is an
indicator of how much the truncation is still deciding.

Notes
-----
The spread is an indicator, not a bound. Gauss and Gauss-Radau bracket
:math:`\int f d\mu` when `f` has derivatives of constant sign - Golub and Meurant,
*Matrices, Moments and Quadrature* - which holds for the resolvent at a real energy outside
the support, but a quasiparticle energy here is an eigenvalue of an upfolded Hamiltonian
rather than such an integral. Nothing in this module may be described as a bound on a
quasiparticle energy without a theorem that covers that step.
"""

import numpy as np
import scipy.linalg

__all__ = ["gauss_radau_jacobi"]


def gauss_radau_jacobi(jacobi, block_size, tau):
    r"""Pin a node at `tau` by modifying only the last diagonal block.

    Partitioning the block-Jacobi matrix as

    .. math::
        J = \begin{bmatrix} J_\mathrm{lead} & C^\dagger \\ C & M \end{bmatrix},

    `tau` is an eigenvalue of the modified matrix exactly when the Schur complement
    :math:`M^* - \tau I - C (J_\mathrm{lead} - \tau I)^{-1} C^\dagger` is singular, so the
    choice that pins it is

    .. math::
        M^* = \tau I + C (J_\mathrm{lead} - \tau I)^{-1} C^\dagger.

    Parameters
    ----------
    jacobi : numpy.ndarray
        Block-Jacobi matrix, Hermitian and block tridiagonal.
    block_size : int
        Size of one block. `jacobi` must be a whole number of blocks across.
    tau : float
        Energy to pin a node at. Must not already be an eigenvalue of the leading
        blocks, which is what makes the linear solve well posed.

    Returns
    -------
    modified : numpy.ndarray
        A copy of `jacobi` with its last diagonal block replaced.

    Raises
    ------
    ValueError
        If `jacobi` is not a whole number of blocks, has only one block - there is no
        leading part to solve against - or if `tau` is an eigenvalue of the leading
        blocks, which makes the pinning singular.
    """
    jacobi = np.asarray(jacobi)
    n = jacobi.shape[-1]
    if n % block_size:
        raise ValueError(
            f"Jacobi matrix of size {n} is not a whole number of blocks of {block_size}."
        )
    if n == block_size:
        raise ValueError(
            "Cannot pin a node with a single block: there are no leading blocks to solve "
            "against, so the rule has no freedom left to spend."
        )

    lead = jacobi[:-block_size, :-block_size]
    border = jacobi[-block_size:, :-block_size]

    shifted = lead - tau * np.eye(lead.shape[0], dtype=lead.dtype)
    # A pinned node has to be reachable from the leading blocks. If `tau` is already one of
    # their eigenvalues the solve is singular, and the failure is the interesting thing:
    # the rule cannot place a node where it has effectively already placed one.
    if np.linalg.cond(shifted) > 1.0 / np.finfo(float).eps:
        raise ValueError(
            f"Cannot pin a node at {tau!r}: it is an eigenvalue of the leading blocks, so "
            "the Schur complement that fixes the last block is singular."
        )

    modified = np.array(jacobi, copy=True)
    update = border @ scipy.linalg.solve(shifted, border.conj().T, assume_a="her")
    modified[-block_size:, -block_size:] = tau * np.eye(block_size) + update
    # The rule stays Hermitian; the solve can leave a little asymmetry behind.
    block = modified[-block_size:, -block_size:]
    modified[-block_size:, -block_size:] = 0.5 * (block + block.conj().T)
    return modified
