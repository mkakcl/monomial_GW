"""Tests for the terminal-closure truncation indicator."""

import numpy as np
import pytest

from momentGW.closure import gauss_radau_jacobi


def block_tridiagonal(nblock, block_size, seed=17):
    """Build a Hermitian block-tridiagonal matrix, as a block Lanczos run leaves."""
    rng = np.random.default_rng(seed)
    n = nblock * block_size
    matrix = np.zeros((n, n))
    for i in range(nblock):
        a = rng.normal(size=(block_size, block_size))
        matrix[i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size] = (
            a + a.T
        )
        if i:
            b = rng.normal(size=(block_size, block_size))
            lo, hi = (i - 1) * block_size, i * block_size
            matrix[hi : hi + block_size, lo:hi] = b
            matrix[lo:hi, hi : hi + block_size] = b.T
    return matrix


@pytest.mark.parametrize("nblock", [2, 3, 5])
@pytest.mark.parametrize("block_size", [1, 3])
@pytest.mark.parametrize("tau", [-3.7, 0.0, 2.5])
def test_pins_the_node(nblock, block_size, tau):
    """`tau` becomes an eigenvalue of the modified matrix."""
    jacobi = block_tridiagonal(nblock, block_size)

    modified = gauss_radau_jacobi(jacobi, block_size, tau)

    assert np.min(np.abs(np.linalg.eigvalsh(modified) - tau)) < 1e-9


@pytest.mark.parametrize("nblock", [2, 4])
def test_changes_only_the_last_diagonal_block(nblock):
    """Everything but the trailing block is left alone; that is what makes it a closure."""
    block_size = 3
    jacobi = block_tridiagonal(nblock, block_size)

    modified = gauss_radau_jacobi(jacobi, block_size, 1.25)

    np.testing.assert_allclose(modified[:-block_size], jacobi[:-block_size], rtol=0, atol=0)
    np.testing.assert_allclose(
        modified[-block_size:, :-block_size], jacobi[-block_size:, :-block_size], rtol=0, atol=0
    )
    assert not np.allclose(modified[-block_size:, -block_size:], jacobi[-block_size:, -block_size:])


def test_stays_hermitian():
    """The rule has to remain a valid Jacobi matrix."""
    jacobi = block_tridiagonal(4, 3)

    modified = gauss_radau_jacobi(jacobi, 3, -1.1)

    np.testing.assert_allclose(modified, modified.T, rtol=0, atol=1e-12)


def test_single_block_has_no_freedom_to_spend():
    """With one block there are no leading blocks to solve against."""
    with pytest.raises(ValueError, match="single block"):
        gauss_radau_jacobi(block_tridiagonal(1, 3), 3, 0.5)


def test_size_must_be_a_whole_number_of_blocks():
    """A block size that does not divide the matrix is a caller error, not a guess."""
    with pytest.raises(ValueError, match="whole number of blocks"):
        gauss_radau_jacobi(block_tridiagonal(3, 3), 2, 0.5)


def test_pinning_an_existing_eigenvalue_fails_loudly():
    """A node cannot be pinned where the leading blocks already have one."""
    jacobi = block_tridiagonal(3, 2)
    existing = float(np.linalg.eigvalsh(jacobi[:-2, :-2])[0])

    with pytest.raises(ValueError, match="eigenvalue of the leading blocks"):
        gauss_radau_jacobi(jacobi, 2, existing)
