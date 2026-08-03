"""Example of the certified rational-approximation eta0 route in `momentGW` calculations."""

from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV

from momentGW import GW

# Define a molecule
mol = gto.Mole()
mol.atom = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
mol.basis = "cc-pvdz"
mol.verbose = 0
mol.build()

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# The zeroth moment of the dRPA density-density response is computed by
# a Clenshaw-Curtis quadrature by default (`eta0_method="clencur"`, with
# its `npoints` parameter). Setting `eta0_method="hht"` replaces it with
# a certified rational approximation of the inverse square root: the
# pole count is selected against `eta0_tol`, the *measured* scalar error
# over a rigorous spectral enclosure, rather than an asymptotic rule.
gw = GW(mf)
gw.polarizability = "dRPA"
gw.eta0_method = "hht"
gw.eta0_tol = 1e-14
gw.kernel(nmom_max=7)
print(f"IP = {gw.qp_energy[mf.mo_occ > 0].max() * HARTREE2EV:#8.8f} eV")

# Structured diagnostics record what the certificate actually measured:
# the certified spectral interval and its condition number, the pole
# count (against the asymptotic estimate and the legacy `npoints`), the
# measured scalar error, and the residual of every auxiliary-space
# Cholesky solve.
diagnostics = gw.eta0_diagnostics
print(f"interval        = [{diagnostics['interval'][0]:.3e}, {diagnostics['interval'][1]:.3e}]")
print(f"condition       = {diagnostics['condition_number']:.3e}")
print(f"n_poles         = {diagnostics['n_poles']} (legacy npoints = 48)")
print(f"scalar error    = {diagnostics['scalar_error']:.3e}")
print(f"max residual    = {diagnostics['cholesky_residuals']['max']:.3e}")
