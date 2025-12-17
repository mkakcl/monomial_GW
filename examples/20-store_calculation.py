"""Example of how to store a `momentGW` calculations."""

import os

import h5py
import numpy as np
from pyscf import dft, gto

from momentGW import GW
from momentGW.rpa import dRPA

# Key calculation and file variables for storage
path = os.path.dirname(os.path.realpath(__file__))
basis = "cc-pvdz"
polarizability = "dRPA"

# Define a molecule
mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
mol.basis = basis
mol.verbose = 5
mol.build()

# Run a DFT calculation
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# When running large calculations you may wish to store the key elements of the calculation.
# The key and most expensive terms are the density-density (dd) response moments and the
# self-energy (SE) moments.


nmom_max = 3

# Construct your GW inputs and integrals
gw = GW(mf)
gw.polarizability = polarizability
gw.npoints = 24
gw.compression = None
integrals = gw.ao2mo()

# Build the static term
static = gw.build_se_static(integrals)

# If you wish to store the minimum essential information regarding the calculation, the SE-moments
# can just be stored. These moments (or a subset) can then be used to solve Dyson's equation
# Memory cost to store - O(2 * nmom_max * N^2_orb)
# Maximum memory - O(N^3_orb) + O(2 * nmom_max * N^2_orb)

# Build the SE-moments
se_moments = gw.build_se_moments(nmom_max, integrals)

# Solve dyson's equation
gw.kernel(nmom_max, integrals=integrals, moments=se_moments)

with h5py.File("%s/data_min_%s_%s.h5" % (path, basis, polarizability), "w") as f:
    f.create_dataset("e_tot", data=np.array([mf.e_tot]))
    f.create_dataset("mo_occ", data=np.asarray(mf.mo_occ))
    f.create_dataset("mo_coeff", data=np.asarray(mf.mo_coeff))
    f.create_dataset("mo_energy", data=np.asarray(mf.mo_energy))
    f.create_dataset("static", data=np.asarray(static))
    f.create_dataset("se_moments", data=np.asarray(se_moments))


# If you also wish to store all the information required to build the se-moments you can also
# store the dd-moments. This requires more memory for the calculation.
# Memory cost to store - O(nmom_max * N^3_orb) + O(2 * nmom_max * N^2_orb)
# Maximum memory - O(nmom_max * N^3_orb) + O(2 * nmom_max * N^2_orb)

# Initialise the screened Coulomb object
rpa = dRPA(gw, nmom_max, integrals)

# Build the DD-moments
moments_dd = rpa.build_dd_moments()

# Build the SE-moments
moments = rpa.build_se_moments(moments_dd)

# Solve dyson's equation
gw.kernel(nmom_max, integrals=integrals, moments=moments)

with h5py.File("%s/data_large_%s_%s.h5" % (path, basis, polarizability), "w") as f:
    f.create_dataset("e_tot", data=np.array([mf.e_tot]))
    f.create_dataset("mo_occ", data=np.asarray(mf.mo_occ))
    f.create_dataset("mo_coeff", data=np.asarray(mf.mo_coeff))
    f.create_dataset("mo_energy", data=np.asarray(mf.mo_energy))
    f.create_dataset("static", data=np.asarray(static))
    f.create_dataset("moments_dd", data=np.asarray(moments_dd))
    f.create_dataset("se_moments", data=np.asarray(moments))
