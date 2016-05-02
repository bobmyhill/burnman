import scipy.constants

"""
molar gas constant (R) in J mol^-1 K^-1
"""
gas_constant = scipy.constants.gas_constant


"""
Avogadro constant (N_A) in mol ^ -1
"""
Avogadro = scipy.constants.Avogadro


"""
Boltzmann constant (k_B) in J K^-1.

Note that we are not using scipy.constants.Boltzmann because it is not
available in older versions.
"""
Boltzmann = 1.3806488e-23 


"""
Dirac constant (hbar, Planck constant / 2*pi) in J s^-1
"""
Dirac = 1.054571726e-34
