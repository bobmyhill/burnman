# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, 2014, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

import numpy as np
import burnman.constants as constants

"""
Functions for the Einstein model of a solid.
"""

eps = np.finfo(np.float).eps

def thermal_energy(T, T_einstein, n):
    """
    calculate the thermal energy of a substance.  Takes the temperature,
    the Einstein temperature, and n, the number of atoms per molecule.
    Returns thermal energy in J/mol
    """
    if T <= eps:
        return 3.*n*constants.gas_constant*T_einstein*0.5 # zero point energy
    x = T_einstein/T
    E_th = 3.*n*constants.gas_constant*T_einstein*( 0.5 + 1. / (np.exp( x ) - 1.0) ) # include the zero point energy
    return E_th


def heat_capacity_v(T,T_einstein,n):
    """
    Heat capacity at constant volume.  In J/K/mol
    """
    if T <= eps:
        return 0.
    x = T_einstein/T
    C_v = 3.0*n*constants.gas_constant* ( x * x * np.exp( x ) / np.power( np.exp( x ) - 1.0, 2.0 ) )
    return C_v



