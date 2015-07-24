#!/usr/bin/python
import os, sys
import numpy as np
sys.path.insert(1,os.path.abspath('..'))

import burnman
from SP1994_eos import *
from burnman import constants

from scipy.optimize import fsolve

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 373.946C (647.096 K)	217.7 atm (22.06 MPa)
Pc_H2O = 22.065417e6 # Pa
Tc_H2O = 647.14
rhoc_H2O = find_rho(Pc_H2O, Tc_H2O, ci_H2O)
rhoc_H2O = 322./0.018 # two different densities at this pressure, fsolve finds the wrong one...

# 31.04C (304.19 K)	72.8 atm (7,380 kPa)
Pc_CO2 = 7380e3 # Pa
Tc_CO2 = 304.13 # K
rhoc_CO2 = find_rho(Pc_CO2, Tc_CO2, ci_CO2)

temperature_arrays = [[1600., 1600., 7., 25.], [800.,800., 7., 25.], [Tc_H2O, Tc_CO2, 4., 5.]]
for array in temperature_arrays:
    temperature_H2O, temperature_CO2, xlim, ylim = array

    reduced_densities = np.linspace(0., 7., 101)
    compression_factors_H2O = np.empty_like(reduced_densities)
    compression_factors_CO2 = np.empty_like(reduced_densities)
    for i, reduced_density in enumerate(reduced_densities):
        rho_SI_H2O = reduced_density*rhoc_H2O
        rho_SI_CO2 = reduced_density*rhoc_CO2
        P_H2O = pressure(rho_SI_H2O, temperature_H2O, ci_H2O)
        P_CO2 = pressure(rho_SI_CO2, temperature_CO2, ci_CO2)
        compression_factors_H2O[i]= P_H2O/(rho_SI_H2O*constants.gas_constant*temperature_H2O)
        compression_factors_CO2[i]= P_CO2/(rho_SI_CO2*constants.gas_constant*temperature_CO2)


    plt.plot(reduced_densities, compression_factors_H2O, label='H2O')
    plt.plot(reduced_densities, compression_factors_CO2, label='CO2')
    plt.xlim(0., xlim)
    plt.ylim(0., ylim)
    plt.legend(loc='upper left')
    plt.show()


####
# FUGACITY
####


print lnf(1.*1.e8, 250, ci_CO2) - np.log(1.e5), '(should be 4.51 [ln bar])'
print lnf(50.*1.e8, 250, ci_CO2) - np.log(1.e5), '(should be 62.85 [ln bar])'
print lnf(1.*1.e8, 2000, ci_CO2) - np.log(1.e5), '(should be 7.11 [ln bar])'
print lnf(50.*1.e8, 2000, ci_CO2) - np.log(1.e5), '(should be 18.88 [ln bar])'


#G(H2O)
#
#   kbar C        500      1000
#   10.0      -338.12   -409.09
#   20.0      -321.55   -389.26

#G(CO2)
#
#   kbar C        500      1000
#   10.0      -484.09   -577.27
#   20.0      -453.32   -542.93

'''
per_HP = HP_2011_ds62.per()
br_HP = HP_2011_ds62.br()

T = 1000. + 273.15
P = 1.e9
print gibbs(P, T, ci_H2O)
print "-409.09"

T = 1000 + 273.15
P = 2.e9
print gibbs(P, T, ci_H2O)
print "-389.26"

exit()
'''
