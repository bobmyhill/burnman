import os, sys
sys.path.insert(1,os.path.abspath('..'))
import burnman
import numpy as np
from scipy import optimize
# This model is based on the de Koker, Karki and Stixrude (2013)
# FPMD simulations for MgO and SiO2.

# Their results are fit reasonably well across the binary by 
# a subregular solution model. However, it should be noted that 
# the MgO side of the binary could be fit better with a more ideal
# model involving mixing between MgO and Mg2SiO4. 
# Between Mg2SiO4 and MgSiO3, a rapid decrease in MgO activity 
# might mirror a change in silicate-oxide 
# speciation in the melt.

# For now, let's stick with the simpler binary model...

# To add FeO, we assume that MgO and FeO 
# behave in a similar way; i.e.:
# W_FeO_SiO2 = W_MgO_SiO2,
# W_FeO_MgO = 0

# Under these assumptions, we can model 
# mu_MgO, mu_FeO and mu_SiO2 as a function of 
# pressure, temperature and melt composition.

MgOSiO2 = burnman.minerals.DKS_2013_liquids.MgSiO3_liquid()
SiO2MgO = burnman.minerals.DKS_2013_liquids.MgSiO3_liquid()

class MgO_SiO2_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular MgO-SiO2 liquid'
        self.type='full_subregular'

        self.endmembers = [[burnman.minerals.DKS_2013_liquids.MgO_liquid(), 
                            '[Mg]O'],
                           [burnman.minerals.DKS_2013_liquids.SiO2_liquid(), 
                            '[Si]O2']]

        self.intermediates = [[[MgOSiO2, SiO2MgO]]]

        burnman.SolidSolution.__init__(self, molar_fractions)

liq_SiO2 = burnman.minerals.DKS_2013_liquids.SiO2_liquid()
liq_MgO  = burnman.minerals.DKS_2013_liquids.MgO_liquid()
liq_MgO_SiO2 =  MgO_SiO2_liquid()

intermediate_liquids = [burnman.minerals.DKS_2013_liquids.MgSiO3_liquid(),
                        burnman.minerals.DKS_2013_liquids.MgSi2O5_liquid(),
                        burnman.minerals.DKS_2013_liquids.MgSi3O7_liquid(),
                        burnman.minerals.DKS_2013_liquids.MgSi5O11_liquid(),
                        burnman.minerals.DKS_2013_liquids.Mg2SiO4_liquid(),
                        burnman.minerals.DKS_2013_liquids.Mg3Si2O7_liquid(),
                        burnman.minerals.DKS_2013_liquids.Mg5SiO7_liquid()]

    
pressures = np.linspace(5.e9, 200.e9, 20)
temperatures = np.linspace(2000., 6000., 20)

PTX = []
MD_excesses = []
for P in pressures:
    for T in temperatures:
        print P, T
        liq_SiO2.set_state(P, T)
        liq_MgO.set_state(P, T)
        for liquid in intermediate_liquids:
            n_cations = liquid.params['formula']['Mg']+liquid.params['formula']['Si']
            X_MgO = liquid.params['formula']['Mg']/n_cations
            liquid.set_state(P, T)
            excess_gibbs = liquid.gibbs/n_cations - ((1.-X_MgO)*liq_SiO2.gibbs + X_MgO*liq_MgO.gibbs)
            PTX.append([P, T, X_MgO])
            MD_excesses.append(excess_gibbs)


MgOSiO2 = burnman.minerals.DKS_2013_liquids.MgSiO3_liquid()
SiO2MgO = burnman.minerals.DKS_2013_liquids.MgSiO3_liquid()

# Parameters to fit ('O_theta': 2 , 'O_f': 3, 'T_0': 3000.)
#'V_0'
#'m' 
#'a' [0], [1], [2], [3], [4], [5], [6], [7], [8], [11]
#'zeta_0'
#'xi'
#'Tel_0'
#'eta' 
#'el_V_0' 


def fit_intermediates(data, V0, m0, a00, a10, a20, a30, a40, a50, a60, a70, a80, a110, zeta0, xi0, Tel0, eta0, elV0,\
                          V1, m1, a01, a11, a21, a31, a41, a51, a61, a71, a81, a111, zeta1, xi1, Tel1, eta1, elV1):

    # INTERMEDIATE 1
    MgOSiO2.params['V_0'] = V0
    MgOSiO2.params['m'] = m0
    MgOSiO2.params['a'] = [a00, a10, a20, a30, a40, a50, a60, a70, a80, 0., 0., a110,\
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    MgOSiO2.params['a'] = burnman.minerals.DKS_2013_liquids.vector_to_array(MgOSiO2.params['a'], MgOSiO2.params['O_f'], MgOSiO2.params['O_theta'])*1.e3 
    MgOSiO2.params['zeta_0'] = zeta0
    MgOSiO2.params['xi'] = xi0
    MgOSiO2.params['Tel_0'] = Tel0
    MgOSiO2.params['eta'] = eta0
    MgOSiO2.params['el_V_0'] = elV0

    # INTERMEDIATE 2
    SiO2MgO.params['V_0'] = V1
    SiO2MgO.params['m'] = m1
    SiO2MgO.params['a'] = [a01, a11, a21, a31, a41, a51, a61, a71, a81, 0., 0., a111,\
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    SiO2MgO.params['a'] = burnman.minerals.DKS_2013_liquids.vector_to_array(SiO2MgO.params['a'], SiO2MgO.params['O_f'], SiO2MgO.params['O_theta'])*1.e3 
    SiO2MgO.params['zeta_0'] = zeta1
    SiO2MgO.params['xi'] = xi1
    SiO2MgO.params['Tel_0'] = Tel1
    SiO2MgO.params['eta'] = eta1
    SiO2MgO.params['el_V_0'] = elV1



    excess_gibbs = []
    for datum in data:
        P, T, X = datum
        liq_MgO_SiO2.set_composition([X_MgO, 1.-X_MgO])
        liq_MgO_SiO2.set_state(P, T)
        excess_gibbs.append(liq_MgO_SiO2.excess_gibbs)
    return excess_gibbs



p = MgOSiO2.params

guesses = [p['V_0'], p['m'], -2984.241297, -380.9839126, 601.8088234, 7307.69753, 7.626381912, -328.367174, 38737.46417, 6251.230413, 402.4716495, -23578.93569, p['zeta_0'], p['xi'], p['Tel_0'], p['eta'], p['el_V_0'],\
               p['V_0'], p['m'], -2984.241297, -380.9839126, 601.8088234, 7307.69753, 7.626381912, -328.367174, 38737.46417, 6251.230413, 402.4716495, -23578.93569, p['zeta_0'], p['xi'], p['Tel_0'], p['eta'], p['el_V_0']]

print optimize.curve_fit(fit_intermediates, PTX, MD_excesses, guesses)
