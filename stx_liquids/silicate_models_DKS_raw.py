import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import DKS_2013_solids, DKS_2013_liquids
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import fsolve, brentq


R = burnman.constants.gas_constant

MgO_liq = DKS_2013_liquids.MgO_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()
stv = DKS_2013_solids.stishovite()
per = DKS_2013_solids.periclase()
mpv = DKS_2013_solids.perovskite()


class FMS_solution():
    def __init__(self, molar_fractions=None):
        self.name = 'MgO-SiO2 solution'


    def _get_properties(self, P, T):

        lmda = 1.43
        WA = lambda P, T: 0000. - T*52. # decent fit at 80 GPa, 5000 K
        WB = lambda P, T: -220000. - T*32. # decent fit at 80 GPa, 5000 K

        X = self.molar_fractions[2]
        Y = X/(X + lmda*(1. - X))
        G_ideal = R*T*(np.sum([xi * np.log(xi) for xi in self.molar_fractions if xi > 1.e-12]))
        self.delta_G = G_ideal + WA(P, T)*Y*Y*(1. - Y) + WB(P, T)*Y*(1. - Y)*(1. - Y) 
        self.delta_S = (WA(P, T-0.5) - WA(P, T+0.5))*Y*Y*(1. - Y) + (WB(P, T-0.5) - WB(P, T+0.5))*Y*(1. - Y)*(1. - Y)  - G_ideal/T
        self.delta_H = self.delta_G + T*self.delta_S
    
        return 1

    def _unit_vector_length(self, v):
        length = np.sqrt(np.sum([ vi*vi for vi in v ]))
        return v/length, length
    
    def set_state(self, P, T):

        molar_fractions = self.molar_fractions

        # Find partial gibbs
        # First, find vector towards MgO, FeO and SiO2
        dX = 0.001
        
        dB, XB = self._unit_vector_length(np.array([0., 1., 0.]) - self.molar_fractions)
        dB = dB*dX
        self.molar_fractions = molar_fractions + dB
        sol = self._get_properties(P, T)
        GB = self.delta_G

        dC, XC = self._unit_vector_length(np.array([0., 0., 1.]) - self.molar_fractions)
        dC = dC*dX
        self.molar_fractions = molar_fractions + dC
        sol = self._get_properties(P, T)
        GC = self.delta_G


        self.molar_fractions = molar_fractions
        sol = self._get_properties(P, T)
        G0 = self.delta_G

        self.partial_gibbs_excesses = np.array( [ 0.,
                                                  G0 + (GB - G0)/dX*XB,
                                                  G0 + (GC - G0)/dX*XC ] )

        MgO_liq.set_state(P, T)
        SiO2_liq.set_state(P, T)

        self.partial_gibbs = ( np.array( [ 0.,
                                           MgO_liq.gibbs,
                                           SiO2_liq.gibbs ] ) +
                               self.partial_gibbs_excesses )
        return sol


FMS = FMS_solution()
