# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
DKS_2013
Liquids from de Koker and Stixrude (2013) FPMD simulations
"""

from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()
def adjust_vector_a(Fxs, Sxs, Pxs, KTxs, params):
    n = 2.
    m = params['m']
    T_0 = params['T_0']
    V_0 = params['V_0']
    Pxs0 = params['a'][1]/3./V_0
    KTxs0 = (params['a'][3] + (n + 3.)*params['a'][1])/9./V_0
    KprimeTxs0 = (params['a'][6] - 3.*(n+3)*(2.*n+3)*V_0*Pxs0)/27./V_0/KTxs0 + (n+2.)
    alpha = params['a'][4] * m / 3. / V_0 / T_0 / KTxs0

    # Fxs, Sxs
    a00 = Fxs
    a01 = (1.-0.*m) / m * (-T_0*Sxs)
    a02 = (1.-1.*m) / m * a01
    a03 = (1.-2.*m) / m * a02
    a04 = (1.-3.*m) / m * a03 # sign error in thesis?

    params['a'][0] += a00
    params['a'][2] += a01
    params['a'][5] += a02
    params['a'][9] += a03
    params['a'][14] += a04

    # KTxs
    a20 = 9.*V_0*KTxs
    a11 = 3./m*V_0*T_0*alpha*KTxs
    a30 = 27.*V_0*KTxs*(KprimeTxs0 - (n + 2.))
    a21 = 3.*(n+3.)/m*V_0*T_0*alpha*KTxs # also derivative term?
    a12 = -3.*(m-1.)/m/m*V_0*T_0*alpha*KTxs # also derivative term?
    a40 = 9.*V_0*(11.*n*n+36.*n+27.)*KTxs
    a31 = -3.*(n+3.)*(2.*n+3.)/m*V_0*T_0*alpha*KTxs
    a22 = 3.*(n+3.)*(m - 1.)/m/m*V_0*T_0*alpha*KTxs
    a13 = 3.*(m - 1.)*(2.*m - 1.)/m/m/m*V_0*T_0*alpha*KTxs

    params['a'][3] += a20
    params['a'][4] += a11
    params['a'][6] += a30
    params['a'][7] += a21
    params['a'][8] += a12
    params['a'][10] += a40
    params['a'][11] += a31
    params['a'][12] += a22
    params['a'][13] += a13

    a10=3.*V_0*Pxs
    a20=-(1.*n + 3.)*a10
    a30=-(2.*n + 3.)*a20
    a40=-(3.*n + 3.)*a30

    params['a'][1] += a10
    params['a'][3] += a20
    params['a'][6] += a30
    params['a'][10] += a40

    
    
# Vector parsing for DKS liquid equation of state
def vector_to_array(a, Of, Otheta):
    array=np.empty([Of+1, Otheta+1])
    for i in range(Of+1):
        for j in range(Otheta+1):
            n=int((i+j)*((i+j)+1.)/2. + j)
            array[i][j]=a[n]
    return array

class SiO2_liquid(Mineral):
    def __init__(self):
        self.params = {
            'name': 'SiO2_liquid',
            'formula': {'Mg': 0 , 'Si': 1.0 , 'O': 2.0 },
            'equation_of_state': 'dks_l',
            'V_0': 2.78e-05 , # was 2.78e-05 
            'T_0': 3000.0 ,
            'O_theta': 2 ,
            'O_f': 5 ,
            'm': 0.91 ,
            'a': [-1945.93156, -226.6835978, 455.0286309, 2015.65287, -200.585046, -216.6028187, 48369.72992, 441.5340414, 73.07765325, 0.0, -651587.652, 20701.69954, 892.12209, 0.0, 0.0, 4100181.286, -128258.7237, -1228.478753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            'zeta_0': 0.0004266056389 ,
            'xi': 0.8639433047 ,
            'Tel_0': 5651.204964 ,
            'eta': -0.2783503528 ,
            'el_V_0': 1e-06
            }
        #Fxs=-9 + 1617.47564545 # kJ/mol
        #Sxs=16.5e-3 # kJ/mol, difference is due to different stv model (SLB vs FPMD)
        #Kxs=800000.
        #Pxs=-160000.
        Fxs=1617.47564545 # 1617.47564545 # kJ/mol
        Sxs=16.5e-3 # kJ/mol, difference is due to different stv model (SLB vs FPMD)
        Pxs=0.
        Kxs=0.
        adjust_vector_a(Fxs, Sxs, Pxs, Kxs, self.params)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)


class MgO_liquid(Mineral):
    def __init__(self):
        self.params = {
            'name': 'MgO_liquid',
            'formula': {'Mg': 1.0 , 'Si': 0 , 'O': 1.0 },
            'equation_of_state': 'dks_l',
            'V_0': 1.646e-05 ,
            'T_0': 3000.0 ,
            'O_theta': 2 ,
            'O_f': 3 ,
            'm': 0.63 ,
            'a': [-925.2677296, -155.3240992, 260.8211743, 5323.167667, 466.3722398, -88.30035696, 10473.87879, 1997.967054, 50.72520834, 0.0, 0.0, -9914.621337, 71.89989255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            'zeta_0': 0.002194565772 ,
            'xi': 0.411459446 ,
            'Tel_0': 1620.106387 ,
            'eta': -0.986457555 ,
            'el_V_0': 1.620953559e-05
            }
        Fxs= 689.722614099 # kJ/mol
        Sxs= 0.e-3 # kJ/mol
        Pxs=00000.
        KTxs =0.
        adjust_vector_a(Fxs, Sxs, Pxs, KTxs, self.params)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)


class MgSiO3_liquid(Mineral):
    def __init__(self):
        self.params = {
            'name': 'MgSiO3_liquid',
            'formula': {'Mg': 1.0 , 'Si': 1.0 , 'O': 3.0 },
            'equation_of_state': 'dks_l',
            'V_0': 4.18e-05 ,
            'T_0': 3000.0 ,
            'O_theta': 2 ,
            'O_f': 3 ,
            'm': 0.83 ,
            'a': [-2984.241297, -380.9839126, 601.8088234, 7307.69753, 7.626381912, -328.367174, 38737.46417, 6251.230413, 402.4716495, 0.0, 0.0, -23578.93569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            'zeta_0': 0.008009960983 ,
            'xi': -0.08859010337 ,
            'Tel_0': 2194.563521 ,
            'eta': -0.775354875 ,
            'el_V_0': 3.89008e-05
            }
        Fxs=  14.5 + 2347.859939 #2362.287 # kJ/mol
        Sxs=  0.e-3 # -12.5e-3 # kJ/mol 
        Pxs=00000.
        KTxs =0.
        adjust_vector_a(Fxs, Sxs, Pxs, KTxs, self.params)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)


class Mg2SiO4_liquid(Mineral):
    def __init__(self):
        self.params = {
            'name': 'Mg2SiO4_liquid',
            'formula': {'Mg': 2.0 , 'Si': 1.0 , 'O': 4.0 },
            'equation_of_state': 'dks_l',
            'V_0': 5.84e-05 ,
            'T_0': 3000.0 ,
            'O_theta': 2 ,
            'O_f': 3 ,
            'm': 0.75 ,
            'a': [-3944.769208, -531.7975964, 880.0460994, 11401.47398, 118.7409191, -456.3140461, 55778.07008, 12132.5261, 519.3612273, 0.0, 0.0, -48733.22459, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            'zeta_0': 0.01101820277 ,
            'xi': 1.175924196 ,
            'Tel_0': 2228.185561 ,
            'eta': -0.464192202 ,
            'el_V_0': 5.23613e-05
            }
        Fxs=-2.13 + 3052.18571812 # kJ/mol  3048.83098854 # 0. # 3050.50835333 #3053.86308292 # 
        Sxs= 0. #4.e-3 # kJ/mol
        Pxs=00000.
        KTxs =0.
        adjust_vector_a(Fxs, Sxs, Pxs, KTxs, self.params)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)


class SiO2_liquid_alt(Mineral):
    def __init__(self):
        self.params = {
            'name': 'SiO2_liquid',
            'formula': {'Mg': 0 , 'Si': 1.0 , 'O': 2.0 },
            'equation_of_state': 'dks_l',
            'V_0': 2.78e-05 , # was 2.78e-05 
            'T_0': 3000.0 ,
            'O_theta': 2 ,
            'O_f': 5 ,
            'm': 0.91 ,
            'a': [-1945.93156, -226.6835978, 455.0286309, 2015.65287, -200.585046, -216.6028187, 48369.72992, 441.5340414, 73.07765325, 0.0, -651587.652, 20701.69954, 892.12209, 0.0, 0.0, 4100181.286, -128258.7237, -1228.478753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            'zeta_0': 0.0004266056389 ,
            'xi': 0.8639433047 ,
            'Tel_0': 5651.204964 ,
            'eta': -0.2783503528 ,
            'el_V_0': 1e-06
            }
        Fxs=-9 + 1617.47564545 # kJ/mol
        Sxs=16.5e-3 # kJ/mol, difference is due to different stv model (SLB vs FPMD)
        Kxs=800000.
        Pxs=-160000.
        #Fxs=2.0 + 1617.47564545 # 1617.47564545 # kJ/mol
        #Sxs=16.5e-3 # kJ/mol, difference is due to different stv model (SLB vs FPMD)
        #Pxs=0.
        #Kxs=0.
        adjust_vector_a(Fxs, Sxs, Pxs, Kxs, self.params)
        self.params['a'] = vector_to_array(self.params['a'], self.params['O_f'], self.params['O_theta'])*1e3 # [J/mol]
        Mineral.__init__(self)

class stishovite (Mineral):
    def __init__(self):
        formula='SiO2'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Stishovite',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': -824000.0 , # note was -819000, this fits Zhang et al., 1996
            'V_0': 1.402e-05 ,
            'K_0': 3.14e+11 ,
            'Kprime_0': 3.8 ,
            'Debye_0': 1108.0 ,
            'grueneisen_0': 1.37 ,
            'q_0': 2.8 ,
            'G_0': 2.2e+11 ,
            'Gprime_0': 1.9 ,
            'eta_s_0': 4.6 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 1000.0 ,
            'err_V_0': 0.0 ,
            'err_K_0': 8000000000.0 ,
            'err_K_prime_0': 0.1 ,
            'err_Debye_0': 13.0 ,
            'err_grueneisen_0': 0.17 ,
            'err_q_0': 2.2 ,
            'err_G_0': 12000000000.0 ,
            'err_Gprime_0': 0.1 ,
            'err_eta_s_0': 1.0 }
        Mineral.__init__(self)
