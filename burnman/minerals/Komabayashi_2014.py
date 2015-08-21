# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
Komabayashi_2013
^^^^^^^^^^^^^

Minerals from Komabayashi 2014 and references therein.

Note that Komabayashi reports room pressure Gibbs free energies as the 
polynomial

f + g*T + h*T*lnT + i*T^2 + j/T + k*T^0.5

As G = H_ref + intCpdT - T*S_ref - T*intCpoverTdT)
Cp = a + bT + cT^-2 + dT^-0.5
intCpdT = aT + 0.5bT^2 - c/T + 2dT^0.5 - the value at T_ref
-T*intCpoverTdT = -aTlnT - bT^2 + 0.5c/T + 2dT^0.5 + the value at T_ref

Thus
f = H_ref - intCpdT(T_ref) + T_ref*intCpoverTdT(T_ref)
g = a - S_ref
h = -a
i = 0.5b - b
j = -c + 0.5c
k = 4d

H_ref = f + intCpdT(T_ref) - T_ref*intCpoverTdT(T_ref)
S_ref = - h - g
a = -h
b = -2i
c = -2j
d = 0.25k

"""

from burnman.mineral import Mineral

class fcc_iron():
    def __init__(self):
        formula='Fe1.0'
        self.params = {
            'name': 'FCC iron',
            'a': 16300.,
            'b': 381.47162 ,
            'c': -52.2754 ,
            'd': 0.000177578 ,
            'e': -395355.43,
            'f': -2476.28 ,
            'V_0': 6.82,
            'K_0': 163.4 ,
            'Kprime_0': 5.38 ,
            'a_0': 7.e-5,
            'delta_0': 5.5,
            'kappa': 1.4,
            'T_0': 298.}

class hcp_iron ():
    def __init__(self):
        formula='Fe1.0'
        self.params = {
            'name': 'HCP iron',
            'a': 12460.921,
            'b': 386.99162 ,
            'c': -52.2754 ,
            'd': 0.000177578 ,
            'e': -395355.43,
            'f': -2476.28 ,
            'V_0': 6.753,
            'K_0': 163.4 ,
            'Kprime_0': 5.38 ,
            'a_0': 5.8e-5,
            'delta_0': 5.1,
            'kappa': 1.4,
            'T_0': 298.}

class liquid_iron ():
    def __init__(self):
        formula='Fe1.0'
        self.params = {
            'name': 'Liquid iron',
            'a': -9007.3402,
            'b': 290.29866 ,
            'c': -46.,
            'd': 0.0 ,
            'e': 0.0,
            'f': 0.0,
            'V_0': 6.88,
            'K_0': 148. ,
            'Kprime_0': 5.8,
            'a_0': 9.e-5,
            'delta_0': 5.1,
            'kappa': 0.56,
            'T_0': 298.}

class FeO_solid ():
    def __init__(self):
        formula='Fe1.0O1.0'
        self.params = {
            'name': 'FeO solid',
            'a': -279318.,
            'b': 252.848,
            'c': -46.12826,
            'd': -0.0057402984,
            'e': 0.0,
            'f': 0.0,
            'V_0': 12.256,
            'K_0': 149.,
            'Kprime_0': 3.83,
            'a_0': 4.5e-5,
            'delta_0': 4.25,
            'kappa': 1.4,
            'T_0': 298.}


class FeO_liquid ():
    def __init__(self):
        formula='Fe1.0O1.0'
        self.params = {
            'name': 'FeO liquid',
            'a': -245310.,
            'b': 231.879,
            'c': -46.12826,
            'd': -0.0057402984,
            'e': 0.0,
            'f': 0.0,
            'V_0': 13.16,
            'K_0': 128. ,
            'Kprime_0': 3.85,
            'a_0': 4.7e-5,
            'delta_0': 4.5,
            'kappa': 1.4,
            'T_0': 298.}
