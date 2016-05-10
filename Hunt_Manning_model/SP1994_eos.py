#!/usr/bin/python

import numpy as np
from scipy.optimize import fsolve
from burnman import constants
R=constants.gas_constant # J/K/mol or m**3*Pa/K/mol

# Note that each function has a conversion where necessary,
# such that the user inputs pressure in Pa and it is converted internally to
# MPa
# M=0.01801528 # kg/mol

#######
# H2O
#######

ci_H2O=np.array([[0.,             0.,            0.24657686e6,  0.51359951e2,  0. ,          0.],
             [0.,             0.,            0.58638965e0, -0.28646939e-2, 0.31375577e-4,0.],
             [0.,             0.,           -0.62783840e1,  0.14791599e-1, 0.35779579e-3,0.15432925e-7],
             [0.,             0.,            0,            -0.42719875e0, -0.16325155e-4,0.],
             [0.,             0.,            0.56654978e4, -0.16580167e2,  0.76560762e-1,0.],
             [0.,             0.,            0.,            0.10917883e0,  0.,           0],
             [0.38878656e13, -0.13494878e9,  0.30916564e6,  0.75591105e1,  0.,           0],
             [0.,             0.,           -0.65537898e5,  0.18810675e3,  0.,           0],
             [-0.14182435e14, 0.18165390e9, -0.19769068e6, -0.23530318e2,  0.,           0],
             [0.,             0.,            0.92093375e5,  0.12246777e3,  0.,           0]])


########
# CO2
########
ci_CO2=[[0.,             0.,            0.18261340e7,  0.79224365e2,  0. ,          0.],
    [0.,             0.,            0.,            0.66560660e-4, 0.57152798e-5,0.30222363e-9],
    [0.,             0.,            0.,            0.59957845e-2, 0.71669631e-4,0.62416103e-8],
    [0.,             0.,           -0.13270279e1, -0.15210731e0,  0.53654244e-3,-0.71115142e-7],
    [0.,             0.,            0.12456776e0,  0.49045367e1,  0.98220560e-2,0.55962121e-5],
    [0.,             0.,            0.,            0.75522299e0,  0.,           0.],
    [-0.39344644e12, 0.90918237e8,  0.42776716e6, -0.22347856e2,  0.,           0],
    [0.,             0.,            0.40282608e3,  0.11971627e3,  0.,           0],
    [0.,             0.22995650e8, -0.78971817e5, -0.63376456e2,  0.,           0],
    [0.,             0.,            0.95029765e5,  0.18038071e2,  0.,           0]]


def c_array(T, ci):
    array=[]
    for i in range( len(ci) ):
        array.append(ci[i][0]/(T*T*T*T) + ci[i][1]/(T*T) + ci[i][2]/T + ci[i][3] + ci[i][4]*T + ci[i][5]*T*T)
    return array


def helmholtz_free_energy(rho_SI, T, ci):
    rho = rho_SI*1.e-6
    c=c_array(T, ci)
    AoverRT = c[0]*rho + (1./(c[1] + c[2]*rho \
                                   + c[3]*rho*rho + c[4]*rho*rho*rho \
                                   + c[5]*rho*rho*rho*rho) - 1./c[1]) \
                                   - (c[6]/c[7])*(np.exp(-c[7]*rho) - 1.) \
                                   - (c[8]/c[9])*(np.exp(-c[9]*rho) - 1.)
    return AoverRT*R*T
    
def pressure(rho_SI, T, ci): # rho supplied in mol/m^3, P returned in Pa
    rho = rho_SI*1.e-6
    c=c_array(T, ci)
    trm=(c[2] + 2.*c[3]*rho + 3.*c[4]*rho*rho + 4.*c[5]*rho*rho*rho) \
        / (np.power((c[1] + c[2]*rho + c[3]*rho*rho \
            + c[4]*rho*rho*rho + c[5]*rho*rho*rho*rho),2.)) # the term in brackets in eq. 2
    return 1.e6*R*T*(rho + c[0]*rho*rho - rho*rho*trm \
                    + c[6]*rho*rho*np.exp(-c[7]*rho) \
                    + c[8]*rho*rho*np.exp(-c[9]*rho))


def _rho(rho_SI, P, T, ci): # solve for density in mol/m^3, P in Pa
    return P - pressure(rho_SI[0], T, ci)

def find_rho(P, T, ci): # finds rho (mol/m^3) in SI units for a given P, T, ci
    return fsolve(_rho, 1.e10, args=(P, T, ci))[0]

def lnf(P, T, ci): # P is in Pa, returned in ln(Pa)
    rho = fsolve(_rho, 1.e10, args=(P, T, ci))[0]
    A = helmholtz_free_energy(rho, T, ci)
    return (np.log(rho) + A/(R*T) + P/(rho*R*T)) + np.log(R*T) - 1.

def lnfH2O(P, T): # P is in Pa, returned in ln(Pa)
    ci = ci_H2O
    rho = fsolve(_rho, 1.e10, args=(P, T, ci))[0]
    A = helmholtz_free_energy(rho, T, ci)
    return (np.log(rho) + A/(R*T) + P/(rho*R*T)) + np.log(R*T) - 1.

