from __future__ import absolute_import
from __future__ import print_function
import os.path
import sys
sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman import equilibrate
import numpy as np
from MSH_endmembers import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit

W = -91000.
R = 8.31446

class high_pressure_hydrous_melt(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'melt'
        self.endmembers = [[Mg2SiO4L, '[Mg]'],
                           [H2OL, '[Hh]']] # associated solution model
        self.solution_type = 'symmetric'
        self.energy_interaction = [[W]]
        burnman.SolidSolution.__init__(self, molar_fractions=molar_fractions)


class hydrous_forsterite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'hydrous forsterite'
        self.endmembers = [[fo, '[Mg]MgSiO4'],
                           [H2MgSiO4fo, '[Hh]MgSiO4']] # ordered model
        self.solution_type = 'ideal'
        burnman.SolidSolution.__init__(self, molar_fractions=molar_fractions)


class hydrous_wadsleyite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'hydrous wadsleyite'
        self.endmembers = [[wad, '[Mg]MgSiO4'],
                           [H2MgSiO4wad, '[Hh]MgSiO4']] # ordered model
        self.solution_type = 'ideal'
        burnman.SolidSolution.__init__(self, molar_fractions=molar_fractions)

class hydrous_ringwoodite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'hydrous ringwoodite'
        self.endmembers = [[ring, '[Mg]MgSiO4'],
                           [H2MgSiO4ring, '[Hh]MgSiO4']] # ordered model
        self.solution_type = 'ideal'
        burnman.SolidSolution.__init__(self, molar_fractions=molar_fractions)


#hyfo = hydrous_forsterite()
melt = high_pressure_hydrous_melt()

#hyfo.guess = np.array([0.999, 0.001])
melt.guess = np.array([0.01, 0.99])

pressure = 13.e9
ol_polymorph = fo # ring # fo
composition = {'Mg': 2 , 'Si': 1 , 'O': 4.}
assemblage = burnman.Composite([ol_polymorph, Mg2SiO4L])

assemblage.set_state(pressure, 2600.)
equality_constraints = [('P', pressure),
                        ('phase_proportion', (ol_polymorph, np.array([0.0])))]
sols, prm = equilibrate(composition, assemblage,
                        equality_constraints,
                        initial_state_from_assemblage=True,
                        store_iterates=False)

Tm = assemblage.temperature


composition = {'Mg': 2 , 'Si': 1.5 , 'O': 5.01, 'H': 0.02}

assemblage = burnman.Composite([hen, ol_polymorph, melt])
temperatures = np.linspace(1000., Tm-1., 101)

assemblage.set_state(pressure, 1400.)

equality_constraints = [('P', pressure),
                        ('T', temperatures)]
sols, prm = equilibrate(composition, assemblage,
                        equality_constraints,
                        initial_state_from_assemblage=True,
                        store_assemblage=True,
                        store_iterates=False)

ps = np.array([sol.assemblage.phases[2].molar_fractions for sol in sols]).T
acts = np.array([sol.assemblage.phases[2].activities for sol in sols]).T
gs = np.array([sol.assemblage.phases[2].activity_coefficients for sol in sols]).T

lngammas = np.log(gs)


#plt.plot(lngammas[0], temperatures, label=assemblage.phases[2].endmember_names[0])
#plt.plot(lngammas[1], temperatures, label=assemblage.phases[2].endmember_names[1])

#plt.scatter(lng2[0], temperatures, label=assemblage.phases[2].endmember_names[0])
#plt.scatter(lng2[1], temperatures, label=assemblage.phases[2].endmember_names[1])
plt.plot(ps[0], temperatures, label=assemblage.phases[2].endmember_names[0])
plt.plot(ps[1], temperatures, label=assemblage.phases[2].endmember_names[1])



def fit_RTlng_Mg2SiO4(Tm, R, W):
    def fn_RTlng_Mg2SiO4(temperatures, b):
        a = -W/((b - 1.)**2*Tm*temperatures)
        return a*((temperatures - Tm)*(temperatures - Tm) + (1. - b**2)*Tm*(temperatures - Tm))
    return fn_RTlng_Mg2SiO4

"""
temperatures = np.linspace(b*Tm, 2560, 1001)

lnRTg_Mg2SiO4 = fn_RTlng_Mg2SiO4(temperatures, Tm, b, R, W)
p_H2OL = np.sqrt(lnRTg_Mg2SiO4/W)
plt.plot(temperatures, p_H2OL)
"""
sol = curve_fit(fit_RTlng_Mg2SiO4(Tm, R, W), temperatures, R*temperatures*lngammas[0], [0.5])[0]
print(Tm, sol[0], sol[0]*Tm, Tm- sol[0]*Tm)

RTlng_Mg2SiO4 = fit_RTlng_Mg2SiO4(Tm, R, W)(temperatures, *sol)
ps_H2O = np.sqrt(RTlng_Mg2SiO4/W)
RTlng_H2O = (1. - ps_H2O) * (1. - ps_H2O) * W
plt.plot(ps_H2O, temperatures, label='H2O model', linestyle=':')


hyfo_img = mpimg.imread('data/hyfo_melting_Myhill_et_al_2017.png')
plt.imshow(hyfo_img, extent=[0.0, 1.0, 1073.15, 2873.15], aspect='auto')


data = np.genfromtxt('data/13GPa_fo-H2O.dat',
                     dtype=[float, float, float, (np.unicode_, 16)])
phases = list(set([d[3] for d in data]))

experiments = {ph: np.array([[d[0], d[1], d[2]] for d in data if d[3]==ph]).T
               for ph in phases}

for phase, expts in experiments.items():
    #plt.scatter(expts[2]/100., expts[0]+273.15, label=phase) # on a 1-cation basis
    plt.scatter(expts[2]/(expts[1] + expts[2]), expts[0]+273.15, label=phase)

plt.legend()
plt.show()
