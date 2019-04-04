# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
example_analytical_processing
-----------------------------
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman.processchemistry import dictionarize_formula, formula_mass
from burnman import Mineral, CombinedMineral, SolidSolution
from burnman.minerals import HP_2011_ds62, HP_2011_fluids, SE_2015
from burnman.solutionbases import transform_solution_to_new_basis
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit

from scipy.optimize import minimize



# TODO
# Add parameter constraints and uncertainties -> could do this inside a second misfit function similar to get_params
# Add state constraints and uncertainties -> could do this by grouping assemblages and systematic state uncertainties together inside a ?dictionary?, with P, T offset as hyperparameter(s)
# Add compositional degrees of freedom for solutions as hyperparameter(s)? Especially for Fe3+ and order/disorder? (might need to add constraints though...)





# Here we define all of our desired minerals
# All declared solution endmembers must be declared here,
# otherwise we can't fit them later

O2 = HP_2011_fluids.O2()
fcc_iron = SE_2015.fcc_iron()
bcc_iron = SE_2015.bcc_iron()
periclase = HP_2011_ds62.per()
formula = dictionarize_formula('FeO')
FeO = burnman.Mineral(params={'name': 'stoichiometric wustite (Sundman, 1991 + HP)',
                              'formula': formula,
                              'equation_of_state': 'hp_tmt',
                              'H_0': -265055.0,
                              'S_0': 59.52,
                              'V_0': 1.2239e-05,
                              'Cp': [46.13, 0.01148, 59.07, -0.023],
                              'a_0': 3.22e-05,
                              'K_0': 152.e9,
                              'Kprime_0': 4.9,
                              'Kdprime_0': -4.9/152.e9,
                              'n': sum(formula.values()),
                              'molar_mass': formula_mass(formula)})
formula = dictionarize_formula('Fe2/3O')
Fe23O = burnman.Mineral(params={'name': 'defect wustite (Sundman, 1991 + HP)',
                                'formula': formula,
                                'equation_of_state': 'hp_tmt',
                                'H_0': -267032.0,
                                'S_0': 31.655,
                                'V_0': 1.10701e-05,
                                'Cp': [38.44, 9.567e-3, 56.7, -2.056e-2],
                                'a_0': 2.79e-05,
                                'K_0': 152.e9,
                                'Kprime_0': 4.9,
                                'Kdprime_0': -4.9/152.e9,
                                'n': sum(formula.values()),
                                'molar_mass': formula_mass(formula)})

# Now we declare the solid solutions
# Again, remember that all of the endmembers must have been instantiated above
fper = burnman.SolidSolution(name = 'ferropericlase',
                             solution_type = 'subregular',
                             endmembers = [[periclase, '[Mg]O'],
                                           [FeO, '[Fe]O'],
                                           [Fe23O, '[Fef2/3V1/3]O']],
                             energy_interaction = [[[7625.0, 9485.0],
                                                    [40590., 23595.]],
                                                   [[-17136.0, 5164.0]]]) # the last two values are from Sundman, 1991. We don't want to change these.

gt = burnman.minerals.SLB_2011.garnet()
# Make a dictionary of the parent solutions which we will use
# All of the solid solutions should have been instantiated above
parent_ss = {'fper': fper,
             'gt': gt}

# Make another dictionary of the required child solutions
# (those which are identical to the parent solutions in some restricted subspace)
child_ss = {'alm-gr': transform_solution_to_new_basis(parent_ss['gt'],
                                                      np.array([[0., 1., 0., 0, 0],
                                                                [0., 0., 1., 0, 0]]))}



###################
# DATA SUBSECTION #
###################


# 1) Data from O'Neill et al., 2003 
# ---------------------------------

data = np.genfromtxt('data/ONeill_et_al_2003_iron_magnesiowuestite.dat', dtype=None, encoding='utf8')

F = 96484.56 # value of Faraday constant from paper
# All the magnesiowuestites are in equilibrium with metallic iron (fcc at 1473 K)
components = ['Mg', 'Fe', 'O']

# Choose the elements measured by EPMA/Moessbauer
fitted_elements = ['Mg', 'Fe', 'O']

# Calculate the states and associated uncertainties
states = [(d[1], d[2]) for d in data]
state_uncertainties = [np.diag([1.e6, 9.]) for d in data] # arbitrary 3 K uncertainty in T

# Calculate the compositions and covariance matrices
compositions = [np.array([d[5], 1. - d[5], 1. + 0.5*d[7]*(1. - d[5])]) for d in data]

# c = nMg, nFe, nO
# p = XMg, Fe3+/sumFe
dcdp = [np.array([[1., 0.],
                  [-1., 0.],
                  [-0.5*d[7], -0.5*d[5]]]) for d in data]
covp = [np.diag([d[6]*d[6], d[8]*d[8]]) for d in data]

compositional_uncertainties = [dcdp[i].dot(covp[i]).dot(dcdp[i].T) for i in range(len(covp))]
for i in range(len(covp)):
    compositional_uncertainties = [c + 0.01*np.diag(np.diag(c)) for c in compositional_uncertainties] # increase variance a bit to avoid singular matrix


# Make assemblages
O2.set_state(1.e5, 1473.)
emf_0 = 1.1 # # the correction *should* be +1.1! Don't make this negative!!
diff_muO2s = np.array([-4.*F*((d[3] + emf_0)*1.e-3) +
                       burnman.constants.gas_constant*d[2]*np.log(0.20946*d[1]/1.e5)
                       for d in data]) 

# Now let's prepare the assemblage by assigning covariances to each of the solid solutions        
assemblages = [burnman.Composite([fcc_iron, parent_ss['fper'], CombinedMineral([O2], [1.], [dmu, 0., 0.])]) for dmu in diff_muO2s] 
molar_fractions = []
molar_fraction_covariances = []
for i, assemblage in enumerate(assemblages):
    assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
    assemblage.phases[1].fitted_elements = fitted_elements
    assemblage.phases[1].composition = compositions[i]
    assemblage.phases[1].compositional_uncertainties = compositional_uncertainties[i]
    compute_and_set_phase_compositions(assemblage)
    assemblage.stored_compositions[1] = (assemblage.phases[1].molar_fractions,
                                         assemblage.phases[1].molar_fraction_covariances)
    
    assemblage.nominal_state = states[i]
    assemblage.state_covariances = state_uncertainties[i] 

assemblages = assemblages[0:-1] # THE LAST DATA POINT HAS A HIGH RESIDUAL...

##############################    
# DECLARE FITTING PARAMETERS #
##############################

def get_params():
    return [parent_ss['fper'].energy_interaction[0][0][0],
            parent_ss['fper'].energy_interaction[0][0][1],
            parent_ss['fper'].energy_interaction[0][1][0],
            parent_ss['fper'].energy_interaction[0][1][1]]

def set_params(params):
    [parent_ss['fper'].energy_interaction[0][0][0],
     parent_ss['fper'].energy_interaction[0][0][1],
     parent_ss['fper'].energy_interaction[0][1][0],
     parent_ss['fper'].energy_interaction[0][1][1]] = params



###################################################
# FITTING THE DATA (THIS BIT SHOULD BE AUTOMATIC) #
###################################################

def minimize_func(params, assemblages, chisqr):
    # set all params
    set_params(params)

    # reinitialize solutions (to update the underlying matrices)
    for k, ss in parent_ss.items():
        burnman.SolidSolution.__init__(ss)
    for k, ss in child_ss.items():
        ss.__dict__.update(transform_solution_to_new_basis(ss.parent,
                                                           ss.basis).__dict__)

    for i, assemblage in enumerate(assemblages):
        # Assign compositions and uncertainties to solid solutions
        for j, phase in enumerate(assemblage.phases):
            if isinstance(phase, burnman.SolidSolution):
                molar_fractions, phase.molar_fraction_covariances = assemblage.stored_compositions[j]
                phase.set_composition(molar_fractions)
            
        # Assign a state to the assemblage
        assemblage.set_state(*assemblage.nominal_state)
        
        # Calculate the misfit
        chisqr[i] = assemblage_affinity_misfit(assemblage)

    sumchisqr = np.sum(chisqr)
    print(sumchisqr)
    return sumchisqr

chisqr = np.empty(len(assemblages))
res = minimize(minimize_func, get_params(), args=(assemblages, chisqr), tol=1.e-5, options={'eps':1.e-3})

print(chisqr)
print(res.x)




# Some random plots


from burnman.equilibrate import equilibrate
xMgs = np.linspace(0.001, 0.99, 11)
fper_compositions = np.empty((len(xMgs), 3))

Fe3oversumFe = np.empty_like(xMgs)
volumes = np.empty_like(xMgs)
assemblage = burnman.Composite([fcc_iron, parent_ss['fper']])


import ternary

"""
fontsize = 12
offset = 0.14
fig, tax = ternary.figure(scale=1.) # points normalized to one
tax.boundary(linewidth=1.5)
tax.gridlines(color="black", multiple=0.2)
#tax.left_axis_label("$x_{Fe_{2/3}O}$", fontsize=fontsize, offset=offset)
#tax.right_axis_label("$x_{FeO}$", fontsize=fontsize, offset=offset)
#tax.bottom_axis_label("$x_{MgO}$", fontsize=fontsize, offset=offset)

tax.right_corner_label("$x_{MgO}$", position=None, rotation=0, offset=0.2, fontsize=fontsize)
tax.top_corner_label("$x_{FeO}$", position=None, rotation=0, offset=0.2, fontsize=fontsize)
tax.left_corner_label("$x_{Fe_{2/3}O}$", position=None, rotation=0, offset=0.2, fontsize=fontsize)
"""
for P in [1.e5, 10.e9, 20.e9]:
    equality_constraints = [('P', P), ('T', 1473.)]
    fguess = 0.1
    for i, xMg in enumerate(xMgs):
        composition = {'Mg': xMg, 'Fe': 1. - xMg, 'O': 1.}
        parent_ss['fper'].guess = np.array([xMg, (1. - fguess)*(1. - xMg), fguess*(1. - xMg)])
        sols, prm = equilibrate(composition, assemblage, equality_constraints, tol=1.-5)
        f = assemblage.phases[1].molar_fractions
        fper_compositions[i] = f
        fguess = f[1]/(1. - f[0])
        assemblage.set_state(1.e5, 300.)
        volumes[i] = assemblage.phases[1].V
        Fe3oversumFe[i] = 2./3.*f[2]/(f[1] + 2./3.*f[2])

    #plt.plot(xMgs, Fe3oversumFe, label='fper (Fe-sat; {0:.2f} GPa)'.format(P/1.e9))
    plt.plot(xMgs, volumes, label='fper (Fe-sat; {0:.2f} GPa)'.format(P/1.e9))
    #tax.plot(fper_compositions, linewidth=2.0, label='fper (Fe-sat; {0:.2f} GPa)'.format(P/1.e9))
    #tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)
    
#tax.get_axes().axis('off')
#tax.clear_matplotlib_ticks()
#tax.legend()
plt.show()


"""
def func_muO2_IW(temperatures):
    muO2 = []
    for T in temperatures:
        if (T < 833 and T < 1042.):
            muO2.append(-605568. + 1366.420*T - 182.7955*T*np.log(T) + 0.10359*T*T)
        elif T < 1184.:
            muO2.append(-519113. + 59.129*T + 8.9276*T*np.log(T))
        elif T < 1644.:
            muO2.append(-550915. + 269.106*T - 16.9484*T*np.log(T))
    return np.array(muO2)

temperatures = np.linspace(834, 1643., 11)
plt.plot(temperatures, func_muO2_IW(temperatures)/(np.log(10.)*burnman.constants.gas_constant*temperatures), label='O\'Neill')

assemblage = burnman.Composite([fcc_iron, parent_ss['fper']])
equality_constraints = [('P', 1.e5), ('T', temperatures)]
composition = {'Fe': 1., 'O': 1.}
parent_ss['fper'].guess = np.array([0., 0.9, 0.1])
sols, prm = equilibrate(composition, assemblage, equality_constraints, tol=1.-4, store_assemblage=True)
mu_O2 = np.array([4.*(1.5*sol.assemblage.phases[1].partial_gibbs[2] -
                   sol.assemblage.phases[1].partial_gibbs[1]) for sol in sols])



plt.plot(temperatures, (mu_O2 - O2.evaluate(['gibbs'], temperatures*0. + 1.e5, temperatures)[0])/(np.log(10.)*burnman.constants.gas_constant*temperatures), label='model')


plt.legend()
plt.show()

exit()


func_muO2 = lambda T, X_MgO: -337.06e3 + 2*burnman.constants.gas_constant*T*np.log(1. - X_MgO) + 2.*X_MgO*X_MgO*(10.57e3 + 0.41e3*(3. - 4.*X_MgO))
XMgO, T, muO2 = np.array([[d[5], d[2], -4.*F*((d[3] + 1.1)*1.e-3) + burnman.constants.gas_constant*d[2]*np.log(0.20946*d[1]/1.e5)] for d in data]).T
plt.scatter(XMgO, muO2)
XMgOs = np.linspace(0., 0.9, 101)
plt.plot(XMgOs, func_muO2(1473., XMgOs))

temperatures = np.linspace(834, 1643., 101)
plt.plot(0.*temperatures, func_muO2_IW(temperatures)+150.)
plt.plot(0.*temperatures, func_muO2_IW(temperatures)-150.)
plt.scatter([0., 0., 0.], func_muO2_IW([1473.])[0] - np.array([-150., 0., 150.]))
plt.show()


exit()
mt = HP_2011_ds62.mt()
temperatures = np.linspace(300., 2000., 101)
plt.plot(temperatures, mt.evaluate(['S'], temperatures*0. + 1.e5, temperatures)[0])
plt.scatter([298.15, 500., 750., 1000., 1500., 1900.], [145.804, 233.661, 320.022, 388.757, 471.739, 520.680]) # S
plt.show()
exit()


fper = transform_solution_to_new_basis(fper, np.array([[0., 1., 0.],[0., 0., 1.]]), n_mbrs = None,
                                       solution_name=None, endmember_names=None,
                                       molar_fractions=None)
print(fper.endmembers)

"""


"""

# Let's first create instances of the minerals in our sample and provide compositional uncertainties
components = ['CaO', 'FeO', 'MgO', 'Al2O3', 'SiO2'] # Create a garnet model for a sodium-free system
garnet = feasible_solution_in_component_space(SLB_2011.garnet(), components)

garnet.fitted_elements = ['Mg', 'Ca', 'Al', 'Si', 'Fe']
garnet.composition = np.array([1.64, 1.5, 1.85, 3.05, 0.01])
garnet.compositional_uncertainties = np.array([0.1, 0.1, 0.1, 0.2, 0.01])

olivine = SLB_2011.mg_fe_olivine()
olivine.fitted_elements = ['Mg', 'Fe', 'Si']
olivine.composition = np.array([1.8, 0.2, 1.0])
olivine.compositional_uncertainties = np.array([0.1, 0.1, 0.1])

quartz = SLB_2011.quartz()

assemblage = burnman.Composite([garnet, olivine, quartz])
assemblage.nominal_state = (10.e9, 500.)
assemblage.state_covariances = np.diag(np.array([1.e9*1.e9, 10.*10.])) 

# Now let's prepare the assemblage by assigning covariances to each of the solid solutions
compute_and_set_phase_compositions(assemblage)

# Assign a state to the assemblage
assemblage.set_state(*assemblage.nominal_state)

# Calculate the misfit
print(assemblage_affinity_misfit(assemblage))




# We can do the same for an olivine-wadsleyite composition:

olivine = SLB_2011.mg_fe_olivine()
olivine.fitted_elements = ['Mg', 'Fe', 'Si']
olivine.composition = np.array([0.82, 0.18, 0.5])
olivine.compositional_uncertainties = np.array([0.05, 0.05, 0.05])

wadsleyite = SLB_2011.mg_fe_wadsleyite()
wadsleyite.fitted_elements = ['Mg', 'Fe', 'Si']
wadsleyite.composition = np.array([0.72, 0.28, 0.5])
wadsleyite.compositional_uncertainties = np.array([0.05, 0.05, 0.05])


assemblage = burnman.Composite([olivine, wadsleyite])
assemblage.nominal_state = (13.e9, 1673.)
assemblage.state_covariances = np.diag(np.array([1.e9*1.e9, 10.*10.])) 

# Now let's prepare the assemblage by assigning covariances to each of the solid solutions
compute_and_set_phase_compositions(assemblage)

# Assign a state to the assemblage
assemblage.set_state(*assemblage.nominal_state)

# Calculate the misfit
print(assemblage_affinity_misfit(assemblage))
"""
