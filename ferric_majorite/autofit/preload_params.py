from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import minimize, fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../burnman'):
    sys.path.insert(1, os.path.abspath('../..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit
from burnman.equilibrate import equilibrate


from input_dataset import * 



def set_params_from_special_constraints():
    # Destabilise fwd
    fa.set_state(6.25e9, 1673.15)
    frw.set_state(6.25e9, 1673.15)
    fwd.set_state(6.25e9, 1673.15)

    # First, determine the entropy which will give the fa-fwd reaction the same slope as the fa-frw reaction
    dPdT = (frw.S - fa.S)/(frw.V - fa.V) # = dS/dV
    dV = fwd.V - fa.V
    dS = dPdT*dV
    fwd.params['S_0'] += fa.S - fwd.S + dS
    fwd.params['H_0'] += frw.gibbs - fwd.gibbs + 100. # make fwd a little less stable than frw
    
def minimize_func(params, assemblages):
    # Set parameters 
    set_params(params)
    
    chisqr = []
    # Run through all assemblages for affinity misfit
    for i, assemblage in enumerate(assemblages):
        #print(i, assemblage.experiment_id, [phase.name for phase in assemblage.phases])
        # Assign compositions and uncertainties to solid solutions
        for j, phase in enumerate(assemblage.phases):
            if isinstance(phase, burnman.SolidSolution):
                molar_fractions, phase.molar_fraction_covariances = assemblage.stored_compositions[j]
                phase.set_composition(molar_fractions)
                
        # Assign a state to the assemblage
        P, T = np.array(assemblage.nominal_state)
        try:
             P += dict_experiment_uncertainties[assemblage.experiment_id]['P']
             T += dict_experiment_uncertainties[assemblage.experiment_id]['T']
        except:
            pass
        
        assemblage.set_state(P, T)
        
        
        # Calculate the misfit and store it
        assemblage.chisqr = assemblage_affinity_misfit(assemblage)
        #print('assemblage chisqr', assemblage.experiment_id, [phase.name for phase in assemblage.phases], assemblage.chisqr)
        chisqr.append(assemblage.chisqr)

    # Endmember priors
    for p in endmember_priors:
        c = np.power(((dict_endmember_args[p[0]][p[1]] - p[2])/p[3]), 2.)
        #print('endmember_prior', p[0], p[1], dict_endmember_args[p[0]][p[1]], p[2], p[3], c)
        chisqr.append(c)
        
    # Solution priors
    for p in solution_priors:
        c = np.power(((dict_solution_args[p[0]]['{0}{1}{2}'.format(p[1], p[2], p[3])] -
                       p[4])/p[5]), 2.)
        #print('solution_prior', c)
        chisqr.append(c)

    # Experiment uncertainties
    for u in experiment_uncertainties:
        c = np.power(u[2]/u[3], 2.)
        #print('pressure uncertainty', c)
        chisqr.append(c)

    rms_misfit = np.sqrt(np.sum(chisqr)/float(len(chisqr)))
    print(rms_misfit)
    string = '['
    for i, p in enumerate(params):
        string += '{0:.4f}'.format(p)
        if i < len(params) - 1:
            string += ', '
    string+=']'
    print(string)
    return rms_misfit


# Component defining endmembers (for H_0 and S_0) are:
# Fe: Fe metal (BCC, FCC, HCP)
# O: O2
# Mg: MgO per
# Si: SiO2 qtz
# Al: Mg3Al2Si3O12 pyrope
# Ca: CaMgSi2O6 diopside
# Na: NaAlSi2O6 jadeite

endmember_args = [['wus', 'H_0', wus.params['H_0'], 1.e3], 
                  ['fo',  'H_0', fo.params['H_0'],  1.e3],
                  ['fa',  'H_0', fa.params['H_0'],  1.e3],
                  ['mwd', 'H_0', mwd.params['H_0'], 1.e3], # fwd H_0 and S_0 obtained in set_params_from_special_constraints()
                  ['mrw', 'H_0', mrw.params['H_0'], 1.e3],
                  ['frw', 'H_0', frw.params['H_0'], 1.e3],
                  ['alm', 'H_0', alm.params['H_0'], 1.e3],
                  ['coe', 'H_0', coe.params['H_0'], 1.e3],
                  ['stv', 'H_0', stv.params['H_0'], 1.e3],
                  ['oen', 'H_0', oen.params['H_0'], 1.e3],
                  ['ofs', 'H_0', ofs.params['H_0'], 1.e3],
                  ['hen', 'H_0', hen.params['H_0'], 1.e3],
                  ['hfs', 'H_0', hfs.params['H_0'], 1.e3],
                  ['per', 'S_0', per.params['S_0'], 1.], # has a prior associated with it, so can be inverted
                  ['wus', 'S_0', wus.params['S_0'], 1.],
                  ['fo',  'S_0', fo.params['S_0'],  1.],
                  ['fa',  'S_0', fa.params['S_0'],  1.],
                  ['mwd', 'S_0', mwd.params['S_0'], 1.], 
                  ['mrw', 'S_0', mrw.params['S_0'], 1.],
                  ['frw', 'S_0', frw.params['S_0'], 1.],
                  ['alm', 'S_0', alm.params['S_0'], 1.],
                  ['oen', 'S_0', oen.params['S_0'], 1.],
                  ['ofs', 'S_0', ofs.params['S_0'], 1.],
                  ['hen', 'S_0', hen.params['S_0'], 1.],
                  ['hfs', 'S_0', hfs.params['S_0'], 1.],
                  ['coe', 'S_0', coe.params['S_0'], 1.],
                  ['stv', 'S_0', stv.params['S_0'], 1.],
                  ['fwd', 'V_0', fwd.params['V_0'], 1.e-5],
                  ['fwd', 'K_0', fwd.params['K_0'], 1.e11],
                  ['frw', 'K_0', frw.params['K_0'], 1.e11],
                  ['per', 'a_0', per.params['a_0'], 1.e-5],
                  ['wus', 'a_0', wus.params['a_0'], 1.e-5],
                  ['fo',  'a_0', fo.params['a_0'],  1.e-5],
                  ['fa',  'a_0', fa.params['a_0'],  1.e-5],
                  ['mwd', 'a_0', mwd.params['a_0'], 1.e-5],
                  ['fwd', 'a_0', fwd.params['a_0'], 1.e-5],
                  ['mrw', 'a_0', mrw.params['a_0'], 1.e-5],
                  ['frw', 'a_0', frw.params['a_0'], 1.e-5]]

solution_args = [['mw', 'E', 0, 0, fper.energy_interaction[0][0], 1.e3],
                 ['ol', 'E', 0, 0, ol.energy_interaction[0][0], 1.e3],
                 ['wad', 'E', 0, 0, wad.energy_interaction[0][0], 1.e3],
                 ['sp', 'E', 3, 0, spinel.energy_interaction[3][0], 1.e3], # mrw-frw
                 ['opx', 'E', 0, 0, opx_od.energy_interaction[0][0], 1.e3]]

# ['gt', 'E', 0, 0, gt.energy_interaction[0][0], 1.e3] # py-alm interaction fixed as ideal

endmember_priors = [['per', 'S_0', per.params['S_0_orig'][0], per.params['S_0_orig'][1]],
                    ['wus', 'S_0', wus.params['S_0_orig'][0], wus.params['S_0_orig'][1]],
                    ['fo',  'S_0', fo.params['S_0_orig'][0],  fo.params['S_0_orig'][1]],
                    ['fa',  'S_0', fa.params['S_0_orig'][0],  fa.params['S_0_orig'][1]],
                    ['mwd', 'S_0', mwd.params['S_0_orig'][0], mwd.params['S_0_orig'][1]], #['fwd', 'S_0', fwd.params['S_0_orig'][0], fwd.params['S_0_orig'][1]],
                    ['mrw', 'S_0', mrw.params['S_0_orig'][0], mrw.params['S_0_orig'][1]],
                    ['frw', 'S_0', frw.params['S_0_orig'][0], frw.params['S_0_orig'][1]],
                    ['alm', 'S_0', alm.params['S_0_orig'][0], alm.params['S_0_orig'][1]],
                    ['oen', 'S_0', oen.params['S_0_orig'][0], oen.params['S_0_orig'][1]],
                    ['ofs', 'S_0', ofs.params['S_0_orig'][0], ofs.params['S_0_orig'][1]],
                    ['fwd', 'V_0', fwd.params['V_0'], 2.15e-7], # 0.5% uncertainty, somewhat arbitrary
                    ['fwd', 'K_0', fwd.params['K_0'], fwd.params['K_0']/100.*2.], # 2% uncertainty, somewhat arbitrary
                    ['frw', 'K_0', frw.params['K_0'], frw.params['K_0']/100.*0.5], # 0.5% uncertainty, somewhat arbitrary
                    ['per', 'a_0', per.params['a_0_orig'], 2.e-7],
                    ['wus', 'a_0', wus.params['a_0_orig'], 5.e-7],
                    ['fo',  'a_0', fo.params['a_0_orig'], 2.e-7],
                    ['fa',  'a_0', fa.params['a_0_orig'], 2.e-7],
                    ['mwd', 'a_0', mwd.params['a_0_orig'], 5.e-7],
                    ['fwd', 'a_0', fwd.params['a_0_orig'], 20.e-7],
                    ['mrw', 'a_0', mrw.params['a_0_orig'], 2.e-7],
                    ['frw', 'a_0', frw.params['a_0_orig'], 5.e-7]]

solution_priors = [] # ['gt', 'E', 0, 0, 0.3e3, 0.4e3]

experiment_uncertainties = [['49Fe', 'P', 0., 0.5e9],
                            ['50Fe', 'P', 0., 0.5e9],
                            ['61Fe', 'P', 0., 0.5e9],
                            ['62Fe', 'P', 0., 0.5e9],
                            ['63Fe', 'P', 0., 0.5e9],
                            ['64Fe', 'P', 0., 0.5e9],
                            ['66Fe', 'P', 0., 0.5e9],
                            ['67Fe', 'P', 0., 0.5e9],
                            ['68Fe', 'P', 0., 0.5e9],
                            ['V189', 'P', 0., 0.5e9],
                            ['V191', 'P', 0., 0.5e9],
                            ['V192', 'P', 0., 0.5e9],
                            ['V200', 'P', 0., 0.5e9],
                            ['V208', 'P', 0., 0.5e9],
                            ['V209', 'P', 0., 0.5e9],
                            ['V212', 'P', 0., 0.5e9],
                            ['V217', 'P', 0., 0.5e9],
                            ['V220', 'P', 0., 0.5e9],
                            ['V223', 'P', 0., 0.5e9],
                            ['V227', 'P', 0., 0.5e9],
                            ['V229', 'P', 0., 0.5e9],
                            ['V252', 'P', 0., 0.5e9],
                            ['V254', 'P', 0., 0.5e9]]  


# Make dictionaries
dict_endmember_args = {a[0]: {} for a in endmember_args}
for a in endmember_args:
    dict_endmember_args[a[0]][a[1]] = a[2]

dict_solution_args = {a[0]: {} for a in solution_args}
for a in solution_args:
    dict_solution_args[a[0]]['{0}{1}{2}'.format(a[1], a[2], a[3])] = a[4]

dict_experiment_uncertainties = {u[0] : {'P': 0., 'T': 0.} for u in experiment_uncertainties}
for u in experiment_uncertainties:
    dict_experiment_uncertainties[u[0]][u[1]] = u[2]


def get_params():
    """
    This function gets the parameters from the parameter lists
    """
    
    # Endmember parameters
    args = [a[2]/a[3] for a in endmember_args]

    # Solution parameters
    args.extend([a[4]/a[5] for a in solution_args])

    # Experimental uncertainties
    args.extend([u[2]/u[3] for u in experiment_uncertainties])
    return args

def set_params(args):
    """
    This function sets the parameters *both* in the parameter lists, 
    the parameter dictionaries and also in the minerals / solutions.
    """
    
    i=0

    # Endmember parameters
    for j, a in enumerate(endmember_args):
        dict_endmember_args[a[0]][a[1]] = args[i]*a[3]
        endmember_args[j][2] = args[i]*a[3]
        endmembers[a[0]].params[a[1]] = args[i]*a[3]
        i+=1

    # Solution parameters
    for j, a in enumerate(solution_args):
        dict_solution_args[a[0]][a[1]] = args[i]*a[5]
        solution_args[j][4] = args[i]*a[5]
        if a[1] == 'E':
            solutions[a[0]].energy_interaction[int(a[2])][int(a[3])] = args[i]*a[5]
        else:
            raise Exception('Not implemented')
        i+=1

    # Reinitialize solutions
    for name in solutions:
        burnman.SolidSolution.__init__(solutions[name])

    # Reset dictionary of child solutions
    for k, ss in child_solutions.items():
        ss.__dict__.update(transform_solution_to_new_basis(ss.parent,
                                                           ss.basis).__dict__)

    # Experimental uncertainties
    for j, u in enumerate(experiment_uncertainties):
        dict_experiment_uncertainties[u[0]][u[1]] = args[i]*u[3]
        experiment_uncertainties[j][2] = args[i]*u[3]
        i+=1

    # Special one-off constraints
    set_params_from_special_constraints()
    return None


#######################
### PUT PARAMS HERE ###
#######################

set_params([-269.0792, -2173.4836, -1476.9122, -2145.7575, -2135.5996, -1472.7334, -5258.2888, -907.0115, -867.3844, -3089.7563, -2389.8787, -3082.4015, -2392.1905, 26.8807, 55.5686, 94.2084, 151.4062, 85.4220, 81.7165, 135.8285, 340.6464, 132.3161, 188.4748, 131.4391, 177.9551, 39.6230, 29.5232, 4.3053, 1.6102, 2.0093, 3.0847, 3.4498, 2.8015, 2.7330, 2.0997, 1.8693, 2.1880, 2.1287, 11.3716, 5.8810, 17.4960, 8.6782, -1.4860, -1.4663, -0.3108, -3.0403, -0.4191, 1.0919, -4.2236, -1.5626, -0.4069, -1.1745, 0.1191, -0.0074, 1.0992, 0.1703, 1.5458, -1.1353, -1.0090, -0.9139, -0.6605, -0.1253, 0.2582, -0.0373, -0.1852, 0.8344])


def set_params_from_special_constraints():
    # Destabilise fwd
    fa.set_state(6.25e9, 1673.15)
    frw.set_state(6.25e9, 1673.15)
    fwd.set_state(6.25e9, 1673.15)

    # First, determine the entropy which will give the fa-fwd reaction the same slope as the fa-frw reaction
    dPdT = (frw.S - fa.S)/(frw.V - fa.V) # = dS/dV
    dV = fwd.V - fa.V
    dS = dPdT*dV
    fwd.params['S_0'] += fa.S - fwd.S + dS
    fwd.params['H_0'] += frw.gibbs - fwd.gibbs + 100. # make fwd a little less stable than frw
    
def minimize_func(params, assemblages):
    # Set parameters 
    set_params(params)
    
    chisqr = []
    # Run through all assemblages for affinity misfit
    for i, assemblage in enumerate(assemblages):
        #print(i, assemblage.experiment_id, [phase.name for phase in assemblage.phases])
        # Assign compositions and uncertainties to solid solutions
        for j, phase in enumerate(assemblage.phases):
            if isinstance(phase, burnman.SolidSolution):
                molar_fractions, phase.molar_fraction_covariances = assemblage.stored_compositions[j]
                phase.set_composition(molar_fractions)
                
        # Assign a state to the assemblage
        P, T = np.array(assemblage.nominal_state)
        try:
             P += dict_experiment_uncertainties[assemblage.experiment_id]['P']
             T += dict_experiment_uncertainties[assemblage.experiment_id]['T']
        except:
            pass
        
        assemblage.set_state(P, T)
        
        
        # Calculate the misfit and store it
        assemblage.chisqr = assemblage_affinity_misfit(assemblage)
        #print('assemblage chisqr', assemblage.experiment_id, [phase.name for phase in assemblage.phases], assemblage.chisqr)
        chisqr.append(assemblage.chisqr)

    # Endmember priors
    for p in endmember_priors:
        c = np.power(((dict_endmember_args[p[0]][p[1]] - p[2])/p[3]), 2.)
        #print('endmember_prior', p[0], p[1], dict_endmember_args[p[0]][p[1]], p[2], p[3], c)
        chisqr.append(c)
        
    # Solution priors
    for p in solution_priors:
        c = np.power(((dict_solution_args[p[0]]['{0}{1}{2}'.format(p[1], p[2], p[3])] -
                       p[4])/p[5]), 2.)
        #print('solution_prior', c)
        chisqr.append(c)

    # Experiment uncertainties
    for u in experiment_uncertainties:
        c = np.power(u[2]/u[3], 2.)
        #print('pressure uncertainty', c)
        chisqr.append(c)

    rms_misfit = np.sqrt(np.sum(chisqr)/float(len(chisqr)))
    print(rms_misfit)
    string = '['
    for i, p in enumerate(params):
        string += '{0:.4f}'.format(p)
        if i < len(params) - 1:
            string += ', '
    string+=']'
    print(string)
    return rms_misfit


# Component defining endmembers (for H_0 and S_0) are:
# Fe: Fe metal (BCC, FCC, HCP)
# O: O2
# Mg: MgO per
# Si: SiO2 qtz
# Al: Mg3Al2Si3O12 pyrope
# Ca: CaMgSi2O6 diopside
# Na: NaAlSi2O6 jadeite

endmember_args = [['wus', 'H_0', wus.params['H_0'], 1.e3], 
                  ['fo',  'H_0', fo.params['H_0'],  1.e3],
                  ['fa',  'H_0', fa.params['H_0'],  1.e3],
                  ['mwd', 'H_0', mwd.params['H_0'], 1.e3], # fwd H_0 and S_0 obtained in set_params_from_special_constraints()
                  ['mrw', 'H_0', mrw.params['H_0'], 1.e3],
                  ['frw', 'H_0', frw.params['H_0'], 1.e3],
                  ['alm', 'H_0', alm.params['H_0'], 1.e3],
                  ['coe', 'H_0', coe.params['H_0'], 1.e3],
                  ['stv', 'H_0', stv.params['H_0'], 1.e3],
                  ['oen', 'H_0', oen.params['H_0'], 1.e3],
                  ['ofs', 'H_0', ofs.params['H_0'], 1.e3],
                  ['hen', 'H_0', hen.params['H_0'], 1.e3],
                  ['hfs', 'H_0', hfs.params['H_0'], 1.e3],
                  ['mbdg', 'H_0', mbdg.params['H_0'], 1.e3],
                  ['fbdg', 'H_0', fbdg.params['H_0'], 1.e3],
                  ['per', 'S_0', per.params['S_0'], 1.], # has a prior associated with it, so can be inverted
                  ['wus', 'S_0', wus.params['S_0'], 1.],
                  ['fo',  'S_0', fo.params['S_0'],  1.],
                  ['fa',  'S_0', fa.params['S_0'],  1.],
                  ['mwd', 'S_0', mwd.params['S_0'], 1.], 
                  ['mrw', 'S_0', mrw.params['S_0'], 1.],
                  ['frw', 'S_0', frw.params['S_0'], 1.],
                  ['alm', 'S_0', alm.params['S_0'], 1.],
                  ['oen', 'S_0', oen.params['S_0'], 1.],
                  ['ofs', 'S_0', ofs.params['S_0'], 1.],
                  ['hen', 'S_0', hen.params['S_0'], 1.],
                  ['hfs', 'S_0', hfs.params['S_0'], 1.],
                  ['coe', 'S_0', coe.params['S_0'], 1.],
                  ['stv', 'S_0', stv.params['S_0'], 1.],
                  ['mbdg', 'S_0', mbdg.params['S_0'], 1.],
                  ['fbdg', 'S_0', fbdg.params['S_0'], 1.],
                  ['fwd', 'V_0', fwd.params['V_0'], 1.e-5],
                  ['wus', 'K_0', wus.params['K_0'], 1.e11],
                  ['fwd', 'K_0', fwd.params['K_0'], 1.e11],
                  ['frw', 'K_0', frw.params['K_0'], 1.e11],
                  ['per', 'a_0', per.params['a_0'], 1.e-5],
                  ['wus', 'a_0', wus.params['a_0'], 1.e-5],
                  ['fo',  'a_0', fo.params['a_0'],  1.e-5],
                  ['fa',  'a_0', fa.params['a_0'],  1.e-5],
                  ['mwd', 'a_0', mwd.params['a_0'], 1.e-5],
                  ['fwd', 'a_0', fwd.params['a_0'], 1.e-5],
                  ['mrw', 'a_0', mrw.params['a_0'], 1.e-5],
                  ['frw', 'a_0', frw.params['a_0'], 1.e-5],
                  ['mbdg', 'a_0', mbdg.params['a_0'], 1.e-5],
                  ['fbdg', 'a_0', fbdg.params['a_0'], 1.e-5]]

solution_args = [['mw', 'E', 0, 0, fper.energy_interaction[0][0], 1.e3],
                 ['ol', 'E', 0, 0, ol.energy_interaction[0][0], 1.e3],
                 ['wad', 'E', 0, 0, wad.energy_interaction[0][0], 1.e3],
                 ['sp', 'E', 3, 0, spinel.energy_interaction[3][0], 1.e3], # mrw-frw
                 ['opx', 'E', 0, 0, opx_od.energy_interaction[0][0], 1.e3]] #, ['bdg', 'E', 0, 0, bdg.energy_interaction[0][0], 1.e3]]

bdg.energy_interaction[0][0] = 0. # make bdg ideal

# ['gt', 'E', 0, 0, gt.energy_interaction[0][0], 1.e3] # py-alm interaction fixed as ideal

endmember_priors = [['per', 'S_0', per.params['S_0_orig'][0], per.params['S_0_orig'][1]],
                    ['wus', 'S_0', wus.params['S_0_orig'][0], wus.params['S_0_orig'][1]],
                    ['fo',  'S_0', fo.params['S_0_orig'][0],  fo.params['S_0_orig'][1]],
                    ['fa',  'S_0', fa.params['S_0_orig'][0],  fa.params['S_0_orig'][1]],
                    ['mwd', 'S_0', mwd.params['S_0_orig'][0], mwd.params['S_0_orig'][1]], #['fwd', 'S_0', fwd.params['S_0_orig'][0], fwd.params['S_0_orig'][1]],
                    ['mrw', 'S_0', mrw.params['S_0_orig'][0], mrw.params['S_0_orig'][1]],
                    ['frw', 'S_0', frw.params['S_0_orig'][0], frw.params['S_0_orig'][1]],
                    ['alm', 'S_0', alm.params['S_0_orig'][0], alm.params['S_0_orig'][1]],
                    ['oen', 'S_0', oen.params['S_0_orig'][0], oen.params['S_0_orig'][1]],
                    ['ofs', 'S_0', ofs.params['S_0_orig'][0], ofs.params['S_0_orig'][1]],
                    ['mbdg', 'S_0', mbdg.params['S_0_orig'][0], mbdg.params['S_0_orig'][1]],
                    ['fbdg', 'S_0', fbdg.params['S_0_orig'][0], fbdg.params['S_0_orig'][1]],
                    ['fwd', 'V_0', fwd.params['V_0'], 2.15e-7], # 0.5% uncertainty, somewhat arbitrary
                    ['fwd', 'K_0', fwd.params['K_0'], fwd.params['K_0']/100.*2.], # 2% uncertainty, somewhat arbitrary
                    ['frw', 'K_0', frw.params['K_0'], frw.params['K_0']/100.*0.5], # 0.5% uncertainty, somewhat arbitrary
                    ['wus', 'K_0', wus.params['K_0'], wus.params['K_0']/100.*2.],
                    ['per', 'a_0', per.params['a_0_orig'], 2.e-7],
                    ['wus', 'a_0', wus.params['a_0_orig'], 5.e-7],
                    ['fo',  'a_0', fo.params['a_0_orig'], 2.e-7],
                    ['fa',  'a_0', fa.params['a_0_orig'], 2.e-7],
                    ['mwd', 'a_0', mwd.params['a_0_orig'], 5.e-7],
                    ['fwd', 'a_0', fwd.params['a_0_orig'], 20.e-7],
                    ['mrw', 'a_0', mrw.params['a_0_orig'], 2.e-7],
                    ['frw', 'a_0', frw.params['a_0_orig'], 5.e-7],
                    ['mbdg', 'a_0', mbdg.params['a_0_orig'], 2.e-7],
                    ['fbdg', 'a_0', fbdg.params['a_0_orig'], 5.e-7]]

solution_priors = [['opx', 'E', 0, 0, 0.e3, 1.e3]] #, ['bdg', 'E', 0, 0, 4.e3, 0.0001e3]] # ['gt', 'E', 0, 0, 0.3e3, 0.4e3]

# Frost 2003 uncertainties already in preload
"""
experiment_uncertainties = [['49Fe', 'P', 0., 0.5e9],
                            ['50Fe', 'P', 0., 0.5e9],
                            ['61Fe', 'P', 0., 0.5e9],
                            ['62Fe', 'P', 0., 0.5e9],
                            ['63Fe', 'P', 0., 0.5e9],
                            ['64Fe', 'P', 0., 0.5e9],
                            ['66Fe', 'P', 0., 0.5e9],
                            ['67Fe', 'P', 0., 0.5e9],
                            ['68Fe', 'P', 0., 0.5e9],
                            ['V189', 'P', 0., 0.5e9],
                            ['V191', 'P', 0., 0.5e9],
                            ['V192', 'P', 0., 0.5e9],
                            ['V200', 'P', 0., 0.5e9],
                            ['V208', 'P', 0., 0.5e9],
                            ['V209', 'P', 0., 0.5e9],
                            ['V212', 'P', 0., 0.5e9],
                            ['V217', 'P', 0., 0.5e9],
                            ['V220', 'P', 0., 0.5e9],
                            ['V223', 'P', 0., 0.5e9],
                            ['V227', 'P', 0., 0.5e9],
                            ['V229', 'P', 0., 0.5e9],
                            ['V252', 'P', 0., 0.5e9],
                            ['V254', 'P', 0., 0.5e9]]  
"""

# Make dictionaries
dict_endmember_args = {a[0]: {} for a in endmember_args}
for a in endmember_args:
    dict_endmember_args[a[0]][a[1]] = a[2]

dict_solution_args = {a[0]: {} for a in solution_args}
for a in solution_args:
    dict_solution_args[a[0]]['{0}{1}{2}'.format(a[1], a[2], a[3])] = a[4]

dict_experiment_uncertainties = {u[0] : {'P': 0., 'T': 0.} for u in experiment_uncertainties}
for u in experiment_uncertainties:
    dict_experiment_uncertainties[u[0]][u[1]] = u[2]


def get_params():
    """
    This function gets the parameters from the parameter lists
    """
    
    # Endmember parameters
    args = [a[2]/a[3] for a in endmember_args]

    # Solution parameters
    args.extend([a[4]/a[5] for a in solution_args])

    # Experimental uncertainties
    args.extend([u[2]/u[3] for u in experiment_uncertainties])
    return args

def set_params(args):
    """
    This function sets the parameters *both* in the parameter lists, 
    the parameter dictionaries and also in the minerals / solutions.
    """
    
    i=0

    # Endmember parameters
    for j, a in enumerate(endmember_args):
        dict_endmember_args[a[0]][a[1]] = args[i]*a[3]
        endmember_args[j][2] = args[i]*a[3]
        endmembers[a[0]].params[a[1]] = args[i]*a[3]
        i+=1

    # Solution parameters
    for j, a in enumerate(solution_args):
        dict_solution_args[a[0]][a[1]] = args[i]*a[5]
        solution_args[j][4] = args[i]*a[5]
        if a[1] == 'E':
            solutions[a[0]].energy_interaction[int(a[2])][int(a[3])] = args[i]*a[5]
        else:
            raise Exception('Not implemented')
        i+=1

    # Reinitialize solutions
    for name in solutions:
        burnman.SolidSolution.__init__(solutions[name])

    # Reset dictionary of child solutions
    for k, ss in child_solutions.items():
        ss.__dict__.update(transform_solution_to_new_basis(ss.parent,
                                                           ss.basis).__dict__)

    # Experimental uncertainties
    for j, u in enumerate(experiment_uncertainties):
        dict_experiment_uncertainties[u[0]][u[1]] = args[i]*u[3]
        experiment_uncertainties[j][2] = args[i]*u[3]
        i+=1

    # Special one-off constraints
    set_params_from_special_constraints()
    return None

"""
#######################
# EXPERIMENTAL DATA ###
#######################

from Frost_2003_fper_ol_wad_rw import Frost_2003_assemblages
from Seckendorff_ONeill_1992_ol_opx import Seckendorff_ONeill_1992_assemblages
from ONeill_Wood_1979_ol_gt import ONeill_Wood_1979_assemblages
from endmember_reactions import endmember_reaction_assemblages
from Matsuzaka_et_al_2000_rw_wus_stv import Matsuzaka_2000_assemblages
from ONeill_1987_QFI import ONeill_1987_QFI_assemblages
from ONeill_1987_QFM import ONeill_1987_QFM_assemblages
from Nakajima_FR_2012_bdg_fper import Nakajima_FR_2012_assemblages
from Tange_TNFS_2009_bdg_fper_stv import Tange_TNFS_2009_FMS_assemblages

assemblages = [assemblage for assemblage_list in
               [endmember_reaction_assemblages,
                ONeill_1987_QFI_assemblages,
                ONeill_1987_QFM_assemblages,
                Frost_2003_assemblages,
                Seckendorff_ONeill_1992_assemblages,
                Matsuzaka_2000_assemblages,
                ONeill_Wood_1979_assemblages,
                Nakajima_FR_2012_assemblages,
                Tange_TNFS_2009_FMS_assemblages
               ]
               for assemblage in assemblage_list]


#minimize_func(get_params(), assemblages)
"""
#######################
### PUT PARAMS HERE ###
#######################

set_params([-268.0442, -2178.6955, -1477.0452, -2150.6947, -2140.1438, -1472.0362, -5250.7151, -906.8840, -867.8431, -3094.8586, -2390.1721, -3087.5124, -2392.5608, -1452.0700, -1090.8899, 27.2372, 54.9153, 94.1400, 151.3821, 85.4803, 82.2628, 136.3741, 340.5298, 132.3746, 188.3640, 131.5150, 177.7684, 39.7231, 30.6284, 58.3353, 80.3775, 4.3107, 1.6303, 2.0090, 3.0832, 3.3455, 2.8282, 2.7540, 2.1164, 1.8818, 2.2305, 2.1249, 1.7734, 1.9860, 11.5124, 5.9581, 16.8773, 8.2889, -1.4756, -25.5141, -1.5190, -0.3524, -3.2744, -0.4767, 1.4022, -5.0352, -2.7255, -0.3394, -0.9261, -0.1019, -0.0404, 1.1515, 0.7817, 1.5014, -1.2049, -1.3150, -1.0586, -0.7263, -0.2265, 0.1982, -0.1128, -0.1812, 0.7734])

set_params([-267.8200, -2178.6237, -1477.0371, -2150.6945, -2140.1790, -1471.9490, -5250.7543, -906.9012, -868.2552, -3094.8330, -2390.1865, -3087.4807, -2392.5445, -1452.2421, -1091.1665, 27.1463, 54.9759, 94.1524, 151.3822, 85.4305, 82.1752, 136.3832, 340.5746, 132.3665, 188.3472, 131.4926, 177.7735, 39.7114, 30.4026, 58.8860, 80.0311, 4.3118, 1.6299, 2.0115, 3.0674, 3.3426, 2.8244, 2.7620, 2.1074, 1.8493, 2.2189, 2.1199, 1.8405, 1.9631, 11.4721, 5.8873, 16.8619, 8.1646, -1.5297, -25.4313, -1.5369, -0.3540, -3.2841, -0.4786, 1.3961, -5.0158, -2.7011, -0.3406, -0.9397, -0.1212, -0.0447, 1.1576, 0.8177, 1.4803, -1.2283, -1.2980, -1.0745, -0.7532, -0.2526, 0.1735, -0.1325, -0.1820, 0.7494])

set_params([-266.8431, -2179.0130, -1477.0488, -2151.2655, -2140.7418, -1472.1147, -5250.4323, -906.9052, -872.5942, -3095.1946, -2390.1990, -3087.8428, -2392.5944, -1453.0053, -1101.3330, 27.2568, 55.5901, 94.1365, 151.3736, 85.2703, 82.0871, 136.2582, 340.3913, 132.3626, 188.3453, 131.4882, 177.7443, 39.7080, 27.7238, 58.2724, 75.3247, 4.3100, 1.6280, 2.0096, 3.0515, 3.3622, 2.8224, 2.7621, 2.1043, 1.8699, 2.2288, 2.1235, 1.8616, 1.9386, 11.5374, 5.9812, 17.1358, 8.3994, -1.4530, -23.6992, -1.3844, -0.2943, -3.1397, -0.3889, 1.2305, -4.5620, -2.0981, -0.3553, -1.0106, -0.0393, -0.0540, 1.1791, 0.6873, 1.5397, -1.1768, -1.1759, -1.0353, -0.7023, -0.1885, 0.2296, -0.0748, -0.1801, 0.8118])
# good, but large negative bdg interaction.


set_params([-266.9280, -2179.0150, -1477.1346, -2151.2889, -2140.7632, -1472.2053, -5250.2603, -906.9076, -872.5462, -3095.1984, -2390.1578, -3087.8521, -2392.5539, -1452.2194, -1102.5612, 27.4942, 55.5236, 94.2178, 151.2789, 85.2102, 82.2162, 136.4024, 340.1319, 132.3650, 188.2965, 131.4978, 177.6900, 39.7035, 27.4697, 57.0282, 77.6051, 4.3171, 1.7668, 1.6320, 2.0320, 3.0760, 3.2156, 2.8187, 2.8053, 2.0878, 1.9048, 2.2469, 2.2447, 1.7781, 1.5525, 11.6162, 6.3165, 17.1619, 8.4960, -1.3954, -1.3472, -0.2853, -3.1367, -0.3724, 1.2360, -4.5527, -2.1059, -0.3494, -0.9964, -0.0124, -0.0531, 1.1841, 0.6855, 1.6175, -1.0846, -1.0501, -1.0487, -0.6285, -0.1450, 0.3185, 0.0219, -0.1783, 0.9140])
# ideal mg-fe bdg
