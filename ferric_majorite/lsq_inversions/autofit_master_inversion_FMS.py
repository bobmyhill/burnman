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
from preload_params import *


if len(sys.argv) == 2:
    if sys.argv[1] == '--fit':
        run_inversion = True
        print('Running inversion')
    else:
        run_inversion = False
        print('Not running inversion. Use --fit as command line argument to invert parameters')
else:
    run_inversion = False
    print('Not running inversion. Use --fit as command line argument to invert parameters')


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

#######################
### PUT PARAMS HERE ###
#######################

set_params([-268.0442, -2178.6955, -1477.0452, -2150.6947, -2140.1438, -1472.0362, -5250.7151, -906.8840, -867.8431, -3094.8586, -2390.1721, -3087.5124, -2392.5608, -1452.0700, -1090.8899, 27.2372, 54.9153, 94.1400, 151.3821, 85.4803, 82.2628, 136.3741, 340.5298, 132.3746, 188.3640, 131.5150, 177.7684, 39.7231, 30.6284, 58.3353, 80.3775, 4.3107, 1.6303, 2.0090, 3.0832, 3.3455, 2.8282, 2.7540, 2.1164, 1.8818, 2.2305, 2.1249, 1.7734, 1.9860, 11.5124, 5.9581, 16.8773, 8.2889, -1.4756, -25.5141, -1.5190, -0.3524, -3.2744, -0.4767, 1.4022, -5.0352, -2.7255, -0.3394, -0.9261, -0.1019, -0.0404, 1.1515, 0.7817, 1.5014, -1.2049, -1.3150, -1.0586, -0.7263, -0.2265, 0.1982, -0.1128, -0.1812, 0.7734])

set_params([-267.8200, -2178.6237, -1477.0371, -2150.6945, -2140.1790, -1471.9490, -5250.7543, -906.9012, -868.2552, -3094.8330, -2390.1865, -3087.4807, -2392.5445, -1452.2421, -1091.1665, 27.1463, 54.9759, 94.1524, 151.3822, 85.4305, 82.1752, 136.3832, 340.5746, 132.3665, 188.3472, 131.4926, 177.7735, 39.7114, 30.4026, 58.8860, 80.0311, 4.3118, 1.6299, 2.0115, 3.0674, 3.3426, 2.8244, 2.7620, 2.1074, 1.8493, 2.2189, 2.1199, 1.8405, 1.9631, 11.4721, 5.8873, 16.8619, 8.1646, -1.5297, -25.4313, -1.5369, -0.3540, -3.2841, -0.4786, 1.3961, -5.0158, -2.7011, -0.3406, -0.9397, -0.1212, -0.0447, 1.1576, 0.8177, 1.4803, -1.2283, -1.2980, -1.0745, -0.7532, -0.2526, 0.1735, -0.1325, -0.1820, 0.7494])

set_params([-266.8431, -2179.0130, -1477.0488, -2151.2655, -2140.7418, -1472.1147, -5250.4323, -906.9052, -872.5942, -3095.1946, -2390.1990, -3087.8428, -2392.5944, -1453.0053, -1101.3330, 27.2568, 55.5901, 94.1365, 151.3736, 85.2703, 82.0871, 136.2582, 340.3913, 132.3626, 188.3453, 131.4882, 177.7443, 39.7080, 27.7238, 58.2724, 75.3247, 4.3100, 1.6280, 2.0096, 3.0515, 3.3622, 2.8224, 2.7621, 2.1043, 1.8699, 2.2288, 2.1235, 1.8616, 1.9386, 11.5374, 5.9812, 17.1358, 8.3994, -1.4530, -23.6992, -1.3844, -0.2943, -3.1397, -0.3889, 1.2305, -4.5620, -2.0981, -0.3553, -1.0106, -0.0393, -0.0540, 1.1791, 0.6873, 1.5397, -1.1768, -1.1759, -1.0353, -0.7023, -0.1885, 0.2296, -0.0748, -0.1801, 0.8118])
# good, but large negative bdg interaction.


set_params([-266.9161, -2179.0077, -1477.1425, -2151.2877, -2140.7594, -1472.2104, -5250.2553, -906.9080, -872.5526, -3095.1980, -2390.1515, -3087.8512, -2392.5574, -1452.2167, -1102.5466, 27.5308, 55.5112, 94.2128, 151.2863, 85.2069, 82.2150, 136.4079, 340.1255, 132.3654, 188.2892, 131.4958, 177.6948, 39.7045, 27.4847, 57.0120, 77.5785, 4.3185, 1.7593, 1.6219, 2.0216, 3.0612, 3.1691, 2.8199, 2.8124, 2.0903, 1.9060, 2.2503, 2.2513, 1.7909, 1.5437, 11.5941, 6.3214, 17.1623, 8.4940, -1.4002, -1.3476, -0.2854, -3.1367, -0.3725, 1.2365, -4.5535, -2.1066, -0.3494, -0.9961, -0.0127, -0.0531, 1.1841, 0.6849, 1.6172, -1.0845, -1.0535, -1.0486, -0.6308, -0.1474, 0.3205, 0.0213, -0.1783, 0.9166])
# ideal mg-fe bdg



#set_params([-267.8512, -2178.6069, -1477.0829, -2150.6708, -2140.2349, -1471.9591, -5250.6822, -906.8995, -868.2233, -3094.8392, -2390.1602, -3087.4893, -2392.5304, -1451.9595, -1091.5808, 27.2253, 54.9698, 94.1450, 151.3684, 85.4040, 82.2616, 136.3898, 340.4690, 132.3720, 188.3162, 131.5085, 177.7545, 39.7108, 30.2678, 58.3866, 80.7892, 4.3204, 1.6261, 2.0154, 3.0691, 3.4785, 2.8023, 2.8169, 2.0793, 1.8169, 2.2238, 2.1939, 1.8913, 0.9421, 11.5717, 6.0154, 16.8617, 8.1622, -1.5238, 5.4925, -1.5338, -0.3530, -3.2840, -0.4764, 1.3988, -5.0159, -2.7025, -0.3397, -0.9371, -0.1153, -0.0447, 1.1607, 0.8246, 1.5036, -1.2055, -1.2077, -1.0753, -0.7514, -0.2467, 0.1965, -0.1052, -0.1816, 0.7727])
# start for constrained bridgmanite interaction.


########################
# RUN THE MINIMIZATION #
########################
if run_inversion:
    print(minimize(minimize_func, get_params(), args=(assemblages), method='BFGS')) # , options={'eps': 1.e-02}))

# Print the current parameters
print(get_params())

####################
### A few images ###
####################

ol_polymorph_img = mpimg.imread('frost_2003_figures/ol_polymorphs.png')
ol_polymorph_img_1200C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_1200C.png')
ol_polymorph_img_1000C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_1000C.png')
ol_polymorph_img_800C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_800C.png')

ol_fper_img = mpimg.imread('frost_2003_figures/ol_fper_RTlnKD.png')
wad_fper_img = mpimg.imread('frost_2003_figures/wad_fper_RTlnKD.png')
rw_fper_img = mpimg.imread('frost_2003_figures/ring_fper_gt_KD.png')
rw_fper_part_img = mpimg.imread('frost_2003_figures/ring_fper_partitioning.png')


fo_phase_diagram = mpimg.imread('frost_2003_figures/Mg2SiO4_phase_diagram_Jacobs_2017.png')
fa_phase_diagram = mpimg.imread('frost_2003_figures/Fe2SiO4_phase_diagram_Yagi_1987.png')
fa_phase_diagram2 = mpimg.imread('frost_2003_figures/Fe2SiO4_phase_diagram_Jacobs_2001.png')

###################################################
# FUNCTION

def eqm_pressures(m1, m2, temperatures):
    composition = m1.formula
    assemblage = burnman.Composite([m1, m2])
    assemblage.set_state(1.e5, temperatures[0])
    equality_constraints = [('T', temperatures), ('phase_proportion', (m1, 0.0))]
    sols, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                    initial_state_from_assemblage=True,
                                    store_iterates=False)
    if type(sols) is list:
        return np.array([sol.x[0] for sol in sols])
    else:
        return sols.x[0]

###################################################
# PLOTS
"""
# Plot mrw EoS
pressures = np.linspace(1.e5, 25.e9, 101) 
plt.imshow(mrw_volume_diagram, extent=[0., 25., 35.5,40.5], aspect='auto')
for T in [300., 700., 1100., 1500., 1900.]:
    temperatures = pressures*0. + T
    plt.plot(pressures/1.e9, mrw.evaluate(['V'], pressures, temperatures)[0]*1.e6)
plt.show()


fig = plt.figure(figsize=(30, 15))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
ax[0].imshow(fo_phase_diagram, extent=[1000., 2600., 5, 30], aspect='auto')

temperatures = np.linspace(1000., 2600., 21)
ax[0].plot(temperatures, eqm_pressures(fo, mwd, temperatures)/1.e9, linewidth=4.)
ax[0].plot(temperatures, eqm_pressures(mwd, mrw, temperatures)/1.e9, linewidth=4.)


#ax[1].imshow(fa_phase_diagram, extent=[3., 7., 550.+273.15, 1350.+273.15], aspect='auto')
#temperatures = np.linspace(550.+273.15, 1350.+273.15, 21)
#ax[1].plot(eqm_pressures(fa, frw, temperatures)/1.e9, temperatures, linewidth=4.)
#ax[1].plot(eqm_pressures(fa, fwd, temperatures)/1.e9, temperatures, linestyle=':', linewidth=4.)


ax[1].imshow(fa_phase_diagram2, extent=[700., 1900., 0., 10.], aspect='auto')
temperatures = np.linspace(700., 1900., 21)
ax[1].plot(temperatures, eqm_pressures(fa, frw, temperatures)/1.e9, linewidth=4., label='fa-frw')
ax[1].plot(temperatures, eqm_pressures(fa, fwd, temperatures)/1.e9, linestyle=':', linewidth=4., label='fa-fwd (should be metastable)')


ax[0].set_xlabel('T (K)')
ax[0].set_ylabel('P (GPa)')
ax[1].set_xlabel('P (GPa)')
ax[1].set_ylabel('T (K)')
ax[1].legend()
plt.show()

# PLOT VOLUMES COMPARED WITH OTHER DATASETS
class Murnaghan_EOS(object):
    def __init__(self, V0, K, Kprime):
        self.V0 = V0
        self.K = K
        self.Kprime = Kprime
        self.V = lambda P: self.V0*np.power(1. + P*(self.Kprime/self.K),
                                            -1./self.Kprime) 


M_fo  = Murnaghan_EOS(4.6053e-5, 95.7e9, 4.6)
M_mwd = Murnaghan_EOS(4.2206e-5, 146.2544e9, 4.21)
M_mrw = Murnaghan_EOS(4.1484e-5, 145.3028e9, 4.4)
M_per = Murnaghan_EOS(1.1932e-5, 125.9e9, 4.1)
M_py  = Murnaghan_EOS(11.8058e-5, 129.0e9, 4.)

M_fa = Murnaghan_EOS(4.8494e-5, 99.8484e9, 4.)
M_fwd = Murnaghan_EOS(4.4779e-5, 139.9958e9, 4.)
M_frw = Murnaghan_EOS(4.3813e-5, 160.781e9, 5.)
M_fper = Murnaghan_EOS(1.2911e-5, 152.6e9, 4.)
M_alm = Murnaghan_EOS(12.1153e-5, 120.7515e9, 5.5)

from burnman.minerals import HHPH_2013, SLB_2011
dss = [[M_per, M_fper, M_fo, M_fa, M_mwd, M_fwd, M_mrw, M_frw, 'Frost (2003)'],
       [HHPH_2013.per(), HHPH_2013.fper(),
        HHPH_2013.fo(), HHPH_2013.fa(),
        HHPH_2013.mwd(), HHPH_2013.fwd(),
        HHPH_2013.mrw(), HHPH_2013.frw(), 'HHPH'],
       [SLB_2011.periclase(), SLB_2011.wuestite(),
        SLB_2011.forsterite(), SLB_2011.fayalite(),
        SLB_2011.mg_wadsleyite(), SLB_2011.fe_wadsleyite(),
        SLB_2011.mg_ringwoodite(), SLB_2011.fe_ringwoodite(), 'SLB'],
       [per,  wus,  fo,  fa,  mwd,  fwd,  mrw,  frw, 'this study']]


fig = plt.figure(figsize=(24,12))
ax = [fig.add_subplot(2, 4, i) for i in range(1, 9)]
pressures = np.linspace(1.e5, 24.e9, 101)
T=1673.15
temperatures = pressures*0. + T
for i, ds in enumerate(dss):
    if i == 0:
        Vs = [m.V(pressures) for m in ds[0:-1]]
    else:
        Vs = [m.evaluate(['V'], pressures, temperatures)[0] for m in ds[0:-1]]
    if i==3:
        linewidth=3.
    else:
        linewidth=1.
        
    for j, V in enumerate(Vs):
        try:
            ax[j].plot(pressures/1.e9, Vs[j]*1.e6, label=ds[j].name+' '+ds[-1],
                       linewidth=linewidth)
        except:
            ax[j].plot(pressures/1.e9, Vs[j]*1.e6, label=ds[-1],
                       linewidth=linewidth)

for i in range(0, 8):
    ax[i].legend(loc='best')
plt.show()

"""



# FPER-OL POLYMORPH (OR GARNET) PARTITIONING
def affinity_ol_fper(v, x_ol, G, T, W_ol, W_fper):
    #G is deltaG = G_per + G_fa/2. - G_fper - G_fo/2.
    x_fper = v[0]
    if np.abs(np.abs(x_ol - 0.5) - 0.5) < 1.e-10:
        v[0] = x_ol
        return 0.
    else:
        KD = ((x_ol*(1. - x_fper))/
              (x_fper*(1. - x_ol)))
        if KD < 0.:
            KD = 1.e-12
        return G - W_ol*(2.*x_ol - 1.) - W_fper*(1 - 2.*x_fper) + burnman.constants.gas_constant*T*np.log(KD)


# NOW PLOT THE FPER-OL-GT POLYMORPH EQUILIBRIA / PARTITIONING
ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_ol_gt_KD.dat')
ol_gt_data[:,0] *= 1.e9 # P (GPa) to P (Pa)

viridis = cm.get_cmap('viridis', 101)
Tmin = 1273.1
Tmax = 1673.2
fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
for i, Pplot in enumerate(set(ol_gt_data.T[0])):
    for T in sorted(set(ol_gt_data.T[1])):
        for m in mins:
            m.set_state(Pplot, T)
        G = (py.gibbs/3. - alm.gibbs/3. - fo.gibbs/2. + fa.gibbs/2.)
        W_ol = (ol.solution_model.We[0][1] + ol.solution_model.Wv[0][1] * Pplot) /2. # 1 cation
        W_gt = (gt.solution_model.We[0][1] + gt.solution_model.Wv[0][1] * Pplot) /3.

    
        x_ols = np.linspace(0.00001, 0.99999, 101)
        x_gts = np.array([fsolve(affinity_ol_fper, [x_ol], args=(x_ol, G, T, W_ol, W_gt))[0]
                          for x_ol in x_ols])
        KDs = ((x_gts*(1. - x_ols))/
               (x_ols*(1. - x_gts)))
        ax[i].plot(1. - x_ols, np.log(KDs), color = viridis((T-Tmin)/(Tmax-Tmin)), linewidth=3., label='{0} K'.format(T))

    
    Ps, Ts, XMgOl, lnKD, lnKDerr = ol_gt_data.T
    mask = [idx for idx, P in enumerate(Ps) if np.abs(P - Pplot) < 10000.]
    ax[i].errorbar(XMgOl[mask], lnKD[mask], yerr=lnKDerr[mask], linestyle='None')
    ax[i].scatter(XMgOl[mask], lnKD[mask], c=Ts[mask], s=80., label='data', cmap=viridis, vmin=Tmin, vmax=Tmax)

    ax[i].set_xlabel('p(fo)')
    ax[i].set_ylabel('ln(KD ol-gt)')
    ax[i].legend()
    

plt.show()
                                    

fig = plt.figure(figsize=(30,10))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

# OLIVINE
ax[0].imshow(ol_fper_img, extent=[0.0, 0.8, -45000., -5000.], aspect='auto')


Pmin = 0.e9
Pmax = 15.e9

T = 1673.15
for P in [1.e5, 5.e9, 10.e9, 15.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (per.gibbs - wus.gibbs - fo.gibbs/2. + fa.gibbs/2.)
    W_ol = (ol.solution_model.We[0][1] + ol.solution_model.Wv[0][1] * P) /2. # 1 cation
    W_fper = fper.solution_model.We[0][1] + fper.solution_model.Wv[0][1] * P
    
    x_ols = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_ol], args=(x_ol, G, T, W_ol, W_fper))[0]
                        for x_ol in x_ols])
    KDs = ((x_ols*(1. - x_fpers))/
           (x_fpers*(1. - x_ols)))
    ax[0].plot(x_ols, burnman.constants.gas_constant*T*np.log(KDs), color = viridis((P-Pmin)/(Pmax-Pmin)), linewidth=3., label='{0} GPa'.format(P/1.e9))


P_Xol_RTlnKDs = []
for assemblage in Frost_2003_assemblages:
    if solutions['ol'] in assemblage.phases:
        idx_ol = assemblage.phases.index(solutions['ol'])
        idx_mw = assemblage.phases.index(solutions['mw'])
        T = assemblage.nominal_state[1]
        x_ol = assemblage.stored_compositions[idx_ol][0][1]
        x_fper = assemblage.stored_compositions[idx_mw][0][1]
        RTlnKD = burnman.constants.gas_constant*T*np.log((x_ol*(1. - x_fper))/
                                                         (x_fper*(1. - x_ol)))
        P_Xol_RTlnKDs.append([assemblage.nominal_state[0],
                              x_ol, RTlnKD])


pressures, x_ols, RTlnKDs = np.array(P_Xol_RTlnKDs).T
ax[0].scatter(x_ols, RTlnKDs, c=pressures, s=80., label='data',
              cmap=viridis, vmin=Pmin, vmax=Pmax)


ax[0].set_xlim(0., 0.8)
ax[0].legend(loc='best')


# WADSLEYITE
ax[1].imshow(wad_fper_img, extent=[0.0, 0.4, -25000., -5000.], aspect='auto')

viridis = cm.get_cmap('viridis', 101)
Pmin = 10.e9
Pmax = 18.e9


T = 1673.15
for P in [10.e9, 12.e9, 14.e9, 16.e9, 18.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (per.gibbs - wus.gibbs - mwd.gibbs/2. + fwd.gibbs/2.)
    W_wad = (wad.solution_model.We[0][1] + wad.solution_model.Wv[0][1] * P)/2. # 1 cation
    W_fper = fper.solution_model.We[0][1] + fper.solution_model.Wv[0][1] * P
    
    x_wads = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_wad], args=(x_wad, G, T, W_wad, W_fper))[0]
                        for x_wad in x_wads])
    KDs = ((x_wads*(1. - x_fpers))/
           (x_fpers*(1. - x_wads)))
    ax[1].plot(x_wads, burnman.constants.gas_constant*T*np.log(KDs), color = viridis((P-Pmin)/(Pmax-Pmin)), linewidth=3., label='{0} GPa'.format(P/1.e9))

P_Xwad_RTlnKDs = []
for assemblage in Frost_2003_assemblages:
    if solutions['wad'] in assemblage.phases:
        idx_wad = assemblage.phases.index(solutions['wad'])
        idx_mw = assemblage.phases.index(solutions['mw'])
        x_wad = assemblage.stored_compositions[idx_wad][0][1]
        x_fper = assemblage.stored_compositions[idx_mw][0][1]
        T = assemblage.nominal_state[1]
        RTlnKD = burnman.constants.gas_constant*T*np.log((x_wad*(1. - x_fper))/
                                                         (x_fper*(1. - x_wad)))
        P_Xwad_RTlnKDs.append([assemblage.nominal_state[0],
                               x_wad, RTlnKD])



pressures, x_wads, RTlnKDs = np.array(P_Xwad_RTlnKDs).T
ax[1].scatter(x_wads, RTlnKDs, c=pressures, s=80., label='data',
              cmap=viridis, vmin=Pmin, vmax=Pmax)

ax[1].set_xlim(0., 0.4)
ax[1].legend(loc='best')


# RINGWOODITE
#ax[2].imshow(rw_fper_part_img, extent=[0.0, 1., 0., 1.], aspect='auto')

viridis = cm.get_cmap('viridis', 101)
Pmin = 10.e9
Pmax = 24.e9


T = 1673.15
for P in [10.e9, 12.5e9, 15.e9, 17.5e9, 20.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (per.gibbs - wus.gibbs - mrw.gibbs/2. + frw.gibbs/2.) 
    W_rw = (child_solutions['ring'].solution_model.We[0][1]  + child_solutions['ring'].solution_model.Wv[0][1] * P)/2. # 1 cation
    W_fper = fper.solution_model.We[0][1] + fper.solution_model.Wv[0][1] * P
    
    x_rws = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_rw], args=(x_rw, G, T, W_rw, W_fper))[0]
                        for x_rw in x_rws])

    ax[2].plot(x_rws, x_fpers, color=viridis((P-Pmin)/(Pmax-Pmin)), linewidth=3., label=P/1.e9)


P_Xrw_Xfper = []
for assemblage in Frost_2003_assemblages:
    if child_solutions['ring'] in assemblage.phases:
        idx_rw = assemblage.phases.index(child_solutions['ring'])
        idx_mw = assemblage.phases.index(solutions['mw'])
        P_Xrw_Xfper.append([assemblage.nominal_state[0],
                            assemblage.stored_compositions[idx_rw][0][1],
                            assemblage.stored_compositions[idx_mw][0][1]])

            
pressures, x_rws, x_fpers = np.array(P_Xrw_Xfper).T
c = ax[2].scatter(x_rws, x_fpers, c=pressures, s=80., label='data',
                  cmap=viridis, vmin=Pmin, vmax=Pmax)

ax[2].set_xlim(0., 1.)
ax[2].set_ylim(0., 1.)
ax[2].legend(loc='best')
plt.show()

# BINARY PHASE DIAGRAM

#plt.imshow(ol_polymorph_img, extent=[0., 1., 6., 20.], aspect='auto')
#plt.imshow(ol_polymorph_img_1000C, extent=[-0.01, 1.005, 4., 21.], aspect='auto')

for (T0, color) in [(1273.15, 'blue'),
                    (1673.15, 'orange'),
                    (2073.15, 'purple')]:
    
    x_m1 = 0.3

    composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1.-x_m1), 'Si': 1., 'O': 4.}
    wad.guess = np.array([1. - x_m1, x_m1])
    ol.guess = np.array([1. - x_m1, x_m1])
    child_solutions['ring'].guess = np.array([0.15, 0.85])
    assemblage = burnman.Composite([ol, wad, child_solutions['ring']])
    assemblage.set_state(14.e9, T0)
    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False, initial_state_from_assemblage=True)
    P_inv = assemblage.pressure
    x_ol_inv = assemblage.phases[0].molar_fractions[1]
    x_wad_inv = assemblage.phases[1].molar_fractions[1]
    x_rw_inv = assemblage.phases[2].molar_fractions[1]
    for (m1, m2) in [(wad, ol), (wad, child_solutions['ring']), (ol, child_solutions['ring'])]:
        composition = {'Fe': 0., 'Mg': 2., 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([m1.endmembers[0][0], m2.endmembers[0][0]])
        equality_constraints = [('T', T0), ('phase_proportion', (m1.endmembers[0][0], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
        P1 = assemblage.pressure
        composition = {'Fe': 2., 'Mg': 0., 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([m1.endmembers[1][0], m2.endmembers[1][0]])
        equality_constraints = [('T', T0), ('phase_proportion', (m1.endmembers[1][0], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
        P0 = assemblage.pressure
        print(P0/1.e9, P1/1.e9)
        if m1 is wad:
            x_m1s = np.linspace(0.001, x_wad_inv, 21)
        else:
            x_m1s = np.linspace(x_ol_inv, 0.999, 21)
        
        pressures = np.empty_like(x_m1s)
        x_m2s = np.empty_like(x_m1s)
        m1.guess = np.array([1. - x_m1s[0], x_m1s[0]])
        m2.guess = np.array([1. - x_m1s[0], x_m1s[0]])
        for i, x_m1 in enumerate(x_m1s):
            composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
            assemblage = burnman.Composite([m1, m2])
            assemblage.set_state(P1*(1 - x_m1) + P0*x_m1, T0)
            m1.set_composition([1. - x_m1, x_m1])
            m2.set_composition(m2.guess)
            assemblage.n_moles = 1.
            assemblage.set_fractions([1., 0.])
            equality_constraints = [('T', T0), ('phase_proportion', (m2, 0.0))]
            sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           initial_state_from_assemblage=True,
                                           initial_composition_from_assemblage=True,
                                           store_iterates=False)
            
            m2.guess = m2.molar_fractions
            x_m2s[i] = m2.molar_fractions[1]
            pressures[i] = assemblage.pressure

        plt.plot(x_m1s, pressures/1.e9, linewidth=3., color=color)
        plt.plot(x_m2s, pressures/1.e9, linewidth=3., color=color, label='{0} K'.format(T0))
    plt.plot([x_ol_inv, x_rw_inv], [P_inv/1.e9, P_inv/1.e9], linewidth=3., color=color)

    """
    # bdg + fper 
    x_m1s = []
    pressures = []
    x_m2s = []
    for x_m1 in np.linspace(0.3, 0.5, 51):
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
        child_solutions['mg_fe_bdg'].guess = np.array([1. - x_m1, x_m1])
        fper.guess = np.array([1. - x_m1, x_m1])
        assemblage = burnman.Composite([child_solutions['mg_fe_bdg'], fper, stv])
        assemblage.set_state(30.e9, T0)
        equality_constraints = [('T', T0),
                                ('phase_proportion', (stv, 0.))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_state_from_assemblage=True,
                                       store_iterates=False)
        if sol.success:
            print('yo', assemblage.pressure/1.e9)
            x_m1s.append(x_m1)
            x_m2s.append(fper.molar_fractions[1])
            pressures.append(assemblage.pressure)
    
    plt.plot(x_m1s, np.array(pressures)/1.e9, linewidth=3., color=color)
    plt.plot(x_m2s, np.array(pressures)/1.e9, linewidth=3., color=color)
    """
    
    # bdg + fper 
    x_m1s = []
    pressures = []
    x_m2s = []
    x_m1_array = np.linspace(0.01, 0.3, 21)
    
    child_solutions['ring'].guess = np.array([1. - x_m1_array[0], x_m1_array[0]])
    child_solutions['mg_fe_bdg'].guess = np.array([1. - x_m1_array[0], x_m1_array[0]])
    fper.guess = np.array([1. - x_m1_array[0], x_m1_array[0]])
    
    for x_m1 in x_m1_array:
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
        
        child_solutions['ring'].set_composition([1. - x_m1, x_m1])
        child_solutions['mg_fe_bdg'].set_composition(child_solutions['mg_fe_bdg'].guess)
        fper.set_composition(fper.guess)
        
        assemblage = burnman.Composite([child_solutions['ring'],
                                        child_solutions['mg_fe_bdg'], fper], [1., 0., 0.])
        assemblage.set_state(25.e9, T0)
        equality_constraints = [('T', T0),
                                ('phase_proportion', (child_solutions['ring'], 1.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_composition_from_assemblage=True,
                                       initial_state_from_assemblage=True,
                                       store_iterates=False)
        if sol.success:
            print(assemblage.pressure/1.e9)
            x_m1s.append(x_m1)
            x_m2s.append((fper.molar_fractions[1] +
                          child_solutions['mg_fe_bdg'].molar_fractions[1])/2.)

            child_solutions['mg_fe_bdg'].guess = child_solutions['mg_fe_bdg'].molar_fractions
            fper.guess = fper.molar_fractions
            
            pressures.append(assemblage.pressure)
    
    plt.plot(x_m1s, np.array(pressures)/1.e9, linewidth=3., color=color)
    plt.plot(x_m2s, np.array(pressures)/1.e9, linewidth=3., color=color)

    
    # rw -> fper + stv
    x_m1s = np.linspace(0.2, 0.99, 21)
    pressures = np.empty_like(x_m1s)
    x_m2s = np.empty_like(x_m1s)

    
    child_solutions['ring'].guess = np.array([1. - x_m1s[0], x_m1s[0]])
    Pi = 22.e9
    for i, x_m1 in enumerate(x_m1s):
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([child_solutions['ring'], fper, stv], [0., 2./3., 1./3.])
        assemblage.set_state(Pi, T0)

        fper.set_composition([1. - x_m1, x_m1])
        child_solutions['ring'].set_composition(child_solutions['ring'].guess)

        
        equality_constraints = [('T', T0), ('phase_proportion', (child_solutions['ring'], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_composition_from_assemblage=True,
                                       initial_state_from_assemblage=True,
                                       store_iterates=False)
        x_m2s[i] = child_solutions['ring'].molar_fractions[1]
        child_solutions['ring'].guess = child_solutions['ring'].molar_fractions
        pressures[i] = assemblage.pressure
        Pi = assemblage.pressure

    plt.plot(x_m1s, pressures/1.e9, linewidth=3., color=color)
    plt.plot(x_m2s, pressures/1.e9, linewidth=3., color=color)
    
    
P_rw_fper = []
for assemblage in Matsuzaka_2000_assemblages:
    P_rw_fper.append([assemblage.nominal_state[0], assemblage.nominal_state[1],
                      assemblage.stored_compositions[0][0][1],
                      assemblage.stored_compositions[1][0][1]])
    
P, Ts, x_rw, x_fper = np.array(P_rw_fper).T
mask = [i for i, T in enumerate(Ts) if np.abs(T - 1673.15) < 0.1]
plt.scatter(x_rw[mask], P[mask]/1.e9, color='orange', label='1673.15 K')
plt.scatter(x_fper[mask], P[mask]/1.e9, color='orange')
mask = [i for i, T in enumerate(Ts) if np.abs(T - 1873.15) < 0.1]
plt.scatter(x_rw[mask], P[mask]/1.e9, color='red', label='1873.15 K')
plt.scatter(x_fper[mask], P[mask]/1.e9, color='red')
    

P_Xmg_phase = {'ol': [], 'wad': [], 'ring': []}
for assemblage in Frost_2003_assemblages:
    if len(assemblage.phases) > 2:
        for i, phase in enumerate(assemblage.phases):
            for m in ['ol', 'wad', 'ring']:
                try:
                    if phase == solutions[m]:
                        P_shift = dict_experiment_uncertainties[assemblage.experiment_id]['P']
                    
                        P_Xmg_phase[m].append([assemblage.nominal_state[0], P_shift,
                                               assemblage.stored_compositions[i][0][1]])
                except:
                    if phase == child_solutions[m]:
                        P_shift = dict_experiment_uncertainties[assemblage.experiment_id]['P']
                    
                        P_Xmg_phase[m].append([assemblage.nominal_state[0], P_shift,
                                               assemblage.stored_compositions[i][0][1]])
                    
arrow_params = {'shape': 'full',
                'width': 0.001,
                'length_includes_head': True,
                'head_starts_at_zero': False}

for m in ['ol', 'wad', 'ring']:
    pressures, pressure_shift, xs = np.array(P_Xmg_phase[m]).T
    for i in range(len(xs)):
        plt.arrow(xs[i], pressures[i]/1.e9, 0., pressure_shift[i]/1.e9, **arrow_params)
    plt.scatter(xs, pressures/1.e9, s=80., label='data')
    
plt.legend()
plt.show()
