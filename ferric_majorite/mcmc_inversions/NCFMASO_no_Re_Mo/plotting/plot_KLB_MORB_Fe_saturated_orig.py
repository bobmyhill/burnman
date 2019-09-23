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
from preload_params_CFMASO_w_Al_exchange import *

def set_params_from_special_constraints():
    # 1) Destabilise fwd
    fa.set_state(6.25e9, 1673.15)
    frw.set_state(6.25e9, 1673.15)
    fwd.set_state(6.25e9, 1673.15)

    # First, determine the entropy which will give the fa-fwd reaction the same slope as the fa-frw reaction
    dPdT = (frw.S - fa.S)/(frw.V - fa.V) # = dS/dV
    dV = fwd.V - fa.V
    dS = dPdT*dV
    fwd.params['S_0'] += fa.S - fwd.S + dS
    fwd.params['H_0'] += frw.gibbs - fwd.gibbs + 100. # make fwd a little less stable than frw

    # 2) Copy interaction parameters from opx to hpx:
    hpx_od.alphas = opx_od.alphas
    hpx_od.energy_interaction = opx_od.energy_interaction
    hpx_od.entropy_interaction = opx_od.entropy_interaction
    hpx_od.volume_interaction = opx_od.volume_interaction
    
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

endmember_args = [['wus', 'H_0', wus.params['H_0'], 1.e3], # per is a standard 
                  ['fo',  'H_0', fo.params['H_0'],  1.e3],
                  ['fa',  'H_0', fa.params['H_0'],  1.e3],
                  ['mwd', 'H_0', mwd.params['H_0'], 1.e3], # fwd H_0 and S_0 obtained in set_params_from_special_constraints()
                  ['mrw', 'H_0', mrw.params['H_0'], 1.e3],
                  ['frw', 'H_0', frw.params['H_0'], 1.e3],
                  
                  ['alm', 'H_0', alm.params['H_0'], 1.e3], # pyrope is a standard
                  ['gr',   'H_0', gr.params['H_0'], 1.e3],
                  ['andr', 'H_0', andr.params['H_0'], 1.e3],
                  ['dmaj', 'H_0', dmaj.params['H_0'], 1.e3],
                  ['nagt', 'H_0', nagt.params['H_0'], 1.e3],
                  
                  ['coe', 'H_0', coe.params['H_0'], 1.e3], # qtz is a standard
                  ['stv', 'H_0', stv.params['H_0'], 1.e3],
                  
                  ['hed', 'H_0', hed.params['H_0'], 1.e3], # diopside is a standard
                  ['cen', 'H_0', cen.params['H_0'], 1.e3], 
                  # ['cfs', 'H_0', cfs.params['H_0'], 1.e3], 
                  ['cats', 'H_0', cats.params['H_0'], 1.e3], # jadeite is a standard
                  # ['aeg', 'H_0', aeg.params['H_0'], 1.e3],
                  
                  ['oen', 'H_0', oen.params['H_0'], 1.e3],
                  ['ofs', 'H_0', ofs.params['H_0'], 1.e3], #  no odi, as based on di (a std)
                  ['mgts', 'H_0', mgts.params['H_0'], 1.e3],
                  
                  ['hen', 'H_0', hen.params['H_0'], 1.e3],
                  ['hfs', 'H_0', hfs.params['H_0'], 1.e3],
                  
                  ['mbdg', 'H_0', mbdg.params['H_0'], 1.e3],
                  ['fbdg', 'H_0', fbdg.params['H_0'], 1.e3],
                  
                  ['sp', 'H_0', sp.params['H_0'], 1.e3],
                  
                  ['per', 'S_0', per.params['S_0'], 1.], # has a prior associated with it, so can be inverted
                  ['wus', 'S_0', wus.params['S_0'], 1.],
                  ['fo',  'S_0', fo.params['S_0'],  1.],
                  ['fa',  'S_0', fa.params['S_0'],  1.],
                  ['mwd', 'S_0', mwd.params['S_0'], 1.], 
                  ['mrw', 'S_0', mrw.params['S_0'], 1.],
                  ['frw', 'S_0', frw.params['S_0'], 1.],
                  ['alm', 'S_0', alm.params['S_0'], 1.],
                  ['gr',   'S_0',   gr.params['S_0'], 1.],
                  ['andr', 'S_0', andr.params['S_0'], 1.],
                  ['dmaj', 'S_0', dmaj.params['S_0'], 1.],
                  ['nagt', 'S_0', nagt.params['S_0'], 1.],

                  ['di', 'S_0', di.params['S_0'], 1.], # has a prior associated with it, so can be inverted
                  ['hed', 'S_0', hed.params['S_0'], 1.],
                  
                  ['oen', 'S_0', oen.params['S_0'], 1.],
                  ['ofs', 'S_0', ofs.params['S_0'], 1.],
                  ['mgts', 'S_0', mgts.params['S_0'], 1.],
                  
                  ['hen', 'S_0', hen.params['S_0'], 1.],
                  ['hfs', 'S_0', hfs.params['S_0'], 1.],
                  
                  ['coe', 'S_0', coe.params['S_0'], 1.],
                  ['stv', 'S_0', stv.params['S_0'], 1.],
                  ['mbdg', 'S_0', mbdg.params['S_0'], 1.],
                  ['fbdg', 'S_0', fbdg.params['S_0'], 1.],
                  
                  ['sp', 'S_0', sp.params['S_0'], 1.],
                  
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
                 
                 ['opx', 'E', 0, 0, opx_od.energy_interaction[0][0], 1.e3], # oen-ofs
                 ['opx', 'E', 0, 1, opx_od.energy_interaction[0][1], 1.e3], # oen-mgts
                 ['opx', 'E', 0, 2, opx_od.energy_interaction[0][2], 1.e3], # oen-odi
                 ['opx', 'E', 2, 0, opx_od.energy_interaction[2][0], 1.e3], # mgts-odi
                 
                 ['cpx', 'E', 0, 0, cpx_od.energy_interaction[0][0], 1.e3], # di-hed
                 ['cpx', 'E', 0, 1, cpx_od.energy_interaction[0][1], 1.e3], # di-cen
                 ['cpx', 'E', 0, 3, cpx_od.energy_interaction[0][3], 1.e3], # di-cats
                 ['cpx', 'E', 2, 1, cpx_od.energy_interaction[2][1], 1.e3], # cen-cats
                 
                 ['gt', 'E', 0, 2,  gt.energy_interaction[0][2], 1.e3], # py-andr (py-alm ideal)
                 ['gt', 'E', 0, 3,  gt.energy_interaction[0][3], 1.e3], # py-dmaj
                 ['gt', 'E', 0, 4,  gt.energy_interaction[0][4], 1.e3], # py-nagt
                 ['gt', 'E', 1, 0,  gt.energy_interaction[1][0], 1.e3], # alm-gr
                 ['gt', 'E', 1, 1,  gt.energy_interaction[1][1], 1.e3], # alm-andr
                 ['gt', 'E', 1, 2,  gt.energy_interaction[1][2], 1.e3], # alm-dmaj
                 ['gt', 'E', 1, 3,  gt.energy_interaction[1][3], 1.e3], # alm-nagt
                 ['gt', 'E', 2, 0,  gt.energy_interaction[2][0], 1.e3], # gr-andr
                 ['gt', 'E', 2, 1,  gt.energy_interaction[2][1], 1.e3], # gr-dmaj
                 ['gt', 'E', 2, 2,  gt.energy_interaction[2][2], 1.e3], # gr-nagt
                 ['gt', 'V', 0, 2,  gt.volume_interaction[0][2], 1.e-7], # py-andr
                 ['gt', 'V', 0, 4,  gt.volume_interaction[0][4], 1.e-7], # py-nagt
                 ['gt', 'V', 1, 1,  gt.volume_interaction[1][1], 1.e-7]] # alm-andr, ['bdg', 'E', 0, 0, bdg.energy_interaction[0][0], 1.e3]]

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
                    ['gr', 'S_0', gr.params['S_0_orig'][0], gr.params['S_0_orig'][1]],
                    ['andr', 'S_0', andr.params['S_0_orig'][0], gr.params['S_0_orig'][1]],
                    
                    ['di', 'S_0', di.params['S_0_orig'][0], di.params['S_0_orig'][1]],
                    ['hed', 'S_0', hed.params['S_0_orig'][0], hed.params['S_0_orig'][1]],
                    
                    ['oen', 'S_0', oen.params['S_0_orig'][0], oen.params['S_0_orig'][1]],
                    ['ofs', 'S_0', ofs.params['S_0_orig'][0], ofs.params['S_0_orig'][1]],
                    
                    ['mbdg', 'S_0', mbdg.params['S_0_orig'][0], mbdg.params['S_0_orig'][1]],
                    ['fbdg', 'S_0', fbdg.params['S_0_orig'][0], fbdg.params['S_0_orig'][1]],
                    
                    ['sp', 'S_0', sp.params['S_0_orig'][0], sp.params['S_0_orig'][1]],
                    
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

solution_priors = [['opx', 'E', 0, 0, 0.e3, 1.e3],
                   ['gt', 'E', 1, 0,  gt.energy_interaction[1][0], 10.e3], # alm-gr
                   ['gt', 'E', 2, 0,  2.e3, 1.e3]] # gr-andr , ['bdg', 'E', 0, 0, 4.e3, 0.0001e3]] # ['gt', 'E', 0, 0, 0.3e3, 0.4e3]

# All Frost 2003 uncertainties already in preload
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
                            ['V254', 'P', 0., 0.5e9], 
                            ['Frost_2003_H1554', 'P', 0., 0.5e9],
                            ['Frost_2003_H1555', 'P', 0., 0.5e9],
                            ['Frost_2003_H1556', 'P', 0., 0.5e9],
                            ['Frost_2003_H1582', 'P', 0., 0.5e9],
                            ['Frost_2003_S2773', 'P', 0., 0.5e9],
                            ['Frost_2003_V170', 'P', 0., 0.5e9],
                            ['Frost_2003_V171', 'P', 0., 0.5e9],
                            ['Frost_2003_V175', 'P', 0., 0.5e9],
                            ['Frost_2003_V179', 'P', 0., 0.5e9]])
"""

experiment_uncertainties.extend([['Beyer2019_H4321', 'P', 0., 0.2e9],
                                 ['Beyer2019_H4556', 'P', 0., 0.2e9],
                                 ['Beyer2019_H4557', 'P', 0., 0.2e9],
                                 ['Beyer2019_H4560', 'P', 0., 0.2e9],
                                 ['Beyer2019_H4692', 'P', 0., 0.2e9],
                                 ['Beyer2019_Z1699', 'P', 0., 0.2e9],
                                 ['Beyer2019_Z1700', 'P', 0., 0.2e9],
                                 ['Beyer2019_Z1782', 'P', 0., 0.2e9],
                                 ['Beyer2019_Z1785', 'P', 0., 0.2e9],
                                 ['Beyer2019_Z1786', 'P', 0., 0.2e9]])

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
        elif a[1] == 'S':
            solutions[a[0]].entropy_interaction[int(a[2])][int(a[3])] = args[i]*a[5]
        elif a[1] == 'V':
            solutions[a[0]].volume_interaction[int(a[2])][int(a[3])] = args[i]*a[5]
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
from ONeill_Wood_1979_CFMAS_ol_gt import ONeill_Wood_1979_CFMAS_assemblages
from endmember_reactions import endmember_reaction_assemblages
from Matsuzaka_et_al_2000_rw_wus_stv import Matsuzaka_2000_assemblages
from ONeill_1987_QFI import ONeill_1987_QFI_assemblages
from ONeill_1987_QFM import ONeill_1987_QFM_assemblages
from Nakajima_FR_2012_bdg_fper import Nakajima_FR_2012_assemblages
from Tange_TNFS_2009_bdg_fper_stv import Tange_TNFS_2009_FMS_assemblages
from Frost_2003_FMASO_garnet import Frost_2003_FMASO_gt_assemblages


# MAS
from Gasparik_1989_MAS_px_gt import Gasparik_1989_MAS_assemblages
from Gasparik_1992_MAS_px_gt import Gasparik_1992_MAS_assemblages
from Gasparik_Newton_1984_MAS_opx_sp_fo import Gasparik_Newton_1984_MAS_assemblages
from Gasparik_Newton_1984_MAS_py_opx_sp_fo import Gasparik_Newton_1984_MAS_univariant_assemblages
from Perkins_et_al_1981_MAS_py_opx import Perkins_et_al_1981_MAS_assemblages
#from Liu_et_al_2016_gt_bdg_cor import Liu_et_al_2016_MAS_assemblages
#from Liu_et_al_2017_bdg_cor import Liu_et_al_2017_MAS_assemblages
#from Hirose_et_al_2001_ilm_bdg_gt import Hirose_et_al_2001_MAS_assemblages

# CMS
from Carlson_Lindsley_1988_CMS_opx_cpx import Carlson_Lindsley_1988_CMS_assemblages

# CMAS
from Perkins_Newton_1980_CMAS_opx_cpx_gt import Perkins_Newton_1980_CMAS_assemblages
from Gasparik_1989_CMAS_px_gt import Gasparik_1989_CMAS_assemblages
from Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp import Klemme_ONeill_2000_CMAS_assemblages

# NMAS
from Gasparik_1989_NMAS_px_gt import Gasparik_1989_NMAS_assemblages

# NCMAS                    
from Gasparik_1989_NCMAS_px_gt import Gasparik_1989_NCMAS_assemblages
                                
# CFMS
from Perkins_Vielzeuf_1992_CFMS_ol_cpx import Perkins_Vielzeuf_1992_CFMS_assemblages

# FASO
from Woodland_ONeill_1993_FASO_alm_sk import Woodland_ONeill_1993_FASO_assemblages

# NCFMASO
from Rohrbach_et_al_2007_NCFMASO_gt_cpx import Rohrbach_et_al_2007_NCFMASO_assemblages
from Beyer_et_al_2019_NCFMASO import Beyer_et_al_2019_NCFMASO_assemblages

assemblages = [assemblage for assemblage_list in # NOT Woodland and ONeill.
               [endmember_reaction_assemblages,
                ONeill_1987_QFI_assemblages,
                ONeill_1987_QFM_assemblages,
                Frost_2003_assemblages,
                Seckendorff_ONeill_1992_assemblages,
                Matsuzaka_2000_assemblages,
                ONeill_Wood_1979_assemblages,
                ONeill_Wood_1979_CFMAS_assemblages,
                Nakajima_FR_2012_assemblages,
                Tange_TNFS_2009_FMS_assemblages,
                Frost_2003_FMASO_gt_assemblages,
                Perkins_Vielzeuf_1992_CFMS_assemblages, # need ol, di-hed
                Gasparik_Newton_1984_MAS_assemblages, # need sp, oen-mgts
                Gasparik_Newton_1984_MAS_univariant_assemblages, # need sp, oen-mgts
                Perkins_Newton_1980_CMAS_assemblages, # need oen_mgts_odi, di_cen_cats, py_gr
                Klemme_ONeill_2000_CMAS_assemblages, # need sp, oen_mgts_odi, di_cen_cats, py_gr
                Gasparik_1992_MAS_assemblages, # need oen-mgts, py-dmaj
                Gasparik_1989_MAS_assemblages,
                Gasparik_1989_CMAS_assemblages,
                Gasparik_1989_NMAS_assemblages,
                Gasparik_1989_NCMAS_assemblages,
                Rohrbach_et_al_2007_NCFMASO_assemblages, 
                Beyer_et_al_2019_NCFMASO_assemblages
               ]
               for assemblage in assemblage_list]

print('Inversion involves {0} assemblages and {1} parameters'.format(len(assemblages), len(get_params())))

#minimize_func(get_params(), assemblages)
"""

#######################
### PUT PARAMS HERE ###
#######################
set_params([-267.6845, -2177.5606, -1476.2807, -2148.1010, -2136.7039, -1475.0240, -5252.6838, -6641.3333, -5769.3341, -6036.8097, -5987.6560, -906.9341, -871.6971, -2842.4774, -3098.3283, -3305.8749, -3093.6776, -2390.9611, -3196.2064, -3086.1537, -2393.1370, -1450.7003, -1106.2620, -2301.2043, 26.5639, 57.4976, 94.5631, 152.1736, 83.6248, 80.9340, 138.4026, 341.1031, 250.7490, 316.6318, 250.9272, 265.1709, 144.7493, 173.4492, 131.4765, 188.0793, 130.4549, 130.6944, 177.5769, 39.6830, 27.7577, 57.9612, 83.0199, 81.0989, 4.3966, 2.0707, 1.4615, 1.9996, 2.8756, 3.8533, 3.2286, 2.3271, 1.9801, 1.4154, 2.0853, 3.0551, 1.7016, 2.5878, 9.4596, 8.1330, 18.1906, 8.1177, -1.4412, 15.5174, 30.9790, 73.9404, 2.4863, 37.9882, 12.4517, 44.3543, 89.6581, 0.0001, 0.6289, 0.0192, 59.9200, 0.4027, 0.0040, 4.9080, 1.0810, -0.9383, -0.5792, 1.3366, -0.1398, -1.2732, -0.2395, -3.0552, -0.3056, 1.0698, -4.2922, -1.8534, -0.3463, -1.1215, 0.1149, -0.0012, 1.1903, 0.5428, 2.1660, -0.7386, -1.1993, -0.9337, -0.5402, 0.1644, 0.6403, 0.4801, -0.1921, 1.1233, -0.0636, -0.2075, -0.1718, -0.1219, 0.0221, -0.6935, -1.2044, -0.4464, -0.1284, -0.0244, 0.0012, -0.0547, 0.0129, -0.5169, 0.0193, -0.2392, -0.2419, -1.9741, -1.0885])

set_params([-276.5553, -2178.0041, -1476.8761, -2145.9814, -2134.3388, -1492.0933, -5263.3282, -6646.1433, -5762.5577, -6038.5416, -5997.5922, -906.7790, -870.2295, -2840.4853, -3096.5070, -3300.9217, -3094.9185, -2389.3313, -3191.5833, -3087.4791, -2390.4996, -1450.0707, -1106.9159, -2300.7577, 27.3770, 51.0279, 94.1479, 151.5242, 85.3457, 83.6003, 124.7446, 332.4315, 240.5733, 304.7181, 254.5180, 280.4402, 143.1497, 174.7402, 131.0381, 189.2199, 133.0653, 130.1817, 179.5711, 39.7892, 28.6761, 58.8558, 81.1234, 80.2488, 4.3161, 1.7593, 1.7083, 1.9638, 3.0650, 3.4756, 3.1074, 2.5590, 1.9020, 1.8566, 2.1257, 2.4230, 1.8100, 2.3309, 11.4172, 8.1034, 15.9232, 8.1756, 0.0745, 15.7974, 34.0590, 74.9336, 5.4467, 34.9312, 16.1070, 34.8573, 86.8273, 0.0000, -6.7029, 0.2270, 63.1903, 14.9215, 0.8290, 2.8572, 24.0050, 4.5439, -5.3093, -10.7218, 4.8370, -3.4994, -0.8612, -6.2695, -1.5641, -2.1971, -3.4395, 1.5860, -1.2538, -4.2586, -2.6590, -0.0466, 0.8529, 4.9849, 1.5617, -1.5043, -1.4688, -3.5944, -1.1353, -0.3632, -0.0476, -0.2123, -0.2459, 0.6217, -0.1231, -0.4535, -0.3079, -0.2105, 0.1109, -1.3662, -1.2338, -0.8944, -0.2116, -1.3364, 0.0112, -4.5557, 0.7114, -11.7490, 0.7769, -10.5673, 2.3058, -22.8776, -19.6252])

set_params([-276.5553, -2178.0041, -1476.8761, -2145.9814, -2134.3388, -1492.0933, -5263.3282, -6646.1433, -5762.5577, -6038.5416, -5997.5922, -906.7790, -870.2295, -2840.4853, -3096.5070, -3300.9217, -3094.9185, -2389.3313, -3191.5833, -3087.4791, -2390.4996, -1450.0707, -1106.9159, -2300.7577, 27.3770, 51.0279, 94.1479, 151.5242, 85.3457, 83.6003, 124.7446, 332.4315, 240.5733, 304.7181, 254.5180, 280.4402, 143.1497, 174.7402, 131.0381, 189.2199, 133.0653, 130.1817, 179.5711, 39.7892, 28.6761, 58.8558, 81.1234, 80.2488, 4.3161, 1.7593, 1.7083, 1.9638, 3.0650, 3.4756, 3.1074, 2.5590, 1.9020, 1.8566, 2.1257, 2.4230, 1.8100, 2.3309, 11.4172, 8.1034, 15.9232, 8.1756, 0.0745, 15.7974, 34.0590, 74.9336, 5.4467, 34.9312, 16.1070, 34.8573,
            86.8273, 0.0000, -6.7029, 0.2270,
            63.1903, 14.9215, 0.8290, 2.8572, 24.0050, 4.5439, -5.3093, -10.7218, 4.8370, -3.4994, -0.8612, -6.2695, -1.5641, -2.1971, -3.4395, 1.5860, -1.2538, -4.2586, -2.6590, -0.0466, 0.8529, 4.9849, 1.5617, -1.5043, -1.4688, -3.5944, -1.1353, -0.3632, -0.0476, -0.2123, -0.2459, 0.6217, -0.1231, -0.4535, -0.3079, -0.2105, 0.1109, -1.3662, -1.2338, -0.8944, -0.2116, -1.3364, 0.0112, -4.5557, 0.7114, -11.7490, 0.7769, -10.5673, 2.3058, -22.8776, -19.6252])

#set_params([-276.5553, -2178.0041, -1476.8761, -2145.9814, -2134.3388, -1492.0933, -5263.3282, -6646.1433, -5762.5577, -6038.5416, -5997.5922, -906.7790, -870.2295, -2840.4853, -3096.5070, -3300.9217, -3094.9185, -2389.3313, -3191.5833, -3087.4791, -2390.4996, -1450.0707, -1106.9159, -2300.7577, 27.3770, 51.0279, 94.1479, 151.5242, 85.3457, 83.6003, 124.7446, 332.4315, 240.5733, 304.7181, 254.5180, 280.4402, 143.1497, 174.7402, 131.0381, 189.2199, 133.0653, 130.1817, 179.5711, 39.7892, 28.6761, 58.8558, 81.1234, 80.2488, 4.3161, 1.7593, 1.7083, 1.9638, 3.0650, 3.4756, 3.1074, 2.5590, 1.9020, 1.8566, 2.1257, 2.4230, 1.8100, 2.3309, 11.4172, 8.1034, 15.9232, 8.1756, 0.0745, 15.7974, 34.0590, 74.9336, 5.4467, 34.9312, 16.1070, 34.8573,
#            56.0, 0.0000, -6.7029, 0.2270,
#            50.5, 14.9215, 0.8290, 2.8572, 24.0050, 4.5439, -5.3093, -10.7218, 4.8370, -3.4994, -0.8612, -6.2695, -1.5641, -2.1971, -3.4395, 1.5860, -1.2538, -4.2586, -2.6590, -0.0466, 0.8529, 4.9849, 1.5617, -1.5043, -1.4688, -3.5944, -1.1353, -0.3632, -0.0476, -0.2123, -0.2459, 0.6217, -0.1231, -0.4535, -0.3079, -0.2105, 0.1109, -1.3662, -1.2338, -0.8944, -0.2116, -1.3364, 0.0112, -4.5557, 0.7114, -11.7490, 0.7769, -10.5673, 2.3058, -22.8776, -19.6252])


print(gt.energy_interaction)

# KLB-1 (Takahashi, 1986; Walter, 1998; Holland et al., 2013)
KLB_1_composition = {'Si': 39.4,
                     'Al': 2.*2.,
                     'Ca': 3.3,
                     'Mg': 49.5,
                     'Fe': 5.2 + 5.,
                     'Na': 0.26*2.,
                     'O': 39.4*2. + 2.*3. + 3.3 + 49.5 + 5.2 + 0.26} # reduced starting mix + Fe

# MORB (Litasov et al., 2005)
MORB_composition = {'Si': 53.9,
                    'Al': 9.76*2.,
                    'Ca': 13.0,
                    'Mg': 12.16,
                    'Fe': 8.64 + 5.,
                    'Na': 2.54*2.,
                    'O': 53.9*2. + 9.76*3. + 13.0 + 12.16 + 8.64 + 2.54} # reduced starting mix + Fe



Rohrbach_composition = {'Si': 45.4,
                        'Al': 7.3,
                        'Ca': 3.9,
                        'Mg': 20.9,
                        'Fe': 20.9 + 5.,
                        'Na': 0.9,
                        'O': 149.3} # reduced starting mix + Fe



mars_DW1985 = burnman.Composition({'Na2O': 0.5,
                                   'CaO': 2.46,
                                   'FeO': 18.47,
                                   'MgO': 30.37,
                                   'Al2O3': 3.55,
                                   'SiO2': 44.65,
                                   'Fe': 10.}, 'weight')
mars_DW1985.renormalize('atomic', 'total', 100.)
mars_composition = dict(mars_DW1985.atomic_composition)


# Abbreviate ringwoodite solution
rw = child_solutions['ring']

T0 = 1750.

plot_KLB = True
plot_MORB = True
plot_mars = False

if plot_KLB:

    # KLB-1 first
    ol.guess = np.array([0.9, 0.1])
    wad.guess = np.array([0.87, 0.13])
    rw.guess = np.array([0.84, 0.16])
    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])
    cpx_od.guess = np.array([0.7, 0.1, 0.05, 0.02, 0.05, 0.03, 0.05])
    
    
    P0 = 13.e9
    composition = KLB_1_composition
    assemblage = burnman.Composite([ol, gt, cpx_od, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]
    
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   store_iterates=False)
    
    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    wad.set_composition(wad.guess)
    assemblage = burnman.Composite([ol, gt, cpx_od, fcc_iron, wad], fs)
    assemblage.n_moles = n
    assemblage.set_state(sol.x[0], sol.x[1])
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    
    P_wad_in = assemblage.pressure
    
    
    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_ol_out = assemblage.pressure
    
    assemblage = burnman.Composite([wad, cpx_od, fcc_iron, gt])
    equality_constraints = [('T', T0), ('phase_proportion', (cpx_od, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_assemblage=True,
                                   store_iterates=False)
    P_cpx_out = assemblage.pressure
    
    P0 = 16.e9
    composition = KLB_1_composition
    assemblage = burnman.Composite([wad, gt, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]
    
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)
    
    rw.set_composition(rw.guess)
    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    assemblage = burnman.Composite([wad, gt, fcc_iron, rw], fs)
    assemblage.n_moles = n
    assemblage.set_state(sol.x[0], sol.x[1])
    
    equality_constraints = [('T', T0), ('phase_proportion', (rw, 0.0))]
    
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_rw_in = assemblage.pressure
    
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    
    P_wad_out = assemblage.pressure
    
    
    # ol-cpx-gt-iron
    
    assemblage = burnman.Composite([ol, cpx_od, fcc_iron, gt])
    pressures = np.linspace(10.e9, P_wad_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_ol_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)
    
    
    assemblage = burnman.Composite([wad, cpx_od, fcc_iron, gt])
    pressures = np.linspace(P_ol_out, P_ol_out+0.1e9, 2)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                            store_assemblage=True,
                                            store_iterates=False)
    
    
    assemblage = burnman.Composite([wad, fcc_iron, gt])
    pressures = np.linspace(P_cpx_out, P_rw_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                        store_assemblage=True,
                                        store_iterates=False)
    
    
    
    assemblage = burnman.Composite([rw, fcc_iron, gt])
    pressures = np.linspace(P_wad_out, 20.e9, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_rw, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       store_assemblage=True,
                                       store_iterates=False)
    
    # Plotting
    pressures = []
    Fe3 = []
    FeT = []
    x_dmaj = []
    x_nmaj = []
    for sols in [sols_ol_cpx, sols_wad_cpx, sols_wad, sols_rw]:
        pressures.extend([sol.assemblage.pressure
                          for sol in sols if sol.success])
        c_gt = np.array([sol.assemblage.phases[-1].molar_fractions
                         for sol in sols if sol.success])
        Fe3.extend([c[3]*2./(c[1]*3. + c[3]*2.) for c in c_gt])
        FeT.extend([(c[1]*3. + c[3]*2.) for c in c_gt])
        x_dmaj.extend([c[4] for c in c_gt])
        x_nmaj.extend([c[5] for c in c_gt])


    pressures = np.array(pressures)
    
    plt.style.use('ggplot')
    plt.plot(pressures/1.e9, x_dmaj, label='p(Mg$_3$(MgSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, x_nmaj, label='p(NaMg$_2$(AlSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, FeT, label='Fe atoms per 12 O')
    plt.plot(pressures/1.e9, Fe3, label='Fe$^{{3+}}$/(Fe$^{{2+}}$ + Fe$^{{3+}}$)')
    plt.legend()

    for P in [P_wad_in, P_ol_out, P_cpx_out, P_rw_in, P_wad_out]:
        plt.plot(np.array([P, P])/1.e9, [0., 0.6], color='k', linestyle=':')

    field_labels = [[10.e9, P_wad_in, 'cpx+ol+gt'],
                    [P_ol_out, P_cpx_out, 'cpx+wad+gt'],
                    [P_cpx_out, P_rw_in, 'wad+gt'],
                    [P_wad_out, 20.e9, 'rw+gt']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.3, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')
        
    plt.title('Garnet in iron-saturated KLB-1 peridotite at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(10.,20.)
    plt.ylim(0.,0.6)
    plt.savefig('KLB-1_gt_Fe_saturated.pdf')
    plt.show()


if plot_MORB:

    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])

    cpx = cpx_od # child_solutions['nocfs_cpx']
    
    cpx.guess = np.array([0.5, 0.1, 0.05, 0.05, 0.05, 0.2, 0.05])
    #cpx.guess = np.array([0.5, 0.1, 0.05, 0.05, 0.2, 0.05])
    
    P0 = 10.e9
    composition = MORB_composition
    assemblage = burnman.Composite([gt, cpx, stv, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]
    
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)
    
    equality_constraints = [('T', T0), ('phase_proportion', (cpx, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_cpx_out = sol.x[0]


    pressures = np.linspace(10.e9, (P_cpx_out+10.e9)/3., 21) # hard to find solution for whole range
    assemblage = burnman.Composite([cpx, stv, fcc_iron, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_gt_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)
    
    pressures = np.linspace(P_cpx_out, 20.e9, 21)    
    assemblage = burnman.Composite([stv, fcc_iron, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_gt, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       store_assemblage=True,
                                       store_iterates=False)

    
    # Plotting
    pressures = []
    Fe3 = []
    FeT = []
    x_dmaj = []
    x_nmaj = []
    for sols in [sols_gt_cpx, sols_gt]:
        pressures.extend([sol.assemblage.pressure
                          for sol in sols if sol.success])
        c_gt = np.array([sol.assemblage.phases[-1].molar_fractions
                         for sol in sols if sol.success])
        Fe3.extend([c[3]*2./(c[1]*3. + c[3]*2.) for c in c_gt])
        FeT.extend([(c[1]*3. + c[3]*2.) for c in c_gt])
        x_dmaj.extend([c[4] for c in c_gt])
        x_nmaj.extend([c[5] for c in c_gt])


    pressures = np.array(pressures)
    
    plt.style.use('ggplot')
    plt.plot(pressures/1.e9, x_dmaj, label='p(Mg$_3$(MgSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, x_nmaj, label='p(NaMg$_2$(AlSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, FeT, label='Fe atoms per 12 O')
    plt.plot(pressures/1.e9, Fe3, label='Fe$^{{3+}}$/(Fe$^{{2+}}$ + Fe$^{{3+}}$)')
    plt.legend()

    for P in [P_cpx_out]:
        plt.plot(np.array([P, P])/1.e9, [0., 1.], color='k', linestyle=':')

    field_labels = [[10.e9, P_cpx_out, 'cpx+gt+stv'],
                    [P_cpx_out, 20.e9, 'gt+stv']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.5, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')
        
    plt.title('Garnet in iron-saturated MORB at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(10.,20.)
    plt.ylim(0.,1.)
    plt.savefig('MORB_gt_Fe_saturated.pdf')
    plt.show()

    

if plot_mars:

    # KLB-1 first
    ol.guess = np.array([0.9, 0.1])
    wad.guess = np.array([0.87, 0.13])
    rw.guess = np.array([0.84, 0.16])
    gt.guess = np.array([0.6, 0.2, 0.15, 0.03, 0.01, 0.01])
    cpx_od.guess = np.array([0.7, 0.1, 0.05, 0.02, 0.05, 0.03, 0.05])
    
    
    P0 = 13.e9
    composition = mars_composition
    assemblage = burnman.Composite([ol, gt, cpx_od, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]
    
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   store_iterates=False)
    
    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    wad.set_composition(wad.guess)
    assemblage = burnman.Composite([ol, gt, cpx_od, fcc_iron, wad], fs)
    assemblage.n_moles = n
    assemblage.set_state(sol.x[0], sol.x[1])
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    
    P_wad_in = assemblage.pressure
    
    
    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_ol_out = assemblage.pressure
    
    assemblage = burnman.Composite([wad, cpx_od, fcc_iron, gt])
    equality_constraints = [('T', T0), ('phase_proportion', (cpx_od, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_assemblage=True,
                                   store_iterates=False)
    P_cpx_out = assemblage.pressure
    
    P0 = 16.e9
    composition = KLB_1_composition
    assemblage = burnman.Composite([wad, gt, fcc_iron])
    equality_constraints = [('P', P0), ('T', T0)]
    
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)
    
    rw.set_composition(rw.guess)
    fs = assemblage.molar_fractions
    fs.append(0.)
    n = assemblage.n_moles
    assemblage = burnman.Composite([wad, gt, fcc_iron, rw], fs)
    assemblage.n_moles = n
    assemblage.set_state(sol.x[0], sol.x[1])
    
    equality_constraints = [('T', T0), ('phase_proportion', (rw, 0.0))]
    
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_rw_in = assemblage.pressure
    
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_state_from_assemblage=True,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    
    P_wad_out = assemblage.pressure
    
    
    # ol-cpx-gt-iron
    
    assemblage = burnman.Composite([ol, cpx_od, fcc_iron, gt])
    pressures = np.linspace(10.e9, P_wad_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_ol_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           store_assemblage=True,
                                           store_iterates=False)
    
    
    assemblage = burnman.Composite([wad, cpx_od, fcc_iron, gt])
    pressures = np.linspace(P_ol_out, P_ol_out+0.1e9, 2)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad_cpx, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                            store_assemblage=True,
                                            store_iterates=False)
    
    
    assemblage = burnman.Composite([wad, fcc_iron, gt])
    pressures = np.linspace(P_cpx_out, P_rw_in, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_wad, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                        store_assemblage=True,
                                        store_iterates=False)
    
    
    
    assemblage = burnman.Composite([rw, fcc_iron, gt])
    pressures = np.linspace(P_wad_out, 20.e9, 21)
    equality_constraints = [('P', pressures), ('T', T0)]
    sols_rw, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       store_assemblage=True,
                                       store_iterates=False)
    
    # Plotting
    pressures = []
    Fe3 = []
    FeT = []
    x_dmaj = []
    x_nmaj = []
    for sols in [sols_ol_cpx, sols_wad_cpx, sols_wad, sols_rw]:
        pressures.extend([sol.assemblage.pressure
                          for sol in sols if sol.success])
        c_gt = np.array([sol.assemblage.phases[-1].molar_fractions
                         for sol in sols if sol.success])
        Fe3.extend([c[3]*2./(c[1]*3. + c[3]*2.) for c in c_gt])
        FeT.extend([(c[1]*3. + c[3]*2.) for c in c_gt])
        x_dmaj.extend([c[4] for c in c_gt])
        x_nmaj.extend([c[5] for c in c_gt])


    pressures = np.array(pressures)
    
    plt.style.use('ggplot')
    plt.plot(pressures/1.e9, x_dmaj, label='p(Mg$_3$(MgSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, x_nmaj, label='p(NaMg$_2$(AlSi)Si$_3$O$_{{12}}$)')
    plt.plot(pressures/1.e9, FeT, label='Fe atoms per 12 O')
    plt.plot(pressures/1.e9, Fe3, label='Fe$^{{3+}}$/(Fe$^{{2+}}$ + Fe$^{{3+}}$)')
    plt.legend()

    for P in [P_wad_in, P_ol_out, P_cpx_out, P_rw_in, P_wad_out]:
        plt.plot(np.array([P, P])/1.e9, [0., 0.6], color='k', linestyle=':')

    field_labels = [[10.e9, P_wad_in, 'cpx+ol+gt'],
                    [P_ol_out, P_cpx_out, 'cpx+wad+gt'],
                    [P_cpx_out, P_rw_in, 'wad+gt'],
                    [P_wad_out, 20.e9, 'rw+gt']]

    for P0, P1, lbl in field_labels:
        plt.text((P0+P1)/2./1.e9, 0.3, lbl,
                 horizontalalignment='center',
                 verticalalignment='center')
        
    plt.title('Garnet in iron-saturated Martian peridotite at {0} K'.format(T0))
    plt.xlabel('P (GPa)')
    plt.ylabel('composition')
    plt.xlim(10.,20.)
    plt.ylim(0.,0.6)
    plt.savefig('mars_gt_Fe_saturated.pdf')
    plt.show()
