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
                  ['cfs', 'H_0', cfs.params['H_0'], 1.e3], 
                  ['cats', 'H_0', cats.params['H_0'], 1.e3], # jadeite is a standard
                  ['aeg', 'H_0', aeg.params['H_0'], 1.e3],
                  
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
                 ['gt', 'E', 3, 0,  gt.energy_interaction[3][0], 1.e3], # andr-dmaj
                 ['gt', 'E', 3, 1,  gt.energy_interaction[3][1], 1.e3], # andr-nagt
                 ['gt', 'E', 4, 0,  gt.energy_interaction[4][0], 1.e3], # dmaj-nagt
                 
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
"""

#minimize_func(get_params(), assemblages)

#######################
### PUT PARAMS HERE ###
#######################

set_params([-268.2893, -2173.6798, -1477.0176, -2145.9753, -2135.2978, -1472.2961, -5260.1095, -6651.5345, -5777.1883, -6022.8897, -5985.0066, -906.9429, -867.1506, -2843.2671, -3093.0893, -2386.6104, -3290.1567, -2583.4301, -3089.9910, -2390.5536, -3196.9785, -3082.6958, -2392.9751, -1450.2445, -1108.9245, -2298.8389, 26.9154, 55.8037, 94.1575, 151.4109, 85.3729, 82.3261, 136.1304, 339.0583, 246.7854, 316.4158, 263.0183, 260.6113, 143.7425, 174.2969, 132.0212, 187.9240, 128.3748, 131.1242, 177.3001, 39.6742, 30.1787, 58.1165, 81.3849, 80.8820, 4.3056, 1.6974, 1.6460, 2.0136, 3.0622, 3.3211, 2.8595, 2.5672, 2.1119, 1.7586, 2.2348, 2.1349, 1.8189, 2.2069, 11.9162, 6.3385, 18.3705, 9.4134, -1.0310, 12.1618, 31.7312, 70.9614, 2.2397, 37.4117, -7.0633, 12.1837, 50.5163, -6.8406, 0.0008, -4.4086, 49.0392, 0.0007, 0.0001, 3.5491, 41.4159, -0.0022, 0.0006, 0.0001, -0.0001, -12.3700, 0.0018, -4.4838, -1.2763, -0.2953, -3.1070, -0.3624, 0.6855, -4.1348, -1.4310, -0.4138, -1.4105, 0.3474, -0.0089, 1.2406, 0.2761, 1.5229, -1.2456, -1.3212, -0.9352, -0.7752, -0.2439, 0.1818, -0.1193, -0.2447, 0.7783, -0.1346, -0.3659, -0.2754, -0.2031, -0.0541, -1.2017, -1.2669, -0.4274, -0.3505, -0.0001, -0.0000, -0.0001, 0.0001, -0.0015, 0.0001, -0.0007, -0.0048, -0.0039, -0.0018])

set_params([-268.3076, -2173.7024, -1476.9758, -2145.9640, -2135.3040, -1472.3196, -5260.1108, -6651.5384, -5777.1870, -6022.8797, -5985.0257, -906.9429, -867.1497, -2843.2729, -3093.1033, -2386.6115, -3290.1609, -2583.4305, -3089.9893, -2390.5578, -3196.9828, -3082.6927, -2392.9762, -1450.2461, -1108.9228, -2298.8372, 26.8927, 55.8333, 94.1811, 151.3786, 85.3574, 82.3399, 136.1391, 339.0565, 246.7899, 316.4127, 262.9950, 260.6441, 143.7224, 174.3105, 132.0191, 187.9293, 128.3835, 131.1164, 177.3017, 39.6742, 30.1767, 58.1200, 81.3814, 80.8788, 4.3066, 1.6979, 1.6457, 1.9389, 3.0644, 3.3041, 2.8999, 2.4709, 2.1219, 1.7570, 2.2333, 2.2637, 1.7983, 2.2318, 11.8852, 6.3634, 18.3716, 9.4012, -1.0307, 12.1577, 31.7270, 70.9613, 2.2422, 37.4155, -7.0663, 12.1844, 50.5155, -6.8424, 0.0032, -4.4090, 49.0398, 0.0024, 0.0005, 3.5490, 41.4175, -0.0086, 0.0017, 0.0003, -0.0003, -12.3714, 0.0071, -4.4828, -1.2766, -0.2954, -3.1076, -0.3625, 0.6852, -4.1351, -1.4311, -0.4139, -1.4107, 0.3477, -0.0089, 1.2408, 0.2768, 1.5518, -1.2502, -1.3201, -0.9358, -0.7846, -0.2405, 0.1843, -0.1030, -0.2447, 0.7976, -0.1346, -0.3659, -0.2754, -0.2031, -0.0541, -1.2017, -1.2671, -0.4274, -0.3505, -0.0001, -0.0000, -0.0005, 0.0001, -0.0058, 0.0001, -0.0026, -0.0127, -0.0153, -0.0071])

set_params([-278.8774, -2175.3133, -1476.9771, -2142.8000, -2131.4548, -1493.9261, -5266.9519, -6651.8637, -5769.7186, -6023.3629, -5995.1657, -906.8865, -868.4004, -2843.5120, -3094.0009, -2391.0765, -3290.1266, -2587.7727, -3090.4919, -2389.5276, -3193.3971, -3082.9655, -2390.9884, -1448.6982, -1108.8628, -2299.9885, 27.1671, 49.9705, 94.0233, 151.4379, 86.0979, 83.9042, 123.8391, 332.9305, 243.2975, 303.0178, 259.0832, 276.9434, 143.5690, 174.6087, 132.2791, 188.8331, 128.4640, 131.4910, 178.9328, 39.7113, 29.2753, 58.6601, 79.4929, 79.8175, 4.2957, 1.7951, 1.6878, 1.9741, 3.0808, 3.4054, 3.0296, 2.5743, 1.9409, 2.3348, 2.1311, 2.4879, 1.8010, 2.2890, 11.4651, 8.3122, 15.9210, 7.9208, 0.0433, 6.3134, 30.4510, 73.5624, 5.9741, 36.6943, -3.5942, 13.8462, 48.1324, -14.1920, -5.7973, -6.2162, 52.8411, 10.3376, 1.2018, 1.9737, 38.7009, 3.3961, 5.7343, 0.6047, 0.8430, -16.9442, -9.5292, 1.3093, -3.9989, -1.0213, -6.2228, -1.8615, -2.6074, -3.4490, 1.6895, -1.4468, -4.7845, -2.6465, -0.0693, 0.7423, 3.9473, 1.5835, -1.3905, -0.9658, -3.4137, -1.0198, -0.3002, 0.0621, -0.1626, -0.2424, 0.7293, -0.2157, -0.6248, -0.3595, -0.2581, 0.0286, -1.6031, -1.1005, -0.8884, -0.4211, -0.0000, -0.1620, -4.9855, -0.0000, -14.7294, -0.0000, -12.0516, 4.2989, -19.6626, -15.9005])

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
