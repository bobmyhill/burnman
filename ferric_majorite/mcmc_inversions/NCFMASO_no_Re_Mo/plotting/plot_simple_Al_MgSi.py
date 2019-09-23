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

#set_params([-277.9289, -2178.4294, -1476.5730, -2145.3713, -2133.7708, -1496.9510, -5266.8040, -6647.9477, -5760.0003, -6036.3540, -5999.6693, -906.8381, -868.3694, -2844.5957, -3097.1339, -3298.6196, -3093.8825, -2389.9469, -3189.9214, -3086.3529, -2391.5186, -1451.3627, -1110.0027, -2299.1332, 27.3425, 49.9783, 93.6096, 151.7517, 85.6982, 84.0574, 120.7708, 330.1900, 242.2392, 308.7630, 254.7174, 280.4514, 143.8165, 173.4787, 131.6388, 188.5696, 134.2121, 130.8634, 178.6002, 39.7498, 29.8670, 58.6342, 79.9240, 80.6141, 4.3137, 1.7445, 1.6439, 1.9799, 3.0355, 3.3717, 3.1009, 2.6274, 1.8672, 1.8938, 2.1063, 2.1957, 1.8216, 2.3287, 11.2511, 8.6212, 15.6546, 7.8240, 1.2429, 16.9751, 32.2773, 75.3650, 7.6869, 36.8417, 10.5002, 32.5110, 83.9359, -0.0000, -7.6283, -0.0047, 64.8617, 22.9223, 1.3461, 0.1726, 32.6253, 8.3034, -9.7742, -12.6775, 7.3309, -3.1251, -0.9980, -3.7262, -1.7450, -1.9245, -3.1731, 0.8666, -1.3922, -3.8635, -1.1810, -0.1081, 0.5997, 1.7036, 1.0278, -2.0387, -1.8788, -2.1538, -1.6811, -0.9887, -0.5692, -0.7227, -0.0971, 0.0991, -0.1902, -0.5842, -0.4132, -0.2523, 0.1659, -1.5691, -1.5567, -1.0215, -0.2975, -1.6260, 0.0317, -7.4254, 0.8481, -7.9251, 0.7649, -15.9500, -1.3429, -14.8385, -13.7353])

#set_params([-284.9271, -2173.4544, -1476.9320, -2139.1857, -2127.1963, -1502.1781, -5268.5663, -6657.0636, -5740.5142, -6027.0743, -6007.1924, -906.7769, -860.8716, -2844.5075, -3091.5406, -3298.1982, -3089.7719, -2389.6966, -3188.9109, -3082.5331, -2391.7249, -1449.2091, -1124.8305, -2300.9794, 26.9916, 46.5062, 93.7713, 151.4743, 87.0051, 85.6837, 117.1518, 333.9261, 244.7910, 313.1773, 254.9982, 275.4573, 143.1428, 173.9167, 131.1259, 188.7163, 133.6432, 130.2133, 178.4129, 39.7834, 33.5729, 58.5792, 71.8479, 81.7042, 4.2993, 1.7193, 1.6406, 1.9732, 3.0733, 3.4799, 3.0200, 2.7053, 1.8705, 2.0625, 2.1192, 2.1423, 1.7902, 2.2451, 11.5273, 8.4868, 16.5363, 8.8873, 1.4624, 6.9388, 30.3700, 78.7480, 5.2618, 36.2311, -1.2803, 18.0602, 74.4328, 0.0004, -4.5926, 0.6028, 69.9143, 45.7611, 3.0811, -15.2230, 50.1432, 24.0087, -23.2015, -11.4685, 14.3193, -1.2487, -0.2679, -3.7380, -0.5251, -0.8651, -3.6077, -0.3157, -0.5240, -2.1260, 1.1209, -0.0732, 0.6928, 1.3831, 1.1609, -1.9138, -1.8170, -2.2909, -1.5552, -0.9322, -0.4308, -0.6107, 0.0676, 0.2317, -0.4526, -1.1791, -0.7004, -0.3232, 0.1203, -1.8159, -1.4977, -1.5619, -0.6052, -1.7954, 0.0079, -7.2624, 1.2397, -8.5016, 0.8525, -12.8373, -2.1276, -14.9018, -13.7944])

# KLB-1 (Takahashi, 1986; Walter, 1998; Holland et al., 2013)
KLB_1_composition = {'Si': 39.4,
                     'Al': 2.*2.,
                     'Ca': 3.3,
                     'Mg': 49.5,
                     'Fe': 5.2 + 5.,
                     'Na': 0.26*2.,
                     'O': 39.4*2. + 2.*3. + 3.3 + 49.5 + 5.2 + 0.26} # reduced starting mix + Fe



gt = transform_solution_to_new_basis(gt,
                                     np.array([[1., 0., 0., 0., 0., 0.],
                                               [1., 0., -1., 1., 0., 0.],
                                               [0., 1., -1., 1., 0., 0.],
                                               [0., 0., 0., 0., 1., 0.]]),
                                     solution_name='py-kho-sk-maj garnet')


plt.style.use('ggplot')
fig = plt.figure()
ax = [fig.add_subplot(2, 1, i) for i in range(1,3)]

# Abbreviate ringwoodite solution
rw = child_solutions['ring']

T0 = 1750.

#x_Als = [0.01]
x_Al = 0.01
#for x_Al in x_Als:
for n_Fe in [0.05, 0.1, 0.15, 0.2, 0.25]:
    eps = 1.e-1
    composition = {'Si': 1. + 1.*eps,
                   'Al': x_Al,
                   'Mg': (1. - n_Fe) + 2.*eps,
                   'Fe': n_Fe + 1.,
                   'O': 3. + x_Al*1.5 + 4.*eps} # reduced starting mix + Fe

    
    # KLB-1 first
    ol.guess = np.array([0.9, 0.1])
    wad.guess = np.array([0.87, 0.13])
    rw.guess = np.array([0.84, 0.16])
    gt.guess = np.array([0.6, 0.2, 0.15, 0.05])
    
    
    pressures = np.linspace(10.e9, 14.e9, 21)


    assemblage = burnman.Composite([ol, fcc_iron, gt, wad])
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   store_iterates=False)

    ol.guess = np.array(ol.molar_fractions)
    wad.guess = np.array(wad.molar_fractions)
    gt.guess = np.array(gt.molar_fractions)
    
    P_wad_in = assemblage.pressure
    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_ol_out = assemblage.pressure
    rw.set_composition(rw.guess)
    assemblage = burnman.Composite([rw, fcc_iron, gt, wad], assemblage.molar_fractions)
    assemblage.set_state(P_ol_out, T0)
    equality_constraints = [('T', T0), ('phase_proportion', (rw, 0.))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    
    rw.guess = np.array(rw.molar_fractions)
    
    P_rw_in = assemblage.pressure
    equality_constraints = [('T', T0), ('phase_proportion', (wad, 0.))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                   initial_composition_from_assemblage=True,
                                   store_iterates=False)
    P_wad_out = assemblage.pressure

    
    pressures = np.linspace(10.e9, P_wad_in, 10)
    assemblage = burnman.Composite([ol, fcc_iron, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    
    sols_ol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_state_from_assemblage=True,
                                       store_assemblage=True,
                                       store_iterates=False)
    
    pressures = np.linspace(P_ol_out, P_rw_in, 10)
    assemblage = burnman.Composite([wad, fcc_iron, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    
    sols_wad, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_state_from_assemblage=True,
                                       store_assemblage=True,
                                       store_iterates=False)
    
    pressures = np.linspace(P_wad_out, 20.e9, 10)
    assemblage = burnman.Composite([rw, fcc_iron, gt])
    equality_constraints = [('P', pressures), ('T', T0)]
    
    sols_rw, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_state_from_assemblage=True,
                                       store_assemblage=True,
                                       store_iterates=False, tol=1.e-5)

    
    # Plotting
    pressures = []
    Fe3 = []
    FeT = []
    FeOct = []
    AlOct = []
    x_dmaj = []
    for sols in [sols_ol, sols_wad, sols_rw]:
        pressures.extend([sol.assemblage.pressure
                          for sol in sols if sol.success])
        c_gt = np.array([sol.assemblage.phases[-1].molar_fractions
                         for sol in sols if sol.success])
        Fe3.extend([(c[1] + c[2])*2./(c[1]*2. + c[2]*5.) for c in c_gt])
        FeT.extend([(c[1]*2. + c[2]*5.) for c in c_gt])
        AlOct.extend([c[0]*2. for c in c_gt])
        FeOct.extend([(c[1] + c[2])*2. for c in c_gt])
        x_dmaj.extend([c[3] for c in c_gt])


    pressures = np.array(pressures)
    AlOct = np.array(AlOct)
    FeOct = np.array(FeOct)
    
    #plt.plot(pressures/1.e9, x_dmaj, label='p(Mg$_3$(MgSi)Si$_3$O$_{{12}}$)')
    #plt.plot(pressures/1.e9, x_nmaj, label='p(NaMg$_2$(AlSi)Si$_3$O$_{{12}}$)')
    #plt.plot(pressures/1.e9, FeT, label='Fe atoms per 12 O')
    #plt.plot(pressures/1.e9, Fe3, label='Fe$^{{3+}}$/(Fe$^{{2+}}$ + Fe$^{{3+}}$)')
    ax[0].plot(pressures/1.e9, Fe3, label='{0:.1f}'.format(composition['Al']))
    ax[1].plot(pressures/1.e9, FeOct/AlOct, label='{0:.1f}'.format(composition['Al']))

    
for i in range(2):
    ax[i].legend()        
    ax[i].set_xlabel('P (GPa)')
    ax[i].set_xlim(10.,20.)

ax[0].set_ylabel('Fe$^{{3+}}$/(Fe$^{{2+}}$ + Fe$^{{3+}}$)')
ax[1].set_ylabel('$^{VI}$Fe/$^{VI}$Al')
plt.ylim(0.,)
plt.savefig('gt_simple_Fe_saturated.pdf')
plt.show()
