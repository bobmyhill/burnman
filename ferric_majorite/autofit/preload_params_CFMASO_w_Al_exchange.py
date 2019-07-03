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

endmember_args = [['wus', 'H_0', wus.params['H_0'], 1.e3], 
                  ['fo',  'H_0', fo.params['H_0'],  1.e3],
                  ['fa',  'H_0', fa.params['H_0'],  1.e3],
                  ['mwd', 'H_0', mwd.params['H_0'], 1.e3], # fwd H_0 and S_0 obtained in set_params_from_special_constraints()
                  ['mrw', 'H_0', mrw.params['H_0'], 1.e3],
                  ['frw', 'H_0', frw.params['H_0'], 1.e3],
                  
                  ['alm', 'H_0', alm.params['H_0'], 1.e3],
                  ['gr',   'H_0', gr.params['H_0'], 1.e3],
                  ['andr', 'H_0', andr.params['H_0'], 1.e3],
                  ['dmaj', 'H_0', dmaj.params['H_0'], 1.e3],
                  
                  ['coe', 'H_0', coe.params['H_0'], 1.e3],
                  ['stv', 'H_0', stv.params['H_0'], 1.e3],
                  
                  ['hed', 'H_0', hed.params['H_0'], 1.e3], # diopside is a standard
                  ['cen', 'H_0', cen.params['H_0'], 1.e3], 
                  ['cats', 'H_0', cats.params['H_0'], 1.e3],
                  
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
                 ['gt', 'E', 1, 0,  gt.energy_interaction[1][0], 1.e3], # alm-gr
                 ['gt', 'E', 1, 1,  gt.energy_interaction[1][1], 1.e3], # alm-andr
                 ['gt', 'E', 2, 0,  gt.energy_interaction[2][0], 1.e3], # gr-andr
                 ['gt', 'E', 2, 1,  gt.energy_interaction[2][1], 1.e3], # gr-dmaj
                 ['gt', 'V', 0, 2,  gt.volume_interaction[0][2], 1.e-7], # py-andr
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
experiment_uncertainties.extend([['Frost_2003_H1554', 'P', 0., 0.5e9],
                                 ['Frost_2003_H1555', 'P', 0., 0.5e9],
                                 ['Frost_2003_H1556', 'P', 0., 0.5e9],
                                 ['Frost_2003_H1582', 'P', 0., 0.5e9],
                                 ['Frost_2003_S2773', 'P', 0., 0.5e9],
                                 ['Frost_2003_V170', 'P', 0., 0.5e9],
                                 ['Frost_2003_V171', 'P', 0., 0.5e9],
                                 ['Frost_2003_V175', 'P', 0., 0.5e9],
                                 ['Frost_2003_V179', 'P', 0., 0.5e9]])

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
from Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp import Klemme_ONeill_2000_CMAS_assemblages

# CFMS
from Perkins_Vielzeuf_1992_CFMS_ol_cpx import Perkins_Vielzeuf_1992_CFMS_assemblages

# FASO
from Woodland_ONeill_1993_FASO_alm_sk import Woodland_ONeill_1993_FASO_assemblages

# NCFMASO
from Rohrbach_et_al_2007_NCFMASO_gt_cpx import Rohrbach_et_al_2007_NCFMASO_assemblages
from Beyer_et_al_2019_NCFMASO import Beyer_et_al_2019_NCFMASO_assemblages

assemblages = [assemblage for assemblage_list in
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
                Gasparik_1992_MAS_assemblages # need oen-mgts, py-dmaj
               ]
               for assemblage in assemblage_list]


#minimize_func(get_params(), assemblages)
"""

#######################
### PUT PARAMS HERE ###
#######################

#set_params([-266.7804, -2175.8766, -1476.9576, -2148.7595, -2138.1450, -1472.1244, -5253.0046, -6641.5494, -5769.4822, -6038.5050, -906.9301, -871.6192, -2842.8178, -3097.3005, -3305.0463, -3093.4992, -2390.6796, -3196.1322, -3086.1310, -2393.0915, -1450.7721, -1106.3165, -2300.7583, 26.9869, 56.2912, 93.9918, 151.4445, 84.8269, 81.6946, 136.1583, 342.1085, 251.6326, 316.9522, 254.5720, 143.8573, 174.2266, 130.6519, 187.6635, 130.1545, 129.7794, 177.0437, 39.6837, 27.5593, 58.1144, 83.1318, 80.3368, 4.3100, 1.6345, 1.6390, 2.0266, 3.0693, 3.2748, 2.8200, 2.7503, 2.1251, 1.7549, 2.2453, 2.0366, 1.8537, 2.3216, 11.8694, 6.0950, 18.0447, 9.3693, -1.4707, 15.6215, 31.9467, 74.0480, 2.4138, 38.0345, 12.4288, 45.1317, 89.7620, 0.0002, -0.0268, 59.8595, 4.9230, -0.3956, -0.2324, -1.2723, -0.2376, -3.0395, -0.2977, 1.1121, -4.3050, -1.8686, -0.3433, -1.1053, 0.1909, -0.0138, 1.2041, 0.4605, 1.6597, -1.0868, -1.1535, -0.8684, -0.6280, 0.0451, 0.3404, 0.0306, -0.1910, 0.9185, -0.0627, -0.2040, -0.1698, -0.1201, 0.0200, -0.6842, -1.0907, -0.4407, -0.1269])

#set_params([-268.2959, -2175.7842, -1477.0217, -2148.1302, -2137.6182, -1471.9779, -5257.2674, -6644.6791, -5772.1761, -6034.6451, -906.9403, -868.0115, -2841.7610, -3098.9460, -3300.6949, -3093.1648, -2390.5051, -3197.9520, -3085.8146, -2392.6735, -1451.3346, -1109.0253, -2297.7740, 26.9949, 55.2483, 94.1330, 151.3929, 85.2524, 81.9740, 136.3216, 338.8895, 247.5420, 317.2331, 258.9174, 143.4681, 174.3482, 131.2203, 187.9939, 127.7867, 130.3424, 177.5567, 39.6795, 30.0145, 58.0535, 81.3470, 80.8912, 4.3074, 1.6368, 1.6458, 2.0270, 3.0892, 3.2664, 2.8184, 2.7527, 2.1096, 1.7580, 2.2191, 2.0680, 1.8257, 2.2192, 11.8957, 6.3861, 18.4045, 9.1989, -1.0141, 14.4250, 32.6131, 73.0522, 2.2400, 40.0336, 8.4225, 45.8550, 87.9559, 0.0004, -0.7733, 58.9992, 4.4419, -3.3730, -1.6776, -1.5787, -0.2750, -3.2527, -0.3923, 1.0250, -4.5481, -2.0683, -0.4178, -1.3004, 0.2879, -0.0099, 1.1850, 0.4266, 1.6064, -1.1336, -1.1843, -1.0750, -0.6723, -0.1172, 0.2833, -0.0240, -0.2334, 0.8688, -0.1373, -0.4388, -0.3907, -0.2768, -0.0010, -1.4989, -1.1243, -0.7449, -0.3147])

set_params([-268.4739, -2173.8834, -1477.0293, -2146.1451, -2135.5076, -1472.4608, -5260.1885, -6649.9451, -5776.6848, -6025.1763, -906.9269, -867.4287, -2843.1151, -3093.5685, -3290.2902, -3090.1364, -2390.5562, -3202.5809, -3082.8797, -2392.8622, -1450.4009, -1109.0735, -2300.5607, 26.9162, 55.6422, 94.1601, 151.3910, 85.4197, 82.3173, 136.0242, 338.7844, 247.4604, 316.4747, 262.0356, 143.6533, 174.2588, 132.0470, 187.9132, 125.8847, 131.1344, 177.3743, 39.6852, 30.0223, 58.0722, 81.3309, 79.8615, 4.3057, 1.6429, 1.6467, 2.0241, 3.0960, 3.2868, 2.8064, 2.7462, 2.0971, 1.7850, 2.2172, 2.0356, 1.8274, 2.2105, 11.9125, 6.3713, 18.2698, 9.3268, -0.9872, 14.5214, 32.0021, 72.1470, 2.2890, 37.6817, -7.3840, 14.3411,
            51.8055, -8.6978, -2.4498, 48.4941,
            3.6436, 41.0421, -10.3480, -5.2384, -1.3077, -0.1746, -3.0950, -0.2509, 0.8774, -4.1835, -1.6226, -0.3873, -1.3040, 0.2346, -0.0036, 1.1554, 0.4392, 1.5001, -1.2584, -1.3233, -1.0009, -0.7863, -0.2533, 0.1669, -0.1384, -0.2339, 0.7568, -0.1396, -0.4071, -0.2961, -0.2062, -0.0568, -1.2365, -1.2827, -0.4706, -0.3405])

set_params([-268.2971, -2173.6808, -1477.0197, -2145.9808, -2135.2944, -1472.2909, -5260.1103, -6651.5328, -5777.1891, -6022.8930, -906.9429, -867.1509, -2843.2652, -3093.0825, -3290.1546, -3089.9918, -2390.5528, -3196.9774, -3082.6969, -2392.9748, -1450.2440, -1108.9248, -2298.8392, 26.9002, 55.8164, 94.1615, 151.3983, 85.3803, 82.3219, 136.1358, 339.0609, 246.7833, 316.4176, 263.0261, 143.7501, 174.2925, 132.0224, 187.9230, 128.3724, 131.1270, 177.2996, 39.6742, 30.1794, 58.1153, 81.3856, 80.8826, 4.3066, 1.6461, 1.6469, 2.0243, 3.0975, 3.2948, 2.8069, 2.7461, 2.0954, 1.7596, 2.2170, 2.0343, 1.8271, 2.2018, 11.9223, 6.3374, 18.3702, 9.4179, -1.0312, 12.1629, 31.7327, 70.9614, 2.2388, 37.4108, -7.0621, 12.1838, 50.5163, -6.8399, -4.4084, 49.0389, 3.5495, 41.4156, -12.3699, -4.4843, -1.2761, -0.2953, -3.1068, -0.3623, 0.6856, -4.1347, -1.4310, -0.4138, -1.4104, 0.3473, -0.0089, 1.2405, 0.2759, 1.5159, -1.2456, -1.3220, -0.9350, -0.7744, -0.2450, 0.1796, -0.1234, -0.2447, 0.7730, -0.1346, -0.3659, -0.2754, -0.2031, -0.0541, -1.2017, -1.2669, -0.4274, -0.3505])

set_params([-268.6060, -2173.5492, -1477.0201, -2145.7415, -2135.0333, -1472.6349, -5259.7228, -6651.9929, -5780.1667, -6019.6750, -906.9379, -867.0594, -2843.2872, -3092.8758, -3286.7163, -3090.0164, -2390.4286, -3190.6156, -3082.7364, -2392.6614, -1450.2577, -1108.6901, -2299.1377, 26.8932, 55.6628, 94.1608, 151.3982, 85.4486, 82.4300, 135.8881, 339.4845, 246.5373, 316.4337, 264.3511, 143.7099, 174.2987, 131.9248, 188.0285, 128.4783, 131.0218, 177.5432, 39.6772, 30.1824, 58.0869, 81.5214, 80.2874, 4.3076, 1.6491, 1.6463, 2.0241, 3.0980, 3.2889, 2.8069, 2.7494, 2.0911, 1.7669, 2.2163, 2.0276, 1.8274, 2.2104, 11.8693, 6.2602, 18.0136, 9.2142, -1.1185, 3.4911, 32.0530, 60.9578, 2.2942, 37.4378, -11.7381, 7.0377, 42.5861, -6.6332, -10.7712, 52.6959, 3.3976, 41.3092, -24.8677, 0.9900, -0.9105, -0.1396, -3.1691, -0.3362, 0.7695, -4.1474, -1.2179, -0.3549, -1.1982, 0.2290, -0.0028, 1.3234, 0.3745, 1.4667, -1.2948, -1.3691, -1.1263, -0.8252, -0.2949, 0.1296, -0.1734, -0.2209, 0.7218, -0.0249, -0.2407, -0.1820, -0.1336, -0.0243, -1.1291, -1.2798, -0.2424, -0.2123])
