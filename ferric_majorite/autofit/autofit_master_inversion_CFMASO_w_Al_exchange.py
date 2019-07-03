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
from Gasparik_1989_CMAS_px_gt import Gasparik_1989_CMAS_assemblages
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
                Gasparik_1992_MAS_assemblages, # need oen-mgts, py-dmaj
                Gasparik_1989_MAS_assemblages, # need oen-mgts, py-dmaj
                Gasparik_1989_CMAS_assemblages # need oen-mgts, py-dmaj
               ]
               for assemblage in assemblage_list]


#minimize_func(get_params(), assemblages)

#######################
### PUT PARAMS HERE ###
#######################

#set_params([-267.5630, -2178.9874, -1477.0431, -2151.1850, -2140.7452, -1472.0685, -5251.6445, -6606.0805, -5806.1074, -906.8902, -869.3706, -3095.1238, -2390.1953, -3087.7713, -2392.5637, -1453.2125, -1110.7413, 27.2573, 55.0384, 94.1473, 151.3785, 85.3082, 81.9620, 136.3383, 339.5044, 255.0160, 316.3844, 132.3660, 188.3481, 131.4920, 177.7659, 39.7189, 29.7308, 58.0853, 80.5020, 4.3082, 1.6063, 1.6453, 2.0297, 3.0647, 3.2772, 2.8305, 2.7526, 2.1185, 1.7779, 2.2274, 2.0853, 1.8350, 2.2047, 11.8119, 6.1785, 18.0228, 8.7935, -1.2436, 28.5546, -24.6737, 84.6295, -12.4158, -17.1973, 31.0180, -1.8148, -0.3586, -3.5121, -0.5006, 1.2958, -5.1544, -2.7670, -0.3588, -1.1978, 0.1628, -0.0178, 1.3075, 0.6677, 1.6047, -1.1183, -1.1376, -1.1336, -0.6479, -0.1278, 0.2960, -0.0120, -0.2268, 0.8779, -0.0386, -0.1828, -0.0685, -0.1491, -0.0601, -1.3106, -1.0171, -0.0194, -0.2197])

#set_params([-266.7804, -2175.8766, -1476.9576, -2148.7595, -2138.1450, -1472.1244, -5253.0046, -6641.5494, -5769.4822, -6038.5050, -906.9301, -871.6192, -2842.8178, -3097.3005, -3305.0463, -3093.4992, -2390.6796, -3196.1322, -3086.1310, -2393.0915, -1450.7721, -1106.3165, -2300.7583, 26.9869, 56.2912, 93.9918, 151.4445, 84.8269, 81.6946, 136.1583, 342.1085, 251.6326, 316.9522, 254.5720, 143.8573, 174.2266, 130.6519, 187.6635, 130.1545, 129.7794, 177.0437, 39.6837, 27.5593, 58.1144, 83.1318, 80.3368, 4.3100, 1.6345, 1.6390, 2.0266, 3.0693, 3.2748, 2.8200, 2.7503, 2.1251, 1.7549, 2.2453, 2.0366, 1.8537, 2.3216, 11.8694, 6.0950, 18.0447, 9.3693, -1.4707, 15.6215, 31.9467, 74.0480, 2.4138, 38.0345, 12.4288, 45.1317, 89.7620, 0.0002, -0.0268, 59.8595, 4.9230, -0.3956, -0.2324, -1.2723, -0.2376, -3.0395, -0.2977, 1.1121, -4.3050, -1.8686, -0.3433, -1.1053, 0.1909, -0.0138, 1.2041, 0.4605, 1.6597, -1.0868, -1.1535, -0.8684, -0.6280, 0.0451, 0.3404, 0.0306, -0.1910, 0.9185, -0.0627, -0.2040, -0.1698, -0.1201, 0.0200, -0.6842, -1.0907, -0.4407, -0.1269])

#set_params([-268.2959, -2175.7842, -1477.0217, -2148.1302, -2137.6182, -1471.9779, -5257.2674, -6644.6791, -5772.1761, -6034.6451, -906.9403, -868.0115, -2841.7610, -3098.9460, -3300.6949, -3093.1648, -2390.5051, -3197.9520, -3085.8146, -2392.6735, -1451.3346, -1109.0253, -2297.7740, 26.9949, 55.2483, 94.1330, 151.3929, 85.2524, 81.9740, 136.3216, 338.8895, 247.5420, 317.2331, 258.9174, 143.4681, 174.3482, 131.2203, 187.9939, 127.7867, 130.3424, 177.5567, 39.6795, 30.0145, 58.0535, 81.3470, 80.8912, 4.3074, 1.6368, 1.6458, 2.0270, 3.0892, 3.2664, 2.8184, 2.7527, 2.1096, 1.7580, 2.2191, 2.0680, 1.8257, 2.2192, 11.8957, 6.3861, 18.4045, 9.1989, -1.0141, 14.4250, 32.6131, 73.0522, 2.2400, 40.0336, 8.4225, 45.8550, 87.9559, 0.0004, -0.7733, 58.9992, 4.4419, -3.3730, -1.6776, -1.5787, -0.2750, -3.2527, -0.3923, 1.0250, -4.5481, -2.0683, -0.4178, -1.3004, 0.2879, -0.0099, 1.1850, 0.4266, 1.6064, -1.1336, -1.1843, -1.0750, -0.6723, -0.1172, 0.2833, -0.0240, -0.2334, 0.8688, -0.1373, -0.4388, -0.3907, -0.2768, -0.0010, -1.4989, -1.1243, -0.7449, -0.3147])

#set_params([-267.9933, -2175.6482, -1477.0366, -2147.9553, -2137.3511, -1472.2071, -5257.2993, -6644.5061, -5772.3980, -6034.0139, -906.9163, -867.9480, -2841.8564, -3098.7711, -3300.7216, -3093.1707, -2390.4501, -3197.8647, -3085.8708, -2392.8390, -1451.1947, -1109.0806, -2298.0234, 27.0505, 55.5654, 94.1321, 151.3814, 85.2700, 82.1121, 136.1783, 339.0245, 247.6889, 317.2035, 258.8133, 143.3831, 174.2746, 131.1390, 188.0510, 127.9795, 130.2359, 177.4510, 39.6949, 29.9510, 58.1107, 81.1559, 80.8303, 4.3077, 1.6350, 1.6470, 2.0269, 3.0849, 3.2866, 2.8201, 2.7524, 2.1050, 1.7440, 2.2217, 2.0554, 1.8307, 2.1945, 11.9147, 6.3669, 18.4437, 9.3416, -1.0258, 14.5948, 32.8415, 72.9304, 2.2325, 39.9641, 8.2164, 45.6421, 87.7984, 0.0004, -0.8599, 58.9369, 4.4058, -3.6313, -1.7850, -1.5497, -0.2835, -3.2622, -0.3920, 1.0528, -4.6143, -2.1271, -0.4016, -1.2728, 0.2871, -0.0101, 1.2197, 0.4426, 1.5783, -1.1779, -1.2434, -1.0022, -0.7073, -0.1780, 0.2476, -0.0555, -0.2328, 0.8372, -0.1321, -0.4271, -0.3720, -0.2669, 0.0011, -1.4575, -1.2375, -0.7364, -0.3026])

# Misfit for the below params: 2.33996 (1.757 w/out Gasparik1989 CMAS @ HP)
set_params([-267.9933, -2175.6482, -1477.0366, -2147.9553, -2137.3511, -1472.2071, -5257.2993, -6644.5061, -5772.3980, -6034.0139, -906.9163, -867.9480, -2841.8564, -3098.7711, -3300.7216, -3093.1707, -2390.4501, -3197.8647, -3085.8708, -2392.8390, -1451.1947, -1109.0806, -2298.0234, 27.0505, 55.5654, 94.1321, 151.3814, 85.2700, 82.1121, 136.1783, 339.0245, 247.6889, 317.2035, 258.8133, 143.3831, 174.2746, 131.1390, 188.0510, 127.9795, 130.2359, 177.4510, 39.6949, 29.9510, 58.1107, 81.1559, 80.8303, 4.3077, 1.6350, 1.6470, 2.0269, 3.0849, 3.2866, 2.8201, 2.7524, 2.1050, 1.7440, 2.2217, 2.0554, 1.8307, 2.1945, 11.9147, 6.3669, 18.4437, 9.3416, -1.0258, 14.5948, 32.8415, 72.9304, 2.2325, 39.9641, 8.2164, 45.6421,
            53.2, 0.0004, -0.8599, 44.4,
            4.4058, 0.,
            -3.6313, -1.7850, -1.5497, -0.2835, -3.2622, -0.3920, 1.0528, -4.6143, -2.1271, -0.4016, -1.2728, 0.2871, -0.0101, 1.2197, 0.4426, 1.5783, -1.1779, -1.2434, -1.0022, -0.7073, -0.1780, 0.2476, -0.0555, -0.2328, 0.8372, -0.1321, -0.4271, -0.3720, -0.2669, 0.0011, -1.4575, -1.2375, -0.7364, -0.3026])

set_params([-267.9998, -2175.6477, -1477.0346, -2147.9578, -2137.3548, -1472.2068, -5257.3006, -6644.3968, -5772.3970, -6034.0019, -906.9171, -867.9487, -2841.8691, -3098.6794, -3300.8429, -3093.1754, -2390.4421, -3197.8165, -3085.8109, -2392.8297, -1451.1933, -1109.0845, -2298.0039, 27.0447, 55.5655, 94.1388, 151.3901, 85.2785, 82.1151, 136.1812, 339.0253, 247.4751, 317.1978, 258.8043, 143.6547, 174.2922, 131.0614, 188.0353, 127.8968, 130.1543, 177.4395, 39.6943, 29.9487, 58.1095, 81.1618, 80.7955, 4.3064, 1.6279, 1.6482, 2.0264, 3.0859, 3.2842, 2.8216, 2.7344, 2.1071, 1.7468, 2.2228, 2.0457, 1.8328, 2.1860, 11.9242, 6.3636, 18.4421, 9.3386, -1.0223, 14.6373, 32.7734, 72.9169, 2.2278, 40.1914, 8.2346, 45.5438,
            56.0, -0.1241, -0.8625,
            50.5, 4.4057, 0.1241, -3.6314, -1.7835, -1.5493, -0.2835, -3.2612, -0.3920, 1.0519, -4.6119, -2.1243, -0.4016, -1.2731, 0.2871, -0.0101, 1.2197, 0.4408, 1.5757, -1.1756, -1.2432, -1.0029, -0.7096, -0.1783, 0.2458, -0.0583, -0.2328, 0.8350, -0.1320, -0.4268, -0.3716, -0.2667, 0.0005, -1.4566, -1.2287, -0.7349, -0.3026])

set_params([-268.4739, -2173.8834, -1477.0293, -2146.1451, -2135.5076, -1472.4608, -5260.1885, -6649.9451, -5776.6848, -6025.1763, -906.9269, -867.4287, -2843.1151, -3093.5685, -3290.2902, -3090.1364, -2390.5562, -3202.5809, -3082.8797, -2392.8622, -1450.4009, -1109.0735, -2300.5607, 26.9162, 55.6422, 94.1601, 151.3910, 85.4197, 82.3173, 136.0242, 338.7844, 247.4604, 316.4747, 262.0356, 143.6533, 174.2588, 132.0470, 187.9132, 125.8847, 131.1344, 177.3743, 39.6852, 30.0223, 58.0722, 81.3309, 79.8615, 4.3057, 1.6429, 1.6467, 2.0241, 3.0960, 3.2868, 2.8064, 2.7462, 2.0971, 1.7850, 2.2172, 2.0356, 1.8274, 2.2105, 11.9125, 6.3713, 18.2698, 9.3268, -0.9872, 14.5214, 32.0021, 72.1470, 2.2890, 37.6817, -7.3840, 14.3411, 51.8055, -8.6978, -2.4498, 48.4941, 3.6436, 41.0421, -10.3480, -5.2384, -1.3077, -0.1746, -3.0950, -0.2509, 0.8774, -4.1835, -1.6226, -0.3873, -1.3040, 0.2346, -0.0036, 1.1554, 0.4392, 1.5001, -1.2584, -1.3233, -1.0009, -0.7863, -0.2533, 0.1669, -0.1384, -0.2339, 0.7568, -0.1396, -0.4071, -0.2961, -0.2062, -0.0568, -1.2365, -1.2827, -0.4706, -0.3405])

set_params([-268.6060, -2173.5492, -1477.0201, -2145.7415, -2135.0333, -1472.6349, -5259.7228, -6651.9929, -5780.1667, -6019.6750, -906.9379, -867.0594, -2843.2872, -3092.8758, -3286.7163, -3090.0164, -2390.4286, -3190.6156, -3082.7364, -2392.6614, -1450.2577, -1108.6901, -2299.1377, 26.8932, 55.6628, 94.1608, 151.3982, 85.4486, 82.4300, 135.8881, 339.4845, 246.5373, 316.4337, 264.3511, 143.7099, 174.2987, 131.9248, 188.0285, 128.4783, 131.0218, 177.5432, 39.6772, 30.1824, 58.0869, 81.5214, 80.2874, 4.3076, 1.6491, 1.6463, 2.0241, 3.0980, 3.2889, 2.8069, 2.7494, 2.0911, 1.7669, 2.2163, 2.0276, 1.8274, 2.2104, 11.8693, 6.2602, 18.0136, 9.2142, -1.1185, 3.4911, 32.0530, 60.9578, 2.2942, 37.4378, -11.7381, 7.0377, 42.5861, -6.6332, -10.7712, 52.6959, 3.3976, 41.3092, -24.8677, 0.9900, -0.9105, -0.1396, -3.1691, -0.3362, 0.7695, -4.1474, -1.2179, -0.3549, -1.1982, 0.2290, -0.0028, 1.3234, 0.3745, 1.4667, -1.2948, -1.3691, -1.1263, -0.8252, -0.2949, 0.1296, -0.1734, -0.2209, 0.7218, -0.0249, -0.2407, -0.1820, -0.1336, -0.0243, -1.1291, -1.2798, -0.2424, -0.2123])

print(gt.energy_interaction)
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



pressures, x_ols, RTlnKDs = np.array(zip(*P_Xol_RTlnKDs))
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



pressures, x_wads, RTlnKDs = np.array(zip(*P_Xwad_RTlnKDs))
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

            
pressures, x_rws, x_fpers = np.array(zip(*P_Xrw_Xfper))
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
        for i, x_m1 in enumerate(x_m1s):
            composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
            assemblage = burnman.Composite([m1, m2])
            assemblage.set_state(P1, T0)
            m1.guess = np.array([1. - x_m1, x_m1])
            m2.guess = np.array([1. - x_m1, x_m1])
            equality_constraints = [('T', T0), ('phase_proportion', (m2, 0.0))]
            sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           initial_state_from_assemblage=True,
                                           store_iterates=False)
            x_m2s[i] = m2.molar_fractions[1]
            pressures[i] = assemblage.pressure

        plt.plot(x_m1s, pressures/1.e9, linewidth=3., color=color)
        plt.plot(x_m2s, pressures/1.e9, linewidth=3., color=color)
    plt.plot([x_ol_inv, x_rw_inv], [P_inv/1.e9, P_inv/1.e9], linewidth=3., color=color)


    # bdg + fper 
    x_m1s = []
    pressures = []
    x_m2s = []
    for x_m1 in np.linspace(0.3, 0.5, 51):
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
        child_solutions['ring'].guess = np.array([1. - x_m1, x_m1])
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
    plt.plot(x_m2s, np.array(pressures)/1.e9, linewidth=3., color=color, label='{0} K'.format(T0))
    
    
    # bdg + fper 
    x_m1s = []
    pressures = []
    x_m2s = []
    for x_m1 in np.linspace(0.01, 0.3, 21):
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
        child_solutions['ring'].guess = np.array([1. - x_m1, x_m1])
        child_solutions['mg_fe_bdg'].guess = np.array([1. - x_m1, x_m1])
        fper.guess = np.array([1. - x_m1, x_m1])
        assemblage = burnman.Composite([child_solutions['ring'],
                                        child_solutions['mg_fe_bdg'], fper])
        assemblage.set_state(25.e9, T0)
        equality_constraints = [('T', T0),
                                ('phase_proportion', (child_solutions['ring'], 1.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_state_from_assemblage=True,
                                       store_iterates=False)
        if sol.success:
            print(assemblage.pressure/1.e9)
            x_m1s.append(x_m1)
            x_m2s.append((fper.molar_fractions[1] +
                          child_solutions['mg_fe_bdg'].molar_fractions[1])/2.)
            pressures.append(assemblage.pressure)
    
    plt.plot(x_m1s, np.array(pressures)/1.e9, linewidth=3., color=color)
    plt.plot(x_m2s, np.array(pressures)/1.e9, linewidth=3., color=color)
    
    # rw -> fper + stv
    x_m1s = np.linspace(0.2, 0.999, 21)
    pressures = np.empty_like(x_m1s)
    x_m2s = np.empty_like(x_m1s)
    for i, x_m1 in enumerate(x_m1s):
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([child_solutions['ring'], fper, stv])
        assemblage.set_state(25.e9, T0)
        child_solutions['ring'].guess = np.array([1. - x_m1, x_m1])
        fper.guess = np.array([1. - x_m1, x_m1])
        equality_constraints = [('T', T0), ('phase_proportion', (child_solutions['ring'], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                       initial_state_from_assemblage=True,
                                       store_iterates=False)
        x_m2s[i] = child_solutions['ring'].molar_fractions[1]
        pressures[i] = assemblage.pressure

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
    pressures, pressure_shift, xs = np.array(zip(*P_Xmg_phase[m]))
    for i in range(len(xs)):
        plt.arrow(xs[i], pressures[i]/1.e9, 0., pressure_shift[i]/1.e9, **arrow_params)
    plt.scatter(xs, pressures/1.e9, s=80., label='data')

plt.xlabel('p(Fe$_2$SiO$_4$)')
plt.ylabel('Pressure (GPa)')
plt.legend()
plt.show()
