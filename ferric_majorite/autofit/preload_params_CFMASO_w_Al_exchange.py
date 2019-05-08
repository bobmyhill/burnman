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
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
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
                 ['gt', 'E', 2, 0,  gt.energy_interaction[0][3], 1.e3], # py-dmaj
                 ['gt', 'E', 1, 0,  gt.energy_interaction[1][0], 1.e3], # alm-gr
                 ['gt', 'E', 1, 1,  gt.energy_interaction[1][1], 1.e3], # alm-andr
                 ['gt', 'E', 2, 0,  gt.energy_interaction[2][0], 1.e3], # gr-andr
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

#######################
### PUT PARAMS HERE ###
#######################

set_params([-266.7804, -2175.8766, -1476.9576, -2148.7595, -2138.1450, -1472.1244, -5253.0046, -6641.5494, -5769.4822, -6038.5050, -906.9301, -871.6192, -2842.8178, -3097.3005, -3305.0463, -3093.4992, -2390.6796, -3196.1322, -3086.1310, -2393.0915, -1450.7721, -1106.3165, -2300.7583, 26.9869, 56.2912, 93.9918, 151.4445, 84.8269, 81.6946, 136.1583, 342.1085, 251.6326, 316.9522, 254.5720, 143.8573, 174.2266, 130.6519, 187.6635, 130.1545, 129.7794, 177.0437, 39.6837, 27.5593, 58.1144, 83.1318, 80.3368, 4.3100, 1.6345, 1.6390, 2.0266, 3.0693, 3.2748, 2.8200, 2.7503, 2.1251, 1.7549, 2.2453, 2.0366, 1.8537, 2.3216, 11.8694, 6.0950, 18.0447, 9.3693, -1.4707, 15.6215, 31.9467, 74.0480, 2.4138, 38.0345, 12.4288, 45.1317, 89.7620, 0.0002, -0.0268, 59.8595, 4.9230, -0.3956, -0.2324, -1.2723, -0.2376, -3.0395, -0.2977, 1.1121, -4.3050, -1.8686, -0.3433, -1.1053, 0.1909, -0.0138, 1.2041, 0.4605, 1.6597, -1.0868, -1.1535, -0.8684, -0.6280, 0.0451, 0.3404, 0.0306, -0.1910, 0.9185, -0.0627, -0.2040, -0.1698, -0.1201, 0.0200, -0.6842, -1.0907, -0.4407, -0.1269])
