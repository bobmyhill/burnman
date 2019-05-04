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
from fitting_functions import equilibrium_order

from input_dataset import *
from preload_params import *

print('WARNING!!!!!! oen-ofs needs order-disorder!!!')
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
    old_params = get_params()
    
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


        # Equilibrate phases with order-disorder *iff*
        # there has been a significant change in any params
        if any(np.array(old_params) - np.array(params) > 1.e-2):
            for j, phase in enumerate(assemblage.phases):
                if isinstance(phase, burnman.SolidSolution):
                    if 'order-disorder' in phase.name:
                        equilibrium_order(phase)
                        assemblage.stored_compositions[j] = (assemblage.phases[j].molar_fractions,
                                                             assemblage.phases[j].molar_fraction_covariances)
        
        
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

endmember_args = [['sp',   'H_0', sp.params['H_0'], 1.e3], # spinels
                  ['herc', 'H_0', herc.params['H_0'], 1.e3],
                  ['mt',   'H_0', mt.params['H_0'], 1.e3],
                  ['gr',   'H_0', gr.params['H_0'], 1.e3],
                  ['andr', 'H_0', andr.params['H_0'], 1.e3],
                  
                  ['dmaj', 'H_0', dmaj.params['H_0'], 1.e3],
                  ['nagt', 'H_0', nagt.params['H_0'], 1.e3],
                  ['mgts', 'H_0', mgts.params['H_0'], 1.e3],
                  ['hed',  'H_0', hed.params['H_0'], 1.e3], # cpx: no di (Ca std) or jd (Na std)
                  ['cen',  'H_0', cen.params['H_0'], 1.e3],
                  
                  ['cfs',  'H_0', cfs.params['H_0'], 1.e3], 
                  ['cats', 'H_0', cats.params['H_0'], 1.e3],
                  ['aeg', 'H_0',  aeg.params['H_0'], 1.e3],
                  ['mbdg', 'H_0', mbdg.params['H_0'], 1.e3], # bridgmanite
                  ['fbdg', 'H_0', fbdg.params['H_0'], 1.e3],

                  ['gr', 'S_0',   gr.params['S_0'], 1.],
                  ['andr', 'S_0', andr.params['S_0'], 1.],
                  ['dmaj', 'S_0', dmaj.params['S_0'], 1.],
                  ['mgts', 'S_0', mgts.params['S_0'], 1.],
                  ['mbdg', 'S_0', mbdg.params['S_0'], 1.],
                  ['fbdg', 'S_0', fbdg.params['S_0'], 1.]]

solution_args = [['sp', 'E', 1, 0,  spinel.energy_interaction[1][0], 1.e3], # herc-mt
                 ['sp', 'E', 1, 2,  spinel.energy_interaction[1][2], 1.e3], # herc-frw
                 ['sp', 'E', 2, 1,  spinel.energy_interaction[2][1], 1.e3], # mt-frw
                 
                 ['opx', 'E', 0, 0, opx_od.energy_interaction[0][0], 1.e3], # oen-ofs
                 ['opx', 'E', 0, 1, opx_od.energy_interaction[0][1], 1.e3], # oen-mgts
                 ['opx', 'E', 0, 2, opx_od.energy_interaction[0][2], 1.e3], # oen-odi
                 ['opx', 'E', 1, 0, opx_od.energy_interaction[1][0], 1.e3], # ofs-mgts
                 ['opx', 'E', 1, 1, opx_od.energy_interaction[1][1], 1.e3], # ofs-odi
                 ['opx', 'E', 2, 0, opx_od.energy_interaction[2][0], 1.e3], # mgts-odi
                 ['opx', 'E', 2, 1, opx_od.energy_interaction[2][1], 1.e3], # mgts-ofm
                 ['opx', 'E', 3, 0, opx_od.energy_interaction[3][0], 1.e3], # odi-ofm
                 
                 ['cpx', 'E', 0, 0, cpx_od.energy_interaction[0][0], 1.e3], # di-hed
                 ['cpx', 'E', 0, 1, cpx_od.energy_interaction[0][1], 1.e3], # di-cen
                 ['cpx', 'E', 0, 2, cpx_od.energy_interaction[0][1], 1.e3], # di-cfs
                 ['cpx', 'E', 0, 3, cpx_od.energy_interaction[0][3], 1.e3], # di-cats
                 ['cpx', 'E', 2, 0, cpx_od.energy_interaction[2][0], 1.e3], # cen-cfs
                 ['cpx', 'E', 2, 1, cpx_od.energy_interaction[2][1], 1.e3], # cen-cats
                 ['cpx', 'E', 2, 2, cpx_od.energy_interaction[2][2], 1.e3], # cen-jd
                 ['cpx', 'E', 2, 3, cpx_od.energy_interaction[2][3], 1.e3], # cen-aeg
                 ['cpx', 'E', 3, 0, cpx_od.energy_interaction[3][0], 1.e3], # cfs-cats
                 ['cpx', 'E', 3, 1, cpx_od.energy_interaction[3][1], 1.e3], # cfs-jd
                 ['cpx', 'E', 3, 2, cpx_od.energy_interaction[3][2], 1.e3], # cfs-aeg
                 ['cpx', 'E', 4, 0, cpx_od.energy_interaction[4][0], 1.e3], # cats-jd
                 ['cpx', 'E', 4, 1, cpx_od.energy_interaction[4][1], 1.e3], # cats-aeg
                 
                 ['bdg', 'E', 0, 0, bdg.energy_interaction[0][0], 1.e3], # mbdg-fbdg
                 
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
                 ['gt', 'E', 4, 0,  gt.energy_interaction[4][0], 1.e3]] # dmaj-nagt


solution_priors = [['opx', 'E', 0, 1, opx_od.energy_interaction[0][1], 5.e3], # oen-mgts
                   ['opx', 'E', 0, 2, opx_od.energy_interaction[0][2], 5.e3], # oen-odi
                   ['opx', 'E', 1, 0, opx_od.energy_interaction[1][0], 5.e3], # ofs-mgts
                   #['opx', 'E', 1, 1, opx_od.energy_interaction[1][1], 10.e3], # ofs-odi
                   #['opx', 'E', 2, 0, opx_od.energy_interaction[2][0], 30.e3], # mgts-odi
                   ['opx', 'E', 2, 1, opx_od.energy_interaction[2][1], 5.e3], # mgts-ofm
                   #['opx', 'E', 3, 0, opx_od.energy_interaction[3][0], 10.e3], # odi-ofm
                
                   #['cpx', 'E', 0, 1, 30.e3, 5.e3], # di-cen
                   #['cpx', 'E', 0, 2, 26.e3, 5.e3], # di-cfs
                   #['cpx', 'E', 0, 3, 13.e3, 5.e3], # di-cats
                   #['cpx', 'E', 2, 0, 26.e3, 5.e3], # cen-cfs
                   #['cpx', 'E', 2, 1, 21.e3, 5.e3], # cen-cats
                   #['cpx', 'E', 2, 2,  9.e3, 5.e3], # cen-jd
                   #['cpx', 'E', 2, 3, 10.e3, 5.e3], # cen-aeg
                   #['cpx', 'E', 3, 0, 25.e3, 5.e3], # cfs-cats
                   #['cpx', 'E', 3, 1, 24.e3, 5.e3], # cfs-jd
                   #['cpx', 'E', 3, 2, 52.e3, 10.e3], # cfs-aeg
                   #['cpx', 'E', 4, 0, 6.e3,  5.e3], # cats-jd
                   #['cpx', 'E', 4, 1, 17.e3, 5.e3], # cats-aeg
                   
                   ['gt', 'E', 1, 0,  gt.energy_interaction[1][0], 10.e3], # alm-gr
                   ['gt', 'E', 2, 0,  2.e3, 1.e3]] # gr-andr

endmember_priors = [['gr', 'S_0', gr.params['S_0_orig'][0], gr.params['S_0_orig'][1]],
                    ['andr', 'S_0', andr.params['S_0_orig'][0], gr.params['S_0_orig'][1]],
                    ['mbdg', 'S_0', mbdg.params['S_0_orig'][0], mbdg.params['S_0_orig'][1]]]


experiment_uncertainties = [
                            ['Frost_2003_H1554', 'P', 0., 0.5e9],
                            ['Frost_2003_H1555', 'P', 0., 0.5e9],
                            ['Frost_2003_H1556', 'P', 0., 0.5e9],
                            ['Frost_2003_H1582', 'P', 0., 0.5e9],
                            ['Frost_2003_S2773', 'P', 0., 0.5e9],
                            ['Frost_2003_V170', 'P', 0., 0.5e9],
                            ['Frost_2003_V171', 'P', 0., 0.5e9],
                            ['Frost_2003_V175', 'P', 0., 0.5e9],
                            ['Frost_2003_V179', 'P', 0., 0.5e9],
    
                            ['Beyer2019_H4321', 'P', 0., 0.5e9],
                            ['Beyer2019_H4556', 'P', 0., 0.5e9],
                            ['Beyer2019_H4557', 'P', 0., 0.5e9],
                            ['Beyer2019_H4560', 'P', 0., 0.5e9],
                            ['Beyer2019_H4692', 'P', 0., 0.5e9],
                            ['Beyer2019_Z1699', 'P', 0., 0.5e9],
                            ['Beyer2019_Z1700', 'P', 0., 0.5e9],
                            ['Beyer2019_Z1782', 'P', 0., 0.5e9],
                            ['Beyer2019_Z1785', 'P', 0., 0.5e9],
                            ['Beyer2019_Z1786', 'P', 0., 0.5e9]]
"""
    ['49Fe', 'P', 0., 0.5e9], # Frost, 2003
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
#,
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
                                                           ss.basis,
                                                           solution_name=ss.name).__dict__)

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

# Endmembers
from endmember_reactions import endmember_reaction_assemblages

# FSO buffers
from ONeill_1987_QFI import ONeill_1987_QFI_assemblages
from ONeill_1987_QFM import ONeill_1987_QFM_assemblages

# Fe-Mg-Si-O
from Frost_2003_fper_ol_wad_rw import Frost_2003_assemblages
from Seckendorff_ONeill_1992_ol_opx import Seckendorff_ONeill_1992_assemblages
from ONeill_Wood_1979_ol_gt import ONeill_Wood_1979_assemblages
from Matsuzaka_et_al_2000_rw_wus_stv import Matsuzaka_2000_assemblages
from Nakajima_FR_2012_bdg_fper import Nakajima_FR_2012_assemblages
from Tange_TNFS_2009_bdg_fper_stv import Tange_TNFS_2009_FMS_assemblages

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

# FMASO
from Frost_2003_FMASO_garnet import Frost_2003_FMASO_gt_assemblages

# NCFMASO
from Rohrbach_et_al_2007_NCFMASO_gt_cpx import Rohrbach_et_al_2007_NCFMASO_assemblages
from Beyer_et_al_2019_NCFMASO import Beyer_et_al_2019_NCFMASO_assemblages

# from Frost_2003_CFMASO_garnet import Frost_2003_CFMASO_gt_assemblages # lousy Fe3+ estimates


assemblages = [assemblage for assemblage_list in
               [endmember_reaction_assemblages,
                ONeill_1987_QFI_assemblages,
                ONeill_1987_QFM_assemblages, # Frost_2003_assemblages,
                Seckendorff_ONeill_1992_assemblages,
                Matsuzaka_2000_assemblages,
                ONeill_Wood_1979_assemblages,
                Nakajima_FR_2012_assemblages,
                Tange_TNFS_2009_FMS_assemblages,
                Gasparik_1992_MAS_assemblages,
                Gasparik_Newton_1984_MAS_assemblages,
                Gasparik_Newton_1984_MAS_univariant_assemblages,
                Perkins_et_al_1981_MAS_assemblages,
                Carlson_Lindsley_1988_CMS_assemblages,
                Perkins_Newton_1980_CMAS_assemblages,
                Klemme_ONeill_2000_CMAS_assemblages,
                Perkins_Vielzeuf_1992_CFMS_assemblages,
                Woodland_ONeill_1993_FASO_assemblages, 
                Frost_2003_FMASO_gt_assemblages, 
                Rohrbach_et_al_2007_NCFMASO_assemblages,
                Beyer_et_al_2019_NCFMASO_assemblages
               ]
               for assemblage in assemblage_list]


#for a in assemblages:
#    print(a.experiment_id)

#minimize_func(get_params(), assemblages)


#######################
### PUT PARAMS HERE ###
#######################

#set_params([-2300.5260, -1952.8132, -1114.1794, -6642.6518, -5769.3578, -6041.5016, -5985.0000, -3196.5258, -2842.0531, -3093.6480, -2386.6257, -3311.0976, -2583.4365, -1442.3794, -1082.8444, 254.4851, 316.7080, 260.1038, 130.8081, 62.7609, 0.1092, 0.0821, 0.3506, -1.2956, 12.8011, 32.2286, 10.8037, 25.5400, 75.4866, 15.0138, 22.5400, 2.9120, 29.0695, 29.8017, 12.7598, 2.2951, 44.6535, 40.0054, 60.7974, 25.0023, 23.9968, 52.2989, 5.9952, 17.4016, 6.0534, 0.0041, -0.0005, 0.2578, 59.7427, 0.0017, 0.0005, 5.1277, -0.0005, -0.0002, 0.0004, 0.0002, -0.0000, -0.0003, -0.0012, -0.0010, -0.0007, 0.0001, -0.0037, -0.0454, -0.0028, -0.0007, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

set_params([-2301.5435, -1949.4815, -1114.3992, -6639.7209, -5770.0136,
            -6034.1251, -5984.0188, -3196.6548, -2839.8132, -3092.7899,
            -2386.8588, -3313.3077, -2576.6595, -1442.7414, -1082.3128,
            248.7322, 317.8158, 247.8247, 130.9479, 63.5795, fbdg.params['S_0'],
            1.6918, 1.3527, 5.0549, -1.2335, 14.0394, 29.6786, 8.6808, 24.1291, 75.5459, 15.1094, 22.1892, 4.4837, 27.9658, 29.7861, 11.3410, 2.4162, 44.5473, 39.6185, 60.1186, 25.1158, 23.8377, 52.2393, 1.3346, 14.0828, 6.4905, 4.4018, -1.0113, 1.5625, 57.5032, -1.8608, 0.8468, 6.9018, -3.7820, 2.0548, 6.0746, -1.9318, 0.2997, -0.0046, -0.0172, -0.0132, -0.0097, 0.0008, -0.0541, -0.5420, -0.0405, -0.0097, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

"""
['Beyer2019_H4321', 'P', 0.0002, 0.5e9],   # (Re)
['Beyer2019_H4556', 'P', 0.4140, 0.5e9],   # Fe gt ol
['Beyer2019_H4557', 'P', -10.8301, 0.5e9], # Fe gt cpx 
['Beyer2019_H4560', 'P', -0.013, 0.5e9],   # (Re)
['Beyer2019_H4692', 'P', -6.6531, 0.5e9],  # Mo opx ol cpx gt
['Beyer2019_Z1699', 'P', -0.0008, 0.5e9],  # (Re)
['Beyer2019_Z1700', 'P', -12.82, 0.5e9],   # Fe cpx gt
['Beyer2019_Z1782', 'P', -10.2654, 0.5e9], # Mo gt wad hpx 
['Beyer2019_Z1785', 'P', -3.3551, 0.5e9],  # Mo gt ring?
['Beyer2019_Z1786', 'P', -10.8939, 0.5e9]] # Fe mw ring? hpx fcc_iron gt
"""

########################
# RUN THE MINIMIZATION #
########################
if run_inversion:
    print('Running inversion with {0} assemblages for {1} parameters'.format(len(assemblages),
                                                                             len(get_params())))
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

# Mg2SiO4 phase diagram





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
    child_solutions['ring'].guess = np.array([0.15, 0.85])
    wad.guess = np.array([1. - x_m1, x_m1])
    ol.guess = np.array([1. - x_m1, x_m1])
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
        plt.plot(x_m2s, pressures/1.e9, linewidth=3., color=color, label='{0} K'.format(T0))
    plt.plot([x_ol_inv, x_rw_inv], [P_inv/1.e9, P_inv/1.e9], linewidth=3., color=color)
    
    # Now do the same for rw -> fper + stv
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

"""
for m in ['ol', 'wad', 'ring']:
    pressures, pressure_shift, xs = np.array(zip(*P_Xmg_phase[m]))
    for i in range(len(xs)):
        plt.arrow(xs[i], pressures[i]/1.e9, 0., pressure_shift[i]/1.e9, **arrow_params)
    plt.scatter(xs, pressures/1.e9, s=80., label='data')
"""    
plt.legend()
plt.show()
