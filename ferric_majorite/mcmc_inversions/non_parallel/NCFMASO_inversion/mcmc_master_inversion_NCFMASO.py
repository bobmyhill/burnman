from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit
from burnman.equilibrate import equilibrate
from burnman.solutionbases import transform_solution_to_new_basis

from input_dataset import endmembers, solutions, child_solutions
import pickle

if len(sys.argv) == 2:
    if sys.argv[1] == '--fit':
        run_inversion = True
        print('Running inversion')
    else:
        run_inversion = False
        print('Not running inversion. Use --fit as command line'
              ' argument to invert parameters')
else:
    run_inversion = False
    print('Not running inversion. Use --fit as command line argument'
          ' to invert parameters')


def set_params_from_special_constraints():
    # 1) Destabilise fwd
    endmembers['fa'].set_state(6.25e9, 1673.15)
    endmembers['frw'].set_state(6.25e9, 1673.15)
    endmembers['fwd'].set_state(6.25e9, 1673.15)

    # First, determine the entropy which will give the fa-fwd reaction
    # the same slope as the fa-frw reaction
    dPdT = (endmembers['frw'].S
            - endmembers['fa'].S)/(endmembers['frw'].V
                                   - endmembers['fa'].V)  # = dS/dV

    dV = endmembers['fwd'].V - endmembers['fa'].V
    dS = dPdT*dV
    endmembers['fwd'].params['S_0'] += (endmembers['fa'].S
                                        - endmembers['fwd'].S + dS)
    endmembers['fwd'].params['H_0'] += (endmembers['frw'].gibbs
                                        - endmembers['fwd'].gibbs
                                        + 100.)  # fwd less stable than frw

    # 2) Copy interaction parameters from opx to hpx:
    solutions['hpx'].alphas = solutions['opx'].alphas
    solutions['hpx'].energy_interaction = solutions['opx'].energy_interaction
    solutions['hpx'].entropy_interaction = solutions['opx'].entropy_interaction
    solutions['hpx'].volume_interaction = solutions['opx'].volume_interaction


def minimize_func(params, assemblages):
    # Set parameters
    set_params(params)

    chisqr = []
    # Run through all assemblages for affinity misfit
    # This is what takes most of the time
    for i, assemblage in enumerate(assemblages):
        # print(i, assemblage.experiment_id,
        # [phase.name for phase in assemblage.phases])
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
        # print('assemblage chisqr', assemblage.experiment_id,
        # [phase.name for phase in assemblage.phases], assemblage.chisqr)
        chisqr.append(assemblage.chisqr)

    # Endmember priors
    for p in endmember_priors:
        c = np.power(((dict_endmember_args[p[0]][p[1]] - p[2])/p[3]), 2.)
        # print('endmember_prior', p[0], p[1], dict_endmember_args[p[0]][p[1]],
        # p[2], p[3], c)
        chisqr.append(c)

    # Solution priors
    for p in solution_priors:
        c = np.power(((dict_solution_args[p[0]]['{0}{1}{2}'.format(p[1],
                                                                   p[2],
                                                                   p[3])]
                       - p[4])/p[5]), 2.)
        # print('solution_prior', c)
        chisqr.append(c)

    # Experiment uncertainties
    for u in experiment_uncertainties:
        c = np.power(u[2]/u[3], 2.)
        # print('pressure uncertainty', c)
        chisqr.append(c)

    # calculate the squared misfit.
    # this is an approximation to the negative log probability
    # see http://www.physics.utah.edu/~detar/phys6720/handouts/
    # curve_fit/curve_fit/node2.html
    half_sqr_misfit = np.sum(chisqr)/2.
    # print(rms_misfit)
    string = '['
    for i, p in enumerate(params):
        string += '{0:.4f}'.format(p)
        if i < len(params) - 1:
            string += ', '
    string += ']'
    # print(string)

    if np.isnan(half_sqr_misfit) or not np.isfinite(half_sqr_misfit):
        return np.inf  # catch for if one or more EoSes fail
    else:
        return half_sqr_misfit


def log_probability(params, assemblages):
    return -minimize_func(params, assemblages)

# Component defining endmembers (for H_0 and S_0) are:
# Fe: Fe metal (BCC, FCC, HCP)
# O: O2
# Mg: MgO per
# Si: SiO2 qtz
# Al: Mg3Al2Si3O12 pyrope
# Ca: CaMgSi2O6 diopside
# Na: NaAlSi2O6 jadeite

endmember_args = []
endmember_args.extend([[mbr, 'H_0', endmembers[mbr].params['H_0'], 1.e3]
                       for mbr in ['wus',
                                   'fo', 'fa',
                                   'mwd',
                                   'mrw', 'frw',
                                   'alm', 'gr', 'andr', 'dmaj', 'nagt',
                                   'coe', 'stv',
                                   'hed', 'cen', 'cfs', 'cats', 'aeg',
                                   'oen', 'ofs', 'mgts',
                                   'hen', 'hfs',
                                   'mbdg', 'fbdg',
                                   'sp']])

endmember_args.extend([[mbr, 'S_0', endmembers[mbr].params['S_0'], 1.]
                       for mbr in ['per', 'wus',
                                   'fo', 'fa',
                                   'mwd',
                                   'mrw', 'frw',
                                   'alm', 'gr', 'andr', 'dmaj', 'nagt',
                                   'coe', 'stv',
                                   'di', 'hed',  # 'cen', 'cfs', 'cats', 'aeg',
                                   'oen', 'ofs', 'mgts',
                                   'hen', 'hfs',
                                   'mbdg', 'fbdg',
                                   'sp']])

endmember_args.extend([[mbr, 'V_0', endmembers[mbr].params['V_0'], 1.e-5]
                       for mbr in ['fwd']])
endmember_args.extend([[mbr, 'K_0', endmembers[mbr].params['K_0'], 1.e11]
                       for mbr in ['wus', 'fwd', 'frw']])
endmember_args.extend([[mbr, 'a_0', endmembers[mbr].params['a_0'], 1.e-5]
                       for mbr in ['per', 'wus',
                                   'fo', 'fa',
                                   'mwd', 'fwd',
                                   'mrw', 'frw',
                                   'mbdg', 'fbdg']])


solution_args = [['mw', 'E', 0, 0,
                  solutions['mw'].energy_interaction[0][0], 1.e3],
                 ['ol', 'E', 0, 0,
                  solutions['ol'].energy_interaction[0][0], 1.e3],
                 ['wad', 'E', 0, 0,
                  solutions['wad'].energy_interaction[0][0], 1.e3],
                 ['sp', 'E', 3, 0,
                  solutions['sp'].energy_interaction[3][0], 1.e3],

                 ['opx', 'E', 0, 0,
                  solutions['opx'].energy_interaction[0][0], 1.e3],  # oen-ofs
                 ['opx', 'E', 0, 1,
                  solutions['opx'].energy_interaction[0][1], 1.e3],  # oen-mgts
                 ['opx', 'E', 0, 2,
                  solutions['opx'].energy_interaction[0][2], 1.e3],  # oen-odi
                 ['opx', 'E', 2, 0,
                  solutions['opx'].energy_interaction[2][0], 1.e3],  # mgts-odi

                 ['cpx', 'E', 0, 0,
                  solutions['cpx'].energy_interaction[0][0], 1.e3],  # di-hed
                 ['cpx', 'E', 0, 1,
                  solutions['cpx'].energy_interaction[0][1], 1.e3],  # di-cen
                 ['cpx', 'E', 0, 3,
                  solutions['cpx'].energy_interaction[0][3], 1.e3],  # di-cats
                 ['cpx', 'E', 2, 1,
                  solutions['cpx'].energy_interaction[2][1], 1.e3],  # cen-cats
                 ['gt', 'E', 0, 2,
                  solutions['gt'].energy_interaction[0][2], 1.e3],  # py-andr
                 ['gt', 'E', 0, 3,
                  solutions['gt'].energy_interaction[0][3], 1.e3],  # py-dmaj
                 ['gt', 'E', 0, 4,
                  solutions['gt'].energy_interaction[0][4], 1.e3],  # py-nagt
                 ['gt', 'E', 1, 0,
                  solutions['gt'].energy_interaction[1][0], 1.e3],  # alm-gr
                 ['gt', 'E', 1, 1,
                  solutions['gt'].energy_interaction[1][1], 1.e3],  # alm-andr
                 ['gt', 'E', 1, 2,
                  solutions['gt'].energy_interaction[1][2], 1.e3],  # alm-dmaj
                 ['gt', 'E', 1, 3,
                  solutions['gt'].energy_interaction[1][3], 1.e3],  # alm-nagt
                 ['gt', 'E', 2, 0,
                  solutions['gt'].energy_interaction[2][0], 1.e3],  # gr-andr
                 ['gt', 'E', 2, 1,
                  solutions['gt'].energy_interaction[2][1], 1.e3],  # gr-dmaj
                 ['gt', 'E', 2, 2,
                  solutions['gt'].energy_interaction[2][2], 1.e3],  # gr-nagt
                 ['gt', 'E', 3, 0,
                  solutions['gt'].energy_interaction[3][0], 1.e3],  # andr-dmaj
                 ['gt', 'E', 3, 1,
                  solutions['gt'].energy_interaction[3][1], 1.e3],  # andr-nagt
                 ['gt', 'E', 4, 0,
                  solutions['gt'].energy_interaction[4][0], 1.e3],  # dmaj-nagt

                 ['gt', 'V', 0, 2,
                  solutions['gt'].volume_interaction[0][2], 1.e-7],  # py-andr
                 ['gt', 'V', 0, 4,
                  solutions['gt'].volume_interaction[0][4], 1.e-7],  # py-nagt
                 ['gt', 'V', 1, 1,
                  solutions['gt'].volume_interaction[1][1], 1.e-7]]  # alm-andr


solutions['bdg'].energy_interaction[0][0] = 0.  # make bdg ideal

# ['gt', 'E', 0, 0,
#         solutions['gt'].energy_interaction[0][0], 1.e3]  # py-alm ideal

endmember_priors = []
endmember_priors.extend([[mbr, 'S_0', endmembers[mbr].params['S_0_orig'][0],
                          endmembers[mbr].params['S_0_orig'][1]]
                         for mbr in ['per', 'wus',
                                     'fo', 'fa',
                                     'mwd',
                                     'mrw', 'frw',
                                     'alm', 'gr', 'andr',
                                     'di', 'hed',
                                     'oen', 'ofs',
                                     'mbdg', 'fbdg',
                                     'sp']])  # fwd S0 from special prior

endmembers['fwd'].params['V_0_orig'] = [endmembers['fwd'].params['V_0'],
                                        endmembers['fwd'].params['V_0']
                                        / 100. * 0.5] # 0.5% uncertainty
endmember_priors.extend([[mbr, 'V_0', endmembers[mbr].params['V_0_orig'][0],
                          endmembers[mbr].params['V_0_orig'][1]]
                         for mbr in ['fwd']])

endmembers['fwd'].params['K_0_orig'] = [endmembers['fwd'].params['K_0'],
                                        endmembers['fwd'].params['K_0']
                                        / 100. * 2.]  # 2% uncertainty
endmembers['frw'].params['K_0_orig'] = [endmembers['frw'].params['K_0'],
                                        endmembers['frw'].params['K_0']
                                        /100.*0.5]  # 0.5% uncertainty
endmembers['wus'].params['K_0_orig'] = [endmembers['wus'].params['K_0'],
                                        endmembers['wus'].params['K_0']
                                        / 100. * 2.]  # 2% uncertainty
endmember_priors.extend([[mbr, 'K_0', endmembers[mbr].params['K_0_orig'][0],
                          endmembers[mbr].params['K_0_orig'][1]]
                         for mbr in ['fwd', 'frw', 'wus']])

endmember_priors.extend([['per', 'a_0',
                          endmembers['per'].params['a_0_orig'], 2.e-7],
                         ['wus', 'a_0',
                          endmembers['wus'].params['a_0_orig'], 5.e-7],
                         ['fo',  'a_0',
                          endmembers['fo'].params['a_0_orig'], 2.e-7],
                         ['fa',  'a_0',
                          endmembers['fa'].params['a_0_orig'], 2.e-7],
                         ['mwd', 'a_0',
                          endmembers['mwd'].params['a_0_orig'], 5.e-7],
                         ['fwd', 'a_0',
                          endmembers['fwd'].params['a_0_orig'], 20.e-7],
                         ['mrw', 'a_0',
                          endmembers['mrw'].params['a_0_orig'], 2.e-7],
                         ['frw', 'a_0',
                          endmembers['frw'].params['a_0_orig'], 5.e-7],
                         ['mbdg', 'a_0',
                          endmembers['mbdg'].params['a_0_orig'], 2.e-7],
                         ['fbdg', 'a_0',
                          endmembers['fbdg'].params['a_0_orig'], 5.e-7]])

solution_priors = []
# Uncertainties from Frost data
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

experiment_uncertainties.extend([['Frost_2003_H1554', 'P', 0., 0.5e9],
                                 ['Frost_2003_H1555', 'P', 0., 0.5e9],
                                 ['Frost_2003_H1556', 'P', 0., 0.5e9],
                                 ['Frost_2003_H1582', 'P', 0., 0.5e9],
                                 ['Frost_2003_S2773', 'P', 0., 0.5e9],
                                 ['Frost_2003_V170', 'P', 0., 0.5e9],
                                 ['Frost_2003_V171', 'P', 0., 0.5e9],
                                 ['Frost_2003_V175', 'P', 0., 0.5e9],
                                 ['Frost_2003_V179', 'P', 0., 0.5e9]])

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

dict_experiment_uncertainties = {u[0]: {'P': 0., 'T': 0.}
                                 for u in experiment_uncertainties}
for u in experiment_uncertainties:
    dict_experiment_uncertainties[u[0]][u[1]] = u[2]

labels = []
labels.extend([a[0]+'_'+a[1] for a in endmember_args])
labels.extend(['{0}_{1}[{2},{3}]'.format(a[0], a[1], a[2], a[3])
               for a in solution_args])
labels.extend(['{0}_{1}'.format(a[0], a[1]) for a in experiment_uncertainties])


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

    i = 0

    # Endmember parameters
    for j, a in enumerate(endmember_args):
        dict_endmember_args[a[0]][a[1]] = args[i]*a[3]
        endmember_args[j][2] = args[i]*a[3]
        endmembers[a[0]].params[a[1]] = args[i]*a[3]
        i += 1

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
        i += 1

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
        i += 1

    # Special one-off constraints
    set_params_from_special_constraints()
    return None


#######################
# EXPERIMENTAL DATA ###
#######################

from datasets.Frost_2003_fper_ol_wad_rw import Frost_2003_assemblages
from datasets.Seckendorff_ONeill_1992_ol_opx import Seckendorff_ONeill_1992_assemblages
from datasets.ONeill_Wood_1979_ol_gt import ONeill_Wood_1979_assemblages
from datasets.ONeill_Wood_1979_CFMAS_ol_gt import ONeill_Wood_1979_CFMAS_assemblages
from datasets.endmember_reactions import endmember_reaction_assemblages
from datasets.Matsuzaka_et_al_2000_rw_wus_stv import Matsuzaka_2000_assemblages
from datasets.ONeill_1987_QFI import ONeill_1987_QFI_assemblages
from datasets.ONeill_1987_QFM import ONeill_1987_QFM_assemblages
from datasets.Nakajima_FR_2012_bdg_fper import Nakajima_FR_2012_assemblages
from datasets.Tange_TNFS_2009_bdg_fper_stv import Tange_TNFS_2009_FMS_assemblages
from datasets.Frost_2003_FMASO_garnet import Frost_2003_FMASO_gt_assemblages


# MAS
from datasets.Gasparik_1989_MAS_px_gt import Gasparik_1989_MAS_assemblages
from datasets.Gasparik_1992_MAS_px_gt import Gasparik_1992_MAS_assemblages
from datasets.Gasparik_Newton_1984_MAS_opx_sp_fo import Gasparik_Newton_1984_MAS_assemblages
from datasets.Gasparik_Newton_1984_MAS_py_opx_sp_fo import Gasparik_Newton_1984_MAS_univariant_assemblages
from datasets.Perkins_et_al_1981_MAS_py_opx import Perkins_et_al_1981_MAS_assemblages
#from datasets.Liu_et_al_2016_gt_bdg_cor import Liu_et_al_2016_MAS_assemblages
#from datasets.Liu_et_al_2017_bdg_cor import Liu_et_al_2017_MAS_assemblages
#from datasets.Hirose_et_al_2001_ilm_bdg_gt import Hirose_et_al_2001_MAS_assemblages

# CMS
from datasets.Carlson_Lindsley_1988_CMS_opx_cpx import Carlson_Lindsley_1988_CMS_assemblages

# CMAS
from datasets.Perkins_Newton_1980_CMAS_opx_cpx_gt import Perkins_Newton_1980_CMAS_assemblages
from datasets.Gasparik_1989_CMAS_px_gt import Gasparik_1989_CMAS_assemblages
from datasets.Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp import Klemme_ONeill_2000_CMAS_assemblages

# NMAS
from datasets.Gasparik_1989_NMAS_px_gt import Gasparik_1989_NMAS_assemblages

# NCMAS
from datasets.Gasparik_1989_NCMAS_px_gt import Gasparik_1989_NCMAS_assemblages

# CFMS
from datasets.Perkins_Vielzeuf_1992_CFMS_ol_cpx import Perkins_Vielzeuf_1992_CFMS_assemblages

# FASO
from datasets.Woodland_ONeill_1993_FASO_alm_sk import Woodland_ONeill_1993_FASO_assemblages

# NCFMASO
from datasets.Rohrbach_et_al_2007_NCFMASO_gt_cpx import Rohrbach_et_al_2007_NCFMASO_assemblages
from datasets.Beyer_et_al_2019_NCFMASO import Beyer_et_al_2019_NCFMASO_assemblages

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
                Perkins_Vielzeuf_1992_CFMS_assemblages,  # need ol, di-hed
                Gasparik_Newton_1984_MAS_assemblages,  # need sp, oen-mgts
                Gasparik_Newton_1984_MAS_univariant_assemblages,  # need sp, oen-mgts
                Perkins_Newton_1980_CMAS_assemblages,  # need oen_mgts_odi, di_cen_cats, py_gr
                Klemme_ONeill_2000_CMAS_assemblages,  # need sp, oen_mgts_odi, di_cen_cats, py_gr
                Gasparik_1992_MAS_assemblages,  # need oen-mgts, py-dmaj
                Gasparik_1989_MAS_assemblages,
                Gasparik_1989_CMAS_assemblages,
                Gasparik_1989_NMAS_assemblages,
                Gasparik_1989_NCMAS_assemblages,
                Rohrbach_et_al_2007_NCFMASO_assemblages,
                Beyer_et_al_2019_NCFMASO_assemblages]
               for assemblage in assemblage_list]

# minimize_func(get_params(), assemblages)

###################
# PUT PARAMS HERE #
###################

########################
# RUN THE MINIMIZATION #
########################
if run_inversion:

    import emcee

    # Make sure we always get the same walker starting points
    # (good for bug checking)
    np.random.seed(1234)

    jiggle_x0 = 1.e-3
    walker_multiplication_factor = 3  # this number must be greater than 2!
    n_steps_burn_in = 0  # number of steps in the burn in period (not used)
    n_steps_mcmc = 800  # number of steps in the full mcmc run
    n_discard = 0  # discard this number of steps from the full mcmc run
    thin = 1  # thin by this factor when calling get_chain

    x0 = get_params()
    ndim = len(x0)
    nwalkers = ndim*walker_multiplication_factor

    thisfilename = os.path.basename(__file__)
    base = os.path.splitext(thisfilename)[0]

    burnfile = base+'_sampler_after_burn_in.pickle'
    mcmcfile = base+'_sampler_after_mcmc_run.pickle'

    print('Running MCMC inversion with {0} parameters and {1} walkers'.format(ndim, nwalkers))
    print('This inversion will involve {0} burn-in steps and {1} stored steps'.format(n_steps_burn_in, n_steps_mcmc))
    print('The walkers will be clustered around a start point with a random jiggle of {0}'.format(jiggle_x0))
    print('The samplers will be saved to the following two pickle files:')
    print(burnfile)
    print(mcmcfile)

    # Currently the minerals are global objects,
    # so multithreading is not possible
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=[assemblages], threads=1)

    p0 = x0 + jiggle_x0*np.random.randn(nwalkers, ndim)

    if n_steps_burn_in > 0:
        print('Starting burn-in')
        state = sampler.run_mcmc(p0, n_steps_burn_in, progress=True)
        pickle.dump(sampler, open(burnfile, 'wb'))

        sampler.reset()
    else:
        state = p0

    print('Burn-in complete. Starting MCMC run.')

    half_steps = n_steps_mcmc/2
    state = sampler.run_mcmc(state, half_steps, progress=True)

    print('50% complete. Pickling intermediate')
    pickle.dump(sampler, open(mcmcfile+'int', 'wb'))

    state = sampler.run_mcmc(state, half_steps, progress=True)

    print('100% complete. Pickling')
    pickle.dump(sampler, open(mcmcfile, 'wb'))

    # sampler = pickle.load(open(mcmcfile+'int','rb'))
    # flat_samples = sampler.get_chain(discard=300, thin=thin, flat=True)
    flat_samples = sampler.get_chain(discard=n_discard, thin=thin, flat=True)


    for i in range(ndim):
        mcmc_i = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc_i)
        txt = "\\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc_i[1], q[0], q[1], labels[i])
        print(txt)

    # use the 50th percentile for the preferred params
    # (might not represent the best fit
    # if the distribution is strongly non-Gaussian...)
    mcmc_params = np.array([np.percentile(flat_samples[:, i], [50])[0]
                            for i in range(ndim)])
    mcmc_unc = np.array([np.percentile(flat_samples[:, i], [16, 50, 84])
                         for i in range(ndim)])
    set_params(mcmc_params)
    # print(minimize(minimize_func, get_params(),
    # args=(assemblages), method='BFGS')) # , options={'eps': 1.e-02}))

# Print the current parameters
print(get_params())

################
# A few images #
################

ol_polymorph_img = mpimg.imread('figures/ol_polymorphs.png')
ol_polymorph_img_1200C = mpimg.imread('figures/Akimoto_1987_fo_fa_phase_diagram_1200C.png')
ol_polymorph_img_1000C = mpimg.imread('figures/Akimoto_1987_fo_fa_phase_diagram_1000C.png')
ol_polymorph_img_800C = mpimg.imread('figures/Akimoto_1987_fo_fa_phase_diagram_800C.png')

ol_fper_img = mpimg.imread('figures/ol_fper_RTlnKD.png')
wad_fper_img = mpimg.imread('figures/wad_fper_RTlnKD.png')
rw_fper_img = mpimg.imread('figures/ring_fper_gt_KD.png')
rw_fper_part_img = mpimg.imread('figures/ring_fper_partitioning.png')


fo_phase_diagram = mpimg.imread('figures/Mg2SiO4_phase_diagram_Jacobs_2017.png')
fa_phase_diagram = mpimg.imread('figures/Fe2SiO4_phase_diagram_Yagi_1987.png')
fa_phase_diagram2 = mpimg.imread('figures/Fe2SiO4_phase_diagram_Jacobs_2001.png')

############
# FUNCTION #
############


def eqm_pressures(m1, m2, temperatures):
    composition = m1.formula
    assemblage = burnman.Composite([m1, m2])
    assemblage.set_state(1.e5, temperatures[0])
    equality_constraints = [('T', temperatures),
                            ('phase_proportion', (m1, 0.0))]
    sols, prm = burnman.equilibrate(composition, assemblage,
                                    equality_constraints,
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
    # G is deltaG = G_per + G_fa/2. - G_fper - G_fo/2.
    x_fper = v[0]
    if np.abs(np.abs(x_ol - 0.5) - 0.5) < 1.e-10:
        v[0] = x_ol
        return 0.
    else:
        KD = ((x_ol*(1. - x_fper))
              / (x_fper*(1. - x_ol)))
        if KD < 0.:
            KD = 1.e-12
        return (G - W_ol*(2.*x_ol - 1.) - W_fper*(1 - 2.*x_fper)
                + burnman.constants.gas_constant*T*np.log(KD))


viridis = cm.get_cmap('viridis', 101)


fig = plt.figure(figsize=(30, 10))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

# OLIVINE
ax[0].imshow(ol_fper_img, extent=[0.0, 0.8, -45000., -5000.], aspect='auto')


Pmin = 0.e9
Pmax = 15.e9

T = 1673.15
mins = [endmembers[m]
        for m in ['per', 'wus', 'fo', 'fa', 'mwd', 'fwd', 'mrw', 'frw']]
for P in [1.e5, 5.e9, 10.e9, 15.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (endmembers['per'].gibbs - endmembers['wus'].gibbs
         - endmembers['fo'].gibbs/2. + endmembers['fa'].gibbs/2.)
    W_ol = (solutions['ol'].solution_model.We[0][1]
            + solutions['ol'].solution_model.Wv[0][1] * P) / 2.  # 1 cation
    W_fper = (solutions['mw'].solution_model.We[0][1]
              + solutions['mw'].solution_model.Wv[0][1] * P)

    x_ols = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_ol],
                               args=(x_ol, G, T, W_ol, W_fper))[0]
                        for x_ol in x_ols])
    KDs = ((x_ols*(1. - x_fpers))
           / (x_fpers*(1. - x_ols)))
    ax[0].plot(x_ols, burnman.constants.gas_constant*T*np.log(KDs),
               color=viridis((P - Pmin) / (Pmax - Pmin)),
               linewidth=3., label='{0} GPa'.format(P/1.e9))


P_Xol_RTlnKDs = []
for assemblage in Frost_2003_assemblages:
    if solutions['ol'] in assemblage.phases:
        idx_ol = assemblage.phases.index(solutions['ol'])
        idx_mw = assemblage.phases.index(solutions['mw'])
        T = assemblage.nominal_state[1]
        x_ol = assemblage.stored_compositions[idx_ol][0][1]
        x_fper = assemblage.stored_compositions[idx_mw][0][1]
        RTlnKD = (burnman.constants.gas_constant * T
                  * np.log((x_ol*(1. - x_fper))
                           / (x_fper*(1. - x_ol))))
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
    G = (endmembers['per'].gibbs - endmembers['wus'].gibbs
         - endmembers['mwd'].gibbs/2. + endmembers['fwd'].gibbs/2.)
    W_wad = (solutions['wad'].solution_model.We[0][1]
             + solutions['wad'].solution_model.Wv[0][1] * P) / 2.  # 1 cation
    W_fper = (solutions['mw'].solution_model.We[0][1]
              + solutions['mw'].solution_model.Wv[0][1] * P)

    x_wads = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_wad],
                               args=(x_wad, G, T, W_wad, W_fper))[0]
                        for x_wad in x_wads])
    KDs = ((x_wads*(1. - x_fpers))
           / (x_fpers*(1. - x_wads)))
    ax[1].plot(x_wads, burnman.constants.gas_constant*T*np.log(KDs),
               color=viridis((P-Pmin)/(Pmax-Pmin)),
               linewidth=3., label='{0} GPa'.format(P/1.e9))

P_Xwad_RTlnKDs = []
for assemblage in Frost_2003_assemblages:
    if solutions['wad'] in assemblage.phases:
        idx_wad = assemblage.phases.index(solutions['wad'])
        idx_mw = assemblage.phases.index(solutions['mw'])
        x_wad = assemblage.stored_compositions[idx_wad][0][1]
        x_fper = assemblage.stored_compositions[idx_mw][0][1]
        T = assemblage.nominal_state[1]
        RTlnKD = (burnman.constants.gas_constant * T
                  * np.log((x_wad*(1. - x_fper))
                           / (x_fper*(1. - x_wad))))
        P_Xwad_RTlnKDs.append([assemblage.nominal_state[0],
                               x_wad, RTlnKD])

pressures, x_wads, RTlnKDs = np.array(P_Xwad_RTlnKDs).T
ax[1].scatter(x_wads, RTlnKDs, c=pressures, s=80., label='data',
              cmap=viridis, vmin=Pmin, vmax=Pmax)

ax[1].set_xlim(0., 0.4)
ax[1].legend(loc='best')


# RINGWOODITE
# ax[2].imshow(rw_fper_part_img, extent=[0.0, 1., 0., 1.], aspect='auto')

viridis = cm.get_cmap('viridis', 101)
Pmin = 10.e9
Pmax = 24.e9


T = 1673.15
for P in [10.e9, 12.5e9, 15.e9, 17.5e9, 20.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (endmembers['per'].gibbs - endmembers['wus'].gibbs
         - endmembers['mrw'].gibbs / 2. + endmembers['frw'].gibbs / 2.)
    W_rw = ((child_solutions['ring'].solution_model.We[0][1]
             + child_solutions['ring'].solution_model.Wv[0][1] * P)
            / 2.)  # 1 cation
    W_fper = (solutions['mw'].solution_model.We[0][1]
              + solutions['mw'].solution_model.Wv[0][1] * P)

    x_rws = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_rw],
                               args=(x_rw, G, T, W_rw, W_fper))[0]
                        for x_rw in x_rws])

    ax[2].plot(x_rws, x_fpers, color=viridis((P-Pmin)/(Pmax-Pmin)),
               linewidth=3., label=P/1.e9)


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

# plt.imshow(ol_polymorph_img, extent=[0., 1., 6., 20.], aspect='auto')
# plt.imshow(ol_polymorph_img_1000C, extent=[-0.01, 1.005, 4., 21.],
# aspect='auto')

for (T0, color) in [(1273.15, 'blue'),
                    (1673.15, 'orange'),
                    (2073.15, 'purple')]:

    x_m1 = 0.3

    composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1.-x_m1), 'Si': 1., 'O': 4.}
    solutions['wad'].guess = np.array([1. - x_m1, x_m1])
    solutions['ol'].guess = np.array([1. - x_m1, x_m1])
    child_solutions['ring'].guess = np.array([0.15, 0.85])
    assemblage = burnman.Composite([solutions['ol'],
                                    solutions['wad'],
                                    child_solutions['ring']])
    assemblage.set_state(14.e9, T0)
    equality_constraints = [('T', T0), ('phase_proportion',
                                        (solutions['ol'], 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage,
                                   equality_constraints,
                                   store_iterates=False,
                                   initial_state_from_assemblage=True)
    P_inv = assemblage.pressure
    x_ol_inv = assemblage.phases[0].molar_fractions[1]
    x_wad_inv = assemblage.phases[1].molar_fractions[1]
    x_rw_inv = assemblage.phases[2].molar_fractions[1]
    for (m1, m2) in [(solutions['wad'], solutions['ol']),
                     (solutions['wad'], child_solutions['ring']),
                     (solutions['ol'], child_solutions['ring'])]:
        composition = {'Fe': 0., 'Mg': 2., 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([m1.endmembers[0][0],
                                        m2.endmembers[0][0]])
        equality_constraints = [('T', T0), ('phase_proportion',
                                            (m1.endmembers[0][0], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage,
                                       equality_constraints,
                                       store_iterates=False)
        P1 = assemblage.pressure
        composition = {'Fe': 2., 'Mg': 0., 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([m1.endmembers[1][0],
                                        m2.endmembers[1][0]])
        equality_constraints = [('T', T0), ('phase_proportion',
                                            (m1.endmembers[1][0], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage,
                                       equality_constraints,
                                       store_iterates=False)
        P0 = assemblage.pressure
        print(P0/1.e9, P1/1.e9)
        if m1 is solutions['wad']:
            x_m1s = np.linspace(0.001, x_wad_inv, 21)
        else:
            x_m1s = np.linspace(x_ol_inv, 0.999, 21)

        pressures = np.empty_like(x_m1s)
        x_m2s = np.empty_like(x_m1s)
        m1.guess = np.array([1. - x_m1s[0], x_m1s[0]])
        m2.guess = np.array([1. - x_m1s[0], x_m1s[0]])
        for i, x_m1 in enumerate(x_m1s):
            composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1),
                           'Si': 1., 'O': 4.}
            assemblage = burnman.Composite([m1, m2])
            assemblage.set_state(P1*(1 - x_m1) + P0*x_m1, T0)
            m1.set_composition([1. - x_m1, x_m1])
            m2.set_composition(m2.guess)
            assemblage.n_moles = 1.
            assemblage.set_fractions([1., 0.])
            equality_constraints = [('T', T0), ('phase_proportion', (m2, 0.0))]
            sol, prm = burnman.equilibrate(composition, assemblage,
                                           equality_constraints,
                                           initial_state_from_assemblage=True,
                                           initial_composition_from_assemblage=True,
                                           store_iterates=False)

            m2.guess = m2.molar_fractions
            x_m2s[i] = m2.molar_fractions[1]
            pressures[i] = assemblage.pressure

        plt.plot(x_m1s, pressures/1.e9, linewidth=3., color=color)
        plt.plot(x_m2s, pressures/1.e9, linewidth=3.,
                 color=color, label='{0} K'.format(T0))
    plt.plot([x_ol_inv, x_rw_inv], [P_inv/1.e9, P_inv/1.e9], linewidth=3.,
             color=color)

    """
    # bdg + fper
    x_m1s = []
    pressures = []
    x_m2s = []
    for x_m1 in np.linspace(0.3, 0.5, 51):
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
        child_solutions['mg_fe_bdg'].guess = np.array([1. - x_m1, x_m1])
        fper.guess = np.array([1. - x_m1, x_m1])
        assemblage = burnman.Composite([child_solutions['mg_fe_bdg'],
                                        fper, stv])
        assemblage.set_state(30.e9, T0)
        equality_constraints = [('T', T0),
                                ('phase_proportion', (stv, 0.))]
        sol, prm = burnman.equilibrate(composition, assemblage,
                                       equality_constraints,
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

    child_solutions['ring'].guess = np.array([1. - x_m1_array[0],
                                              x_m1_array[0]])
    child_solutions['mg_fe_bdg'].guess = np.array([1. - x_m1_array[0],
                                                   x_m1_array[0]])
    solutions['mw'].guess = np.array([1. - x_m1_array[0], x_m1_array[0]])

    for x_m1 in x_m1_array:
        composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}

        child_solutions['ring'].set_composition([1. - x_m1, x_m1])
        child_solutions['mg_fe_bdg'].set_composition(child_solutions['mg_fe_bdg'].guess)
        solutions['mw'].set_composition(solutions['mw'].guess)

        assemblage = burnman.Composite([child_solutions['ring'],
                                        child_solutions['mg_fe_bdg'],
                                        solutions['mw']], [1., 0., 0.])
        assemblage.set_state(25.e9, T0)
        equality_constraints = [('T', T0),
                                ('phase_proportion',
                                 (child_solutions['ring'], 1.0))]
        sol, prm = burnman.equilibrate(composition, assemblage,
                                       equality_constraints,
                                       initial_composition_from_assemblage=True,
                                       initial_state_from_assemblage=True,
                                       store_iterates=False)
        if sol.success:
            print(assemblage.pressure/1.e9)
            x_m1s.append(x_m1)
            x_m2s.append((solutions['mw'].molar_fractions[1]
                          + child_solutions['mg_fe_bdg'].molar_fractions[1])
                         / 2.)

            child_solutions['mg_fe_bdg'].guess = child_solutions['mg_fe_bdg'].molar_fractions
            solutions['mw'].guess = solutions['mw'].molar_fractions

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
        assemblage = burnman.Composite([child_solutions['ring'],
                                        solutions['mw'],
                                        endmembers['stv']],
                                       [0., 2./3., 1./3.])
        assemblage.set_state(Pi, T0)

        solutions['mw'].set_composition([1. - x_m1, x_m1])
        child_solutions['ring'].set_composition(child_solutions['ring'].guess)

        equality_constraints = [('T', T0),
                                ('phase_proportion',
                                 (child_solutions['ring'], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage,
                                       equality_constraints,
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

                        P_Xmg_phase[m].append([assemblage.nominal_state[0],
                                               P_shift,
                                               assemblage.stored_compositions[i][0][1]])
                except:
                    if phase == child_solutions[m]:
                        P_shift = dict_experiment_uncertainties[assemblage.experiment_id]['P']

                        P_Xmg_phase[m].append([assemblage.nominal_state[0],
                                               P_shift,
                                               assemblage.stored_compositions[i][0][1]])

arrow_params = {'shape': 'full',
                'width': 0.001,
                'length_includes_head': True,
                'head_starts_at_zero': False}

for m in ['ol', 'wad', 'ring']:
    pressures, pressure_shift, xs = np.array(P_Xmg_phase[m]).T
    for i in range(len(xs)):
        plt.arrow(xs[i], pressures[i]/1.e9, 0.,
                  pressure_shift[i]/1.e9,
                  **arrow_params)
    plt.scatter(xs, pressures/1.e9, s=80., label='data')

plt.legend()
plt.show()
