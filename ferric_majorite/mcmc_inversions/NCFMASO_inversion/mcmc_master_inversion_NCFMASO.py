from __future__ import absolute_import
from __future__ import print_function

import platform
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from input_dataset import create_minerals
from fitting_functions import Storage, log_probability
from fitting_functions import get_params, set_params
from output_plots import chain_plotter, plots
import pickle
import emcee
from multiprocessing import Pool

import pandas as pd
import seaborn as sns

print('TODO!! Check endmembers and priors, add solution parameters and priors')
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


# Component defining endmembers (for H_0 and S_0) are:
# Fe: Fe metal (BCC, FCC, HCP)
# O: O2
# Mg: MgO per
# Si: SiO2 qtz
# Al: Mg3Al2Si3O12 pyrope
# Ca: CaMgSi2O6 diopside
# Na: NaAlSi2O6 jadeite


mineral_dataset = create_minerals()
endmembers = mineral_dataset['endmembers']
solutions = mineral_dataset['solutions']
child_solutions = mineral_dataset['child_solutions']

endmember_args = [[mbr, 'H_0', endmembers[mbr].params['H_0'], 1.e3]
                  for mbr in ['wus',
                              'fo', 'fa',
                              'mwd',  # fwd H0 from special prior
                              'mrw', 'frw',
                              'herc', 'sp', 'mt',
                              'alm', 'gr', 'andr', 'dmaj', 'nagt',
                              'coe', 'stv',
                              'hed', 'cen', 'cfs', 'cats', 'aeg',
                              'oen', 'ofs', 'mgts',
                              'hen', 'hfs',
                              'mbdg', 'fbdg',
                              'cpv']]

endmember_args.extend([[mbr, 'S_0', endmembers[mbr].params['S_0'], 1.]
                       for mbr in ['per', 'wus',
                                   'fo', 'fa',
                                   'mwd',  # fwd S0 from special prior
                                   'mrw', 'frw',
                                   'herc', 'sp', 'mt',
                                   'py', 'alm', 'gr', 'andr', 'dmaj', 'nagt',
                                   'coe', 'stv',
                                   'di', 'hed',  # 'cen', 'cfs', 'cats', 'aeg',
                                   'oen', 'ofs', 'mgts',
                                   'hen', 'hfs',
                                   'mbdg', 'fbdg',
                                   'cpv']])

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

endmember_priors = [[mbr, 'S_0', endmembers[mbr].params['S_0_orig'][0],
                     endmembers[mbr].params['S_0_orig'][1]]
                    for mbr in ['per', 'wus',
                                'fo', 'fa',
                                'mwd',
                                'mrw', 'frw',
                                'py', 'alm', 'gr', 'andr',
                                'di', 'hed',
                                'oen', 'ofs',
                                'mbdg', 'fbdg',
                                'sp']]

endmembers['fwd'].params['V_0_orig'] = [endmembers['fwd'].params['V_0'],
                                        endmembers['fwd'].params['V_0']
                                        / 100. * 0.05]  # 0.05% uncertainty
endmember_priors.extend([[mbr, 'V_0', endmembers[mbr].params['V_0_orig'][0],
                          endmembers[mbr].params['V_0_orig'][1]]
                         for mbr in ['fwd']])

endmembers['fwd'].params['K_0_orig'] = [endmembers['fwd'].params['K_0'],
                                        endmembers['fwd'].params['K_0']
                                        / 100. * 2.]  # 2% uncertainty
endmembers['frw'].params['K_0_orig'] = [endmembers['frw'].params['K_0'],
                                        endmembers['frw'].params['K_0']
                                        / 100. * 0.5]  # 0.5% uncertainty
endmembers['wus'].params['K_0_orig'] = [endmembers['wus'].params['K_0'],
                                        endmembers['wus'].params['K_0']
                                        / 100. * 1.]  # 1% uncertainty
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
                          endmembers['fwd'].params['a_0_orig'], 5.e-7],
                         ['mrw', 'a_0',
                          endmembers['mrw'].params['a_0_orig'], 2.e-7],
                         ['frw', 'a_0',
                          endmembers['frw'].params['a_0_orig'], 5.e-7],
                         ['mbdg', 'a_0',
                          endmembers['mbdg'].params['a_0_orig'], 2.e-7],
                         ['fbdg', 'a_0',
                          endmembers['fbdg'].params['a_0_orig'], 5.e-7]])

solution_args = [['mw', 'E', 0, 0,
                  solutions['mw'].energy_interaction[0][0], 1.e3],
                 ['ol', 'E', 0, 0,
                  solutions['ol'].energy_interaction[0][0], 1.e3],
                 ['wad', 'E', 0, 0,
                  solutions['wad'].energy_interaction[0][0], 1.e3],

                 ['sp', 'E', 0, 0,
                  solutions['sp'].energy_interaction[0][0], 1.e3],  # sp-herc
                 ['sp', 'E', 3, 0,
                  solutions['sp'].energy_interaction[3][0], 1.e3],  # mrw-frw

                 ['bdg', 'E', 0, 0,
                  solutions['bdg'].energy_interaction[0][0], 1.e3],  # mgfe bdg

                 ['opx', 'E', 0, 0,
                  solutions['opx'].energy_interaction[0][0], 1.e3],  # oen-ofs
                 ['opx', 'E', 0, 1,
                  solutions['opx'].energy_interaction[0][1], 1.e3],  # oen-mgts
                 ['opx', 'E', 0, 2,
                  solutions['opx'].energy_interaction[0][2], 1.e3],  # oen-odi
                 ['opx', 'E', 1, 0,
                  solutions['opx'].energy_interaction[1][0], 1.e3],  # ofs-mgts
                 ['opx', 'E', 1, 1,
                  solutions['opx'].energy_interaction[1][1], 1.e3],  # ofs-odi
                 ['opx', 'E', 2, 0,
                  solutions['opx'].energy_interaction[2][0], 1.e3],  # mgts-odi
                 ['opx', 'E', 2, 1,
                  solutions['opx'].energy_interaction[2][1], 1.e3],  # mgts-ofm
                 ['opx', 'E', 3, 0,
                  solutions['opx'].energy_interaction[3][0], 1.e3]]  # odi-ofm

for i in range(6):  # di=0, hed=1, cen=2, cfs=3, cats=4, jd=5, aeg=6, ignore od
    for j in range(6-i):
        solution_args.append(['cpx', 'E', i, j,
                              solutions['cpx'].energy_interaction[i][j],
                              1.e3])

for i in range(5):  # py=0, alm=1, gr=2, andr=3, dmaj=4, nagt=5, no od
    for j in range(5-i):
        solution_args.append(['gt', 'E', i, j,
                              solutions['gt'].energy_interaction[i][j],
                              1.e3])

for (i, j) in [(0, 2), (0, 4), (1, 1)]:  # py-andr, py-nagt, alm-andr
    solution_args.append(['gt', 'V', i, j,
                          solutions['gt'].volume_interaction[i][j],
                          1.e-7])

solution_priors = [['opx', 'E', 0, 0, 7.e3, 2.e3],  # oen-ofs
                   ['opx', 'E', 0, 1, 12.5e3, 2.e3],  # oen-mgts
                   ['opx', 'E', 0, 2, 32.2e3, 5.e3],  # oen-odi
                   ['opx', 'E', 1, 0, 11.e3, 2.e3],  # ofs-mgts
                   ['opx', 'E', 1, 1, 25.54e3, 5.e3],  # ofs-odi
                   ['opx', 'E', 2, 0, 75.5e3, 30.e3],  # mgts-odi
                   ['opx', 'E', 2, 1, 15.0e3, 5.e3],  # mgts-ofm
                   ['opx', 'E', 3, 0, 25.54e3, 5.e3],  # odi-ofm

                   ['gt', 'E', 0, 0, 3.e3, 2.e3],  # py-alm
                   ['gt', 'E', 0, 1, 30.e3, 3.e3],  # py-gr
                   ['bdg', 'E', 0, 0, 5.e3, 5.e3]]  # mbdg-fbdg

# Some fairly lax priors on cpx solution parameters
for i in range(6):  # di=0, hed=1, cen=2, cfs=3, cats=4, jd=5, aeg=6, ignore od
    for j in range(6-i):
        solution_priors.append(['cpx', 'E', i, j,
                                solutions['cpx'].energy_interaction[i][j],
                                2.e3 + 0.3*solutions['cpx'].energy_interaction[i][j]])

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


def special_constraints(dataset, storage):

    endmembers = dataset['endmembers']
    solutions = dataset['solutions']

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

    # 2) Fix odi (just in case we decide to fit di H, S or V at some point)
    endmembers['odi'].params['H_0'] = endmembers['di'].params['H_0'] - 0.1e3
    endmembers['odi'].params['S_0'] = endmembers['di'].params['S_0'] - 0.211
    endmembers['odi'].params['V_0'] = endmembers['di'].params['V_0'] + 0.005e-5

    # 3) Make sure the temperature dependence of ordering
    # is preserved in Mg-Fe opx and hpx
    wMgFe = solutions['opx'].energy_interaction[0][0] / 4. + 2.25e3
    Etweak = solutions['opx'].energy_interaction[0][0] / 4. - 8.35e3

    solutions['opx'].energy_interaction[0][-1] = wMgFe  # oen-ofm
    solutions['opx'].energy_interaction[1][-1] = wMgFe  # ofs-ofm
    endmembers['ofm'].property_modifiers[0][1]['delta_E'] = Etweak
    endmembers['hfm'].property_modifiers[0][1]['delta_E'] = Etweak

    # 4) Copy interaction parameters from opx to hpx:
    solutions['hpx'].alphas = solutions['opx'].alphas
    solutions['hpx'].energy_interaction = solutions['opx'].energy_interaction
    solutions['hpx'].entropy_interaction = solutions['opx'].entropy_interaction
    solutions['hpx'].volume_interaction = solutions['opx'].volume_interaction


# Create storage object
storage = Storage({'endmember_args': endmember_args,
                   'solution_args': solution_args,
                   'endmember_priors': endmember_priors,
                   'solution_priors': solution_priors,
                   'experiment_uncertainties': experiment_uncertainties})


# Create labels for each parameter
labels = [a[0]+'_'+a[1] for a in endmember_args]
labels.extend(['{0}_{1}[{2},{3}]'.format(a[0], a[1], a[2], a[3])
               for a in solution_args])
labels.extend(['{0}_{1}'.format(a[0], a[1]) for a in experiment_uncertainties])


#######################
# EXPERIMENTAL DATA ###
#######################


from datasets import endmember_reactions

# CMS
from datasets import Carlson_Lindsley_1988_CMS_opx_cpx

# FMS
from datasets import Frost_2003_fper_ol_wad_rw
from datasets import Tsujino_et_al_2019_FMS_wad_rw
from datasets import Katsura_et_al_2004_FMS_ol_wad
from datasets import Seckendorff_ONeill_1992_ol_opx
from datasets import Matsuzaka_et_al_2000_rw_wus_stv
from datasets import Nakajima_FR_2012_bdg_fper
from datasets import Tange_TNFS_2009_bdg_fper_stv

# MAS
from datasets import Gasparik_1989_MAS_px_gt
from datasets import Gasparik_1992_MAS_px_gt
from datasets import Gasparik_Newton_1984_MAS_opx_sp_fo
from datasets import Gasparik_Newton_1984_MAS_py_opx_sp_fo
from datasets import Perkins_et_al_1981_MAS_py_opx
#from datasets import Liu_et_al_2016_gt_bdg_cor
#from datasets import Liu_et_al_2017_bdg_cor
#from datasets import Hirose_et_al_2001_ilm_bdg_gt


# NMAS
from datasets import Gasparik_1989_NMAS_px_gt

# CMAS
from datasets import Perkins_Newton_1980_CMAS_opx_cpx_gt
from datasets import Gasparik_1989_CMAS_px_gt
from datasets import Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp

# CFMS
from datasets import Perkins_Vielzeuf_1992_CFMS_ol_cpx

# FMSO
from datasets import ONeill_1987_QFI
from datasets import ONeill_1987_QFM

# FMAS
from datasets import ONeill_Wood_1979_ol_gt
from datasets import Jamieson_Roeder_1984_FMAS_ol_sp

# FASO
from datasets import Woodland_ONeill_1993_FASO_alm_sk

# NCMAS
from datasets import Gasparik_1989_NCMAS_px_gt

# CFMAS
from datasets import ONeill_Wood_1979_CFMAS_ol_gt

# FMASO
from datasets import Frost_2003_FMASO_garnet

# NCFMASO
from datasets import Rohrbach_et_al_2007_NCFMASO_gt_cpx
from datasets import Beyer_et_al_2019_NCFMASO

# raw is -215
assemblages = [assemblage for assemblage_list in
               [module.get_assemblages(mineral_dataset)
                for module in [endmember_reactions,  # 73, 2825
                               Frost_2003_fper_ol_wad_rw,  # 143, 512
                               Katsura_et_al_2004_FMS_ol_wad, # 6, 228
                               Tsujino_et_al_2019_FMS_wad_rw, # 13, 429
                               Seckendorff_ONeill_1992_ol_opx,  # 46, 886
                               Jamieson_Roeder_1984_FMAS_ol_sp,  # 18, 1005
                               ONeill_Wood_1979_ol_gt,  # 54, 548
                               ONeill_Wood_1979_CFMAS_ol_gt,  # 20, 176
                               Matsuzaka_et_al_2000_rw_wus_stv,  # 19, 166 assume all Fe as Fe2+
                               ONeill_1987_QFI,  # 103, 168
                               ONeill_1987_QFM,  # 38, 168
                               Nakajima_FR_2012_bdg_fper,  # 10, 364
                               Tange_TNFS_2009_bdg_fper_stv,  # 8, 481
                               Frost_2003_FMASO_garnet,  # 36, 391
                               Gasparik_1989_MAS_px_gt,  # 9, 234
                               Gasparik_1992_MAS_px_gt,  # 12, 232
                               Gasparik_Newton_1984_MAS_opx_sp_fo,  # 14 634
                               Gasparik_Newton_1984_MAS_py_opx_sp_fo,  # 2, 230
                               Perkins_et_al_1981_MAS_py_opx,  # 91, 553
                               Carlson_Lindsley_1988_CMS_opx_cpx,  # 40, 264
                               Perkins_Newton_1980_CMAS_opx_cpx_gt,  # 12, 894
                               Gasparik_1989_CMAS_px_gt,  # 18, 1703
                               Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp,  # 14 572
                               Gasparik_1989_NMAS_px_gt,  # 12 1540
                               # Gasparik_1989_NCMAS_px_gt,  # 5, 1982 threw an error during early fitting
                               Perkins_Vielzeuf_1992_CFMS_ol_cpx,  # 15, 179
                               # Woodland_ONeill_1993_FASO_alm_sk, # 21, 11134 I don't think we have a good enough spinel model yet
                               Rohrbach_et_al_2007_NCFMASO_gt_cpx,  # 5, 1201
                               Beyer_et_al_2019_NCFMASO  # 12, 9230
                               ]]
               for assemblage in assemblage_list]

dataset = {'endmembers': mineral_dataset['endmembers'],
           'solutions': mineral_dataset['solutions'],
           'child_solutions': mineral_dataset['child_solutions'],
           'assemblages': assemblages}


# Initialize parameters and pepare internal arrays
# This should speed things up after depickling
def initialise_params():
    from import_params import FMSO_storage, transfer_storage

    print('Initializing parameters from FMSO output...')
    transfer_storage(from_storage=FMSO_storage,
                     to_storage=storage)
    lnprob = log_probability(get_params(storage), dataset, storage,
                             special_constraints)
    print('Initial ln(p) = {0}'.format(lnprob))
    return None


print('Fitting {0} assemblages'.format(len(dataset['assemblages'])))
initialise_params()

########################
# RUN THE MINIMIZATION #
########################
if run_inversion:
    # Make sure we always get the same walker starting points
    # (good for bug checking)
    np.random.seed(1234)

    jiggle_x0 = 1.e-3
    walker_multiplication_factor = 4  # this number must be greater than 2!
    n_steps_burn_in = 0  # number of steps in the burn in period (not used)
    n_steps_mcmc = 2000  # number of steps in the full mcmc run
    n_discard = 0  # discard this number of steps from the full mcmc run
    thin = 1  # thin by this factor when calling get_chain

    x0 = get_params(storage)
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

    print('using n threads')

    p0 = x0 + jiggle_x0*np.random.randn(nwalkers, ndim)

    if platform.node() == "axolotl.gly.bris.ac.uk": #"ix.gly.bris.ac.uk":
        print('This computer is ix.gly.bris.ac.uk. Leaving two cpus free.')
        processes = os.cpu_count()-2  # NT suggests keeping two cpus free
    else:
        print('This computer is not ix.gly.bris.ac.uk. Using all cpus.')
        processes = None  # default is to use all of the cpus

    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=[dataset, storage,
                                              special_constraints],
                                        pool=pool)

        if n_steps_burn_in > 0:
            print('Starting burn-in')
            state = sampler.run_mcmc(p0, n_steps_burn_in, progress=True)
            pickle.dump(sampler, open(burnfile,'wb'))

            sampler.reset()
        else:
            state = p0

        sampler = pickle.load(open(mcmcfile,'rb'))
        sampler.pool = pool  # deal with change in pool!
        #state = sampler.run_mcmc(sampler._previous_state, n_steps_mcmc, progress=True)

        #print('Burn-in complete. Starting MCMC run.')
        #state = sampler.run_mcmc(state, n_steps_mcmc, progress=True)

        #print('100% complete. Pickling')
        #pickle.dump(sampler, open(mcmcfile,'wb'))

    print('Chain shape: {0}'.format(sampler.get_chain().shape))
    print('Mean acceptance fraction: {0:.2f}'
          ' (should ideally be between 0.25 and 0.5)'.format(np.mean(sampler.acceptance_fraction)))

    if np.mean(sampler.acceptance_fraction) < 0.12:
        print(sampler.get_chain().shape)
        print(sampler.acceptance_fraction)
        exit()

    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(e)

    plot_chains = False
    if plot_chains:
        chain_plotter(sampler, labels)

    flat_samples = sampler.get_chain(discard=1600, thin=thin, flat=True)
    #flat_samples = sampler.get_chain(discard=n_discard, thin=thin, flat=True)

    # use the 50th percentile for the preferred params
    # (might not represent the best fit
    # if the distribution is strongly non-Gaussian...)
    mcmc_params = np.array([np.percentile(flat_samples[:, i], [50])[0]
                            for i in range(ndim)])
    mcmc_unc = np.array([np.percentile(flat_samples[:, i], [16, 50, 84])
                         for i in range(ndim)])

    for i in range(ndim):
        mcmc_i = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc_i)
        txt = "\\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc_i[1], q[0], q[1], labels[i])
        print(txt)

    Mcorr = np.corrcoef(flat_samples.T)
    triu_Mcorr = np.triu(m=Mcorr, k=1)
    abs_Mcorr_sorted_indices = np.unravel_index(np.argsort(np.abs(triu_Mcorr),
                                                           axis=None)[::-1],
                                                triu_Mcorr.shape)
    Mcov = np.cov(flat_samples.T) # each row corresponds to a different variable
    #import corner
    #fig = corner.corner(flat_samples, labels=labels);
    #fig.savefig('corner_plot.pdf')
    #plt.show()

    fig, ax = plt.subplots(figsize=(50,50))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    ax = sns.heatmap(pd.DataFrame(Mcorr),
                     xticklabels=labels,
                     yticklabels=labels,
                     cmap=cmap,
                     vmin=-1.,
                     center=0.,
                     vmax=1.)
    fig.savefig('parameter_correlations.pdf')
    plt.show()

    set_params(mcmc_params, dataset, storage, special_constraints)
    # print(minimize(minimize_func, get_params(),
    # args=(assemblages), method='BFGS')) # , options={'eps': 1.e-02}))

# Print the current parameters
print(get_params(storage))

# Make plots
plots(dataset, storage)
