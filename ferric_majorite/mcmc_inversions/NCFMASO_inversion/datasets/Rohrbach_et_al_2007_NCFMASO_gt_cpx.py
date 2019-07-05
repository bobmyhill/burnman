import numpy as np

from input_dataset import *
from fitting_functions import equilibrium_order

# Garnet-clinopyroxene partitioning data
with open('data/Rohrbach_et_al_2007_NCFMASO_gt_cpx.dat', 'r') as f:
    expt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

set_runs = ['zuII_4', 'zuIII_1', 'zu_8', 'zuII_3', 'zuIV_1']

# Temporarily change the formulae of the endmembers for fitting purposes!!
solutions['gt'].endmember_formulae[solutions['gt'].endmember_names.index('andr')] = {'O':  12.0,
                                                           'Ca':  3.0,
                                                           'Fef': 2.0,
                                                           'Si':  3.0}
solutions['cpx'].endmember_formulae[solutions['cpx'].endmember_names.index('acm')] = {'O':  6.0,
                                                                  'Na':  1.0,
                                                                  'Fef': 1.0,
                                                                  'Si': 2.0}

solutions['cpx'].endmember_formulae[solutions['cpx'].endmember_names.index('cfs')] = {'O':  6.0,
                                                                  'Fea': 1.0,
                                                                  'Fe': 2.0,
                                                                  'Si': 2.0}

Rohrbach_et_al_2007_NCFMASO_assemblages = []
for i, run_id in enumerate(set_runs):
    # always garnet then cpx
    gt_idx, cpx_idx = [idx for idx, d in enumerate(expt_data) if d[0] == run_id]

    pressure = float(expt_data[gt_idx][1]) * 1.e9 # GPa to Pa
    temperature = float(expt_data[gt_idx][2]) + 273.15 # C to K
    if pressure > 5.e9:
        assemblage = burnman.Composite([solutions['gt'],
                                        solutions['cpx'], endmembers['fcc_iron']])
    else:
        assemblage = burnman.Composite([child_solutions['xmj_gt'],
                                        solutions['cpx'], endmembers['fcc_iron']])

    assemblage.experiment_id = 'Rohrbach_et_al_2007_NCFMASO_{0}'.format(run_id)
    assemblage.nominal_state = np.array([pressure, temperature])
    assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

    c_gt = np.array(list(map(float, expt_data[gt_idx][5:])))
    c_px = np.array(list(map(float, expt_data[cpx_idx][5:])))

    gt_Si, gt_Ti, gt_Al, gt_Cr, gt_Fetot, gt_Mg, gt_Ca, gt_Na, gt_FoF, gt_FoF_unc = c_gt
    px_Si, px_Ti, px_Al, px_Cr, px_Fetot, px_Mg, px_Ca, px_Na, px_FoF, px_FoF_unc = c_px


    # Fudge compositions by adding Cr to the Al totals
    gt_Al += gt_Cr
    px_Al += px_Cr


    gt_Fe3 = gt_FoF * gt_Fetot
    gt_Fe2 = gt_Fetot - gt_Fe3

    px_Fe3 = px_FoF * px_Fetot
    px_Fe2 = px_Fetot - px_Fe3

    # Process garnet
    assemblage.phases[0].fitted_elements = ['Na', 'Ca', 'Fe', 'Fef', 'Mg', 'Al', 'Si']
    assemblage.phases[0].composition = np.array([gt_Na, gt_Ca, gt_Fe2, gt_Fe3, gt_Mg, gt_Al, gt_Si])
    assemblage.phases[0].compositional_uncertainties = np.identity(7)*(0.01 + assemblage.phases[0].composition*0.002) # unknown errors

    J = np.array([[1. - gt_FoF, -gt_Fetot],
                  [gt_FoF, gt_Fetot]])
    sig = np.array([[np.power(0.01 + gt_Fetot*0.002, 2.), 0.],
                    [0., np.power(gt_FoF_unc, 2.)]])
    sig_prime = J.dot(sig).dot(J.T)

    assemblage.phases[0].compositional_uncertainties[2,2] = sig_prime[0][0]
    assemblage.phases[0].compositional_uncertainties[2,3] = sig_prime[0][1]
    assemblage.phases[0].compositional_uncertainties[3,2] = sig_prime[1][0]
    assemblage.phases[0].compositional_uncertainties[3,3] = sig_prime[1][1]

    # Process pyroxene
    solutions['cpx'].fitted_elements = ['Na', 'Ca', 'Fe', 'Fef', 'Mg', 'Al', 'Si', 'Fea']
    solutions['cpx'].composition = np.array([px_Na, px_Ca, px_Fe2, px_Fe3, px_Mg, px_Al, px_Si, 0.01])
    solutions['cpx'].compositional_uncertainties = np.identity(8)*(0.01 + solutions['cpx'].composition*0.0005) # unknown errors

    J = np.array([[1. - px_FoF, -px_Fetot],
                  [px_FoF, px_Fetot]])
    sig = np.array([[np.power(0.01 + px_Fetot*0.002, 2.), 0.],
                    [0., np.power(px_FoF_unc, 2.)]])
    sig_prime = J.dot(sig).dot(J.T)

    solutions['cpx'].compositional_uncertainties[2,2] = sig_prime[0][0]
    solutions['cpx'].compositional_uncertainties[2,3] = sig_prime[0][1]
    solutions['cpx'].compositional_uncertainties[3,2] = sig_prime[1][0]
    solutions['cpx'].compositional_uncertainties[3,3] = sig_prime[1][1]


    # The following adjusts compositions to reach equilibrium
    a = burnman.Composite([solutions['cpx']])
    burnman.processanalyses.compute_and_set_phase_compositions(a)
    a.set_state(pressure, temperature)
    equilibrium_order(solutions['cpx'])
    solutions['cpx'].composition[7] = solutions['cpx'].molar_fractions[solutions['cpx'].endmember_names.index('cfs')]




    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

    assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                       assemblage.phases[k].molar_fraction_covariances)
                                      for k in range(2)]

    """
    # py, alm, gt, andr, dmaj, nagt
    # di, hed, cen, cats, jd, aeg, cfs
    print(run_id, pressure)
    print('gt', assemblage.phases[0].molar_fractions)
    print('cpx', assemblage.phases[1].molar_fractions)
    """

    Rohrbach_et_al_2007_NCFMASO_assemblages.append(assemblage)



# Change formulae back
solutions['gt'].endmember_formulae[solutions['gt'].endmember_names.index('andr')] = {'O':  12.0,
                                                           'Ca':  3.0,
                                                           'Fe': 2.0,
                                                           'Si':  3.0}

solutions['cpx'].endmember_formulae[solutions['cpx'].endmember_names.index('acm')] = {'O':  6.0,
                                                                  'Na':  1.0,
                                                                  'Fe': 1.0,
                                                                  'Si': 2.0}

solutions['cpx'].endmember_formulae[solutions['cpx'].endmember_names.index('cfs')] = {'O':  6.0,
                                                                  'Fe': 2.0,
                                                                  'Si': 2.0}
