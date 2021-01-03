import numpy as np
from fitting_functions import equilibrium_order
import burnman


def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    # Garnet-clinopyroxene partitioning data
    with open('data/Rohrbach_et_al_2007_NCFMASO_gt_cpx.dat', 'r') as f:
        expt_data = [line.split() for line in f
                     if line.split() != [] and line[0] != '#']

    set_runs = list(set([d[0] for d in expt_data]))

    Rohrbach_et_al_2007_NCFMASO_assemblages = []
    for i, run_id in enumerate(set_runs):
        # always garnet then cpx
        gt_idx, cpx_idx = [idx for idx, d in enumerate(expt_data)
                           if d[0] == run_id]

        pressure = float(expt_data[gt_idx][1]) * 1.e9  # GPa to Pa
        temperature = float(expt_data[gt_idx][2]) + 273.15  # C to K
        if pressure > 5.e9:
            assemblage = burnman.Composite([solutions['gt'],
                                            solutions['cpx'],
                                            endmembers['fcc_iron']],
                                           [0.2, 0.3, 0.5])
        else:
            assemblage = burnman.Composite([child_solutions['xmj_gt'],
                                            solutions['cpx'],
                                            endmembers['fcc_iron']],
                                           [0.2, 0.3, 0.5])

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
        gt_Fe3_unc = gt_FoF_unc * gt_Fetot

        px_Fe3 = px_FoF * px_Fetot
        px_Fe3_unc = px_FoF_unc * px_Fetot

        # Process garnet
        assemblage.phases[0].fitted_elements = ['Na', 'Ca', 'Fe', 'Fef_B', 'Mg', 'Al', 'Si']
        assemblage.phases[0].composition = np.array([gt_Na, gt_Ca,
                                                     gt_Fetot, gt_Fe3,
                                                     gt_Mg, gt_Al, gt_Si])
        assemblage.phases[0].compositional_uncertainties = 0.01 + assemblage.phases[0].composition*0.02 # unknown errors
        assemblage.phases[0].compositional_uncertainties[3] = gt_Fe3_unc

        # Process pyroxene
        solutions['cpx'].fitted_elements = ['Na', 'Ca', 'Fe', 'Fef_B', 'Mg',
                                            'Al', 'Si', 'Fe_A']
        solutions['cpx'].composition = np.array([px_Na, px_Ca,
                                                 px_Fetot, px_Fe3,
                                                 px_Mg, px_Al, px_Si, 0.01])
        solutions['cpx'].compositional_uncertainties = 0.01 + solutions['cpx'].composition*0.02 # unknown errors
        solutions['cpx'].compositional_uncertainties[3] = px_Fe3_unc

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
        print(assemblage.phases[0].fitted_elements)
        act = [float('{0:.3f}'.format(assemblage.phases[0].formula[e]))
               for e in assemblage.phases[0].fitted_elements]
        diff = act - assemblage.phases[0].composition
        print(diff)
        print(assemblage.phases[0].compositional_uncertainties)
        """
        Rohrbach_et_al_2007_NCFMASO_assemblages.append(assemblage)

    return Rohrbach_et_al_2007_NCFMASO_assemblages
