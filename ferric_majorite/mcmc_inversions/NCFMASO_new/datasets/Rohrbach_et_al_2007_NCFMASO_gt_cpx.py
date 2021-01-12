import numpy as np

from burnman.processanalyses import store_composition
from burnman.processanalyses import AnalysedComposite
from burnman.processanalyses import compute_and_store_phase_compositions
from global_constants import midpoint_proportion
from global_constants import constrain_endmembers
from global_constants import proportion_cutoff


def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']

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

        # All compositions in equilibrium with metallic iron
        assemblage = AnalysedComposite([solutions['gt'],
                                        solutions['cpx'],
                                        endmembers['fcc_iron']])

        assemblage.experiment_id = 'Rohrbach_et_al_2007_NCFMASO_{0}'.format(run_id)
        assemblage.nominal_state = np.array([pressure, temperature])
        assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

        c_gt = np.array(list(map(float, expt_data[gt_idx][5:])))
        c_px = np.array(list(map(float, expt_data[cpx_idx][5:])))

        gt_Si, gt_Ti, gt_Al, gt_Cr, gt_Fetot = c_gt[:5]
        gt_Mg, gt_Ca, gt_Na, gt_FoF, gt_FoF_unc = c_gt[5:]
        px_Si, px_Ti, px_Al, px_Cr, px_Fetot = c_px[:5]
        px_Mg, px_Ca, px_Na, px_FoF, px_FoF_unc = c_px[5:]

        # Fudge compositions by adding Cr to the Al totals
        gt_Al += gt_Cr
        px_Al += px_Cr

        # Calculate variance-covariance matrices
        # sadly,no uncertainties are reported for the Rohrbach compositions

        # Garnet
        gt_Fe_unc = 0.01 + gt_Fetot * 0.02
        gt_Fef = gt_FoF * gt_Fetot

        gt_J = np.array([[1., 0.],
                         [gt_FoF, gt_Fetot]])
        gt_covFe = np.array([[gt_Fe_unc*gt_Fe_unc, 0.],
                             [0., gt_FoF_unc * gt_FoF_unc]])
        gt_covFe = np.einsum('ij, jk, lk', gt_J, gt_covFe, gt_J)

        fitted_elements = ['Na', 'Ca', 'Mg', 'Al', 'Si', 'Fe', 'Fef_B']
        gt_composition = [gt_Na, gt_Ca, gt_Mg, gt_Al, gt_Si, gt_Fetot, gt_Fef]
        gt_cov = np.zeros((7, 7))
        gt_cov[:5, :5] = np.diag([np.max([1.e-10, 0.01 + 0.02 * u]) for u in
                                  [gt_Na, gt_Ca, gt_Mg, gt_Al, gt_Si]])
        gt_cov[5:, 5:] = gt_covFe

        # Clinopyroxene
        px_Fe_unc = 0.01 + px_Fetot * 0.02
        px_Fef = px_FoF * px_Fetot

        px_J = np.array([[1., 0.],
                         [px_FoF, px_Fetot]])
        px_covFe = np.array([[px_Fe_unc*px_Fe_unc, 0.],
                             [0., px_FoF_unc * px_FoF_unc]])
        px_covFe = np.einsum('ij, jk, lk', px_J, px_covFe, px_J)

        fitted_elements = ['Na', 'Ca', 'Mg', 'Al', 'Si', 'Fe', 'Fef_B']
        px_composition = [px_Na, px_Ca, px_Mg, px_Al, px_Si, px_Fetot, px_Fef]
        px_cov = np.zeros((7, 7))
        px_cov[:5, :5] = np.diag([np.max([1.e-10, 0.01 + 0.02 * u]) for u in
                                  [px_Na, px_Ca, px_Mg, px_Al, px_Si]])
        px_cov[5:, 5:] = px_covFe

        store_composition(solutions['gt'], fitted_elements,
                          gt_composition, gt_cov)
        store_composition(solutions['cpx'], fitted_elements,
                          px_composition, px_cov)

        # Tweak compositions with 0.1% of a midpoint proportion
        # Do not consider (transformed) endmembers with < 5% abundance
        # in the solid solution. Copy the stored compositions from
        # each phase to the assemblage storage.
        assemblage.set_state(*assemblage.nominal_state)
        compute_and_store_phase_compositions(assemblage,
                                             midpoint_proportion,
                                             constrain_endmembers,
                                             proportion_cutoff,
                                             copy_storage=True)

        Rohrbach_et_al_2007_NCFMASO_assemblages.append(assemblage)

    return Rohrbach_et_al_2007_NCFMASO_assemblages
