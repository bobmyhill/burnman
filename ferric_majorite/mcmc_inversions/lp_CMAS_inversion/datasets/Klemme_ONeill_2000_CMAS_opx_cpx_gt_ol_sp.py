import numpy as np
import burnman
from burnman.processanalyses import compute_and_set_phase_compositions


def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    child_solutions = mineral_dataset['child_solutions']

    # Garnet-pyroxene partitioning data
    with open('data/Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp.dat', 'r') as f:
        expt_data = [line.split() for line in f if line.split() != []
                     and line[0] != '#']

    set_runs = list(set([d[1] for d in expt_data]))

    Klemme_ONeill_2000_CMAS_assemblages = []

    for i, run in enumerate(set_runs):

        run_indices = [idx for idx, d in enumerate(expt_data) if d[1] == run]
        n_phases = len(run_indices)
        phases = []
        for datum in [expt_data[idx] for idx in run_indices]:
            master_id, run_id = datum[0:2]
            TC, Pkbar, t = list(map(float, datum[2:5]))
            phase = datum[5]

            # composition = Si	Sierr	Al	Alerr	Mg	Mgerr	Ca	Caerr
            cs = list(map(float, datum[6:]))

            if phase == 'fo':
                phases.append(endmembers['fo'])
            elif phase == 'sp':
                phases.append(endmembers['sp'])
            elif phase == 'opx':
                phases.append(child_solutions['oen_mgts_odi'])
            elif phase == 'cpx':
                phases.append(child_solutions['di_cen_cats'])
            elif phase == 'gt':
                phases.append(child_solutions['py_gr_gt'])
            else:
                raise Exception('phase not recognised')

            if type(phases[-1]) is burnman.SolidSolution:
                c = np.array([cs[0], cs[2], cs[4], cs[6]])
                sig_c = np.array([cs[1], cs[3], cs[5], cs[7]])
                phases[-1].fitted_elements = ['Si', 'Al', 'Mg', 'Ca']
                phases[-1].composition = c
                phases[-1].compositional_uncertainties = np.array([max([0.01, sig_c[i], 0.02*c[i]])
                                                                   for i in
                                                                   range(4)])
                # compositional uncertainties max of the reported uncertainty,
                # 2 % of the reported value and 0.01

        assemblage = burnman.Composite(phases)
        assemblage.experiment_id = 'Klemme_ONeill_2000_CMAS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([Pkbar*1.e8,
                                             TC+273.15])  # P TO Pa, T to K
        assemblage.state_covariances = np.array([[1.e8*1.e8, 0.], [0., 100.]])

        compute_and_set_phase_compositions(assemblage)

        assemblage.stored_compositions = ['composition not assigned']*n_phases
        for k in range(n_phases):
            try:
                assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                     assemblage.phases[k].molar_fraction_covariances)
            except AttributeError:  # fo and sp are endmembers
                pass

        Klemme_ONeill_2000_CMAS_assemblages.append(assemblage)

    return Klemme_ONeill_2000_CMAS_assemblages
