import numpy as np

from input_dataset import *


# Garnet-pyroxene partitioning data
with open('data/Woodland_ONeill_1993_FASO_alm_sk_analyses.dat', 'r') as f:
    expt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

set_runs = list(set([d[0] for d in expt_data]))

print('WARNING: o-d required to model fets component in FeSiO3 pyroxene. Opx ignored for now')
Woodland_ONeill_1993_FASO_assemblages = []
for i, run in enumerate(set_runs):
    
    run_indices = [idx for idx, d in enumerate(expt_data) if d[0] == run]
    n_phases = len(run_indices)
    phase_names = [expt_data[idx][3] for idx in run_indices]
    phases = []
    for datum in [expt_data[idx] for idx in run_indices]:
        run_id = datum[0]
        PGPa, TK = map(float, datum[1:3])
        phase = datum[3]
        c = map(float, datum[4:])

        # Temporarily change the formulae of the endmembers for fitting purposes!!
        child_solutions['alm_sk_gt'].endmembers[1][0].params['formula'] = {'O':  12.0,
                                                                           'Fe':  3.0,
                                                                           'Fef': 2.0,
                                                                           'Si':  3.0}
        child_solutions['herc_mt_frw'].endmembers[1][0].params['formula'] = {'O':  4.0,
                                                                             'Fe':  1.0,
                                                                             'Fef': 2.0}
        
        if phase == 'qtz':
            phases.append(qtz)
        elif phase == 'coe':
            phases.append(coe)
        elif phase == 'stv':
            phases.append(stv)
        elif phase == 'iron':
            phases.append(fcc_iron) # all experiments in fcc stability field
        elif phase == 'hem':
            phases.append(hem) 
        elif phase == 'fa':
            phases.append(fa) 
        elif phase == 'opx':
            pass
            #phases.append(ofs) # actually a mix of ofs and fets, but not possible in this model (yet)
        elif phase == 'sp':
            if c[0] < 0.005:
                phases.append(child_solutions['mt_frw']) 
                child_solutions['mt_frw'].fitted_elements = ['Fe', 'Fef', 'Si']
                child_solutions['mt_frw'].composition = np.array([c[1], c[2], c[3]])
                child_solutions['mt_frw'].compositional_uncertainties = np.array([0.01]*3)
            else:
                phases.append(child_solutions['herc_mt_frw']) 
                child_solutions['herc_mt_frw'].fitted_elements = ['Al', 'Fe', 'Fef', 'Si']
                child_solutions['herc_mt_frw'].composition = np.array([c[0], c[1], c[2], c[3]])
                child_solutions['herc_mt_frw'].compositional_uncertainties = np.array([0.01]*4)
        elif phase == 'gt':
            if c[0] < 0.005:
                phases.append(child_solutions['sk_gt'])
            else:
                phases.append(child_solutions['alm_sk_gt'])
                child_solutions['alm_sk_gt'].fitted_elements = ['Al', 'Fe', 'Fef', 'Si']
                child_solutions['alm_sk_gt'].composition = np.array([c[0], c[1], c[2], c[3]])
                child_solutions['alm_sk_gt'].compositional_uncertainties = np.array([0.01]*4)
        else:
            raise Exception('{0} not recognised'.format(phase))

    
    # pick only experiments where one or more reactions are constrained
    if (('sp' in phase_names) or
        ('hem' in phase_names)):
        assemblage = burnman.Composite(phases)
        assemblage.experiment_id = 'Woodland_ONeill_1993_FASO_{0}'.format(run_id)
        assemblage.nominal_state = np.array([PGPa*1.e9, TK]) # CONVERT P TO Pa
        assemblage.state_covariances = np.array([[5.e7*5.e7, 0.], [0., 100.]])
    
        burnman.processanalyses.compute_and_set_phase_compositions(assemblage)
        
        assemblage.stored_compositions = ['composition not assigned']*n_phases
        
        for k in range(n_phases):
            try:
                assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                     assemblage.phases[k].molar_fraction_covariances)
            except:
                pass # some phases are endmembers

        # Change the formulae of the endmembers back
        child_solutions['alm_sk_gt'].endmembers[1][0].params['formula'] = {'O':  12.0,
                                                                           'Fe':  5.0,
                                                                           'Si':  3.0}
        child_solutions['herc_mt_frw'].endmembers[1][0].params['formula'] = {'O':  4.0,
                                                                             'Fe':  3.0}

        Woodland_ONeill_1993_FASO_assemblages.append(assemblage)
    
