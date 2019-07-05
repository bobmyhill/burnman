import numpy as np

from input_dataset import *
from fitting_functions import equilibrium_order

# Garnet-clinopyroxene partitioning data
with open('data/Beyer_et_al_2019_NCFMASO.dat', 'r') as f:
    ds = [line.split() for line in f if line.split() != [] and line[0] != '#']

set_runs = list(set([d[0] for d in ds]))
all_chambers = [list(set([d[2] for d in ds if d[0] == run])) for run in set_runs]

Beyer_et_al_2019_NCFMASO_assemblages = []


# Temporarily change the formulae of the endmembers for fitting purposes!!

# Garnet
gt.endmember_formulae[gt.endmember_names.index('andr')] = {'O':  12.0,
                                                           'Ca':  3.0,
                                                           'Fef': 2.0,
                                                           'Si':  3.0}
child_solutions['xna_gt'].endmember_formulae[child_solutions['xna_gt'].endmember_names.index('andr')] = {'O':  12.0, 'Ca':  3.0, 'Fef': 2.0, 'Si':  3.0}
child_solutions['xmj_gt'].endmember_formulae[child_solutions['xmj_gt'].endmember_names.index('andr')] = {'O':  12.0, 'Ca':  3.0, 'Fef': 2.0, 'Si':  3.0}

# Clinopyroxene
cpx_od.endmember_formulae[cpx_od.endmember_names.index('acm')] = {'O':  6.0,
                                                                  'Na':  1.0,
                                                                  'Fef': 1.0,
                                                                  'Si': 2.0}


# Ordered endmembers
opx_od.endmember_formulae[opx_od.endmember_names.index('ofm')] = {'O':  6.0,
                                                                  'Mga': 1.,
                                                                  'Mg':  1.0,
                                                                  'Fe': 1.0,
                                                                  'Si': 2.0}

hpx_od.endmember_formulae[hpx_od.endmember_names.index('hfm')] = {'O':  6.0,
                                                                  'Mga': 1.0,
                                                                  'Mg':  1.0,
                                                                  'Fe': 1.0,
                                                                  'Si': 2.0}

cpx_od.endmember_formulae[cpx_od.endmember_names.index('cfs')] = {'O':  6.0,
                                                                  'Fea': 1.0,
                                                                  'Fe': 2.0,
                                                                  'Si': 2.0}
    

for i, run in enumerate(set_runs):

    
    run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
    pressure = float(ds[run_indices[0]][3])*1.e9
    temperature = float(ds[run_indices[0]][4])+273.15

    for j, chamber in enumerate(all_chambers[i]):
        chamber_indices = [run_idx for run_idx in run_indices
                           if (ds[run_idx][2] == chamber and
                               ds[run_idx][1] != 'unwanted phases')] # include all phases for now.
      
        if len(chamber_indices) > 1 and (ds[run_indices[0]][5] == 'Fe' or ds[run_indices[0]][5] == 'Mo'): # No Re runs (maybe not at eqm?)
                phases = []
                for idx in chamber_indices:
                    
                    if ds[idx][1] == 'ring?':
                        ds[idx][1] = 'ring'

                        
                    if ds[idx][1] == 'gt':
                        c = np.array([float(ds[idx][cidx]) for cidx in [7,9,11,15,17,19]])
                        if c[-1] < 0.05: # Na < 0.05
                            phases.append(child_solutions['xna_gt'])
                        elif c[-1] > c[0] - 3. - 0.05: # Na > (Si - 3) - 0.05
                            phases.append(child_solutions['xmj_gt'])
                        else:
                            phases.append(gt)
                    elif (ds[idx][1] == 'cpx' and
                          (ds[run_indices[0]][5] == 'Fe' or ds[run_indices[0]][5] == 'Mo')): # only include cpx for Fe-saturated runs
                        phases.append(cpx_od)
                    else:
                        try:
                            phases.append(endmembers[ds[idx][1]])
                        except:
                            try:
                                phases.append(solutions[ds[idx][1]])
                            except:
                                phases.append(child_solutions[ds[idx][1]])

                assemblage = burnman.Composite(phases)
                
                assemblage.experiment_id = 'Beyer2019_{0}'.format(run)
                assemblage.nominal_state = np.array([pressure, temperature])
                assemblage.state_covariances = np.array([[1.e7*1.e7, 0.],[0., 50.*50]]) # pressure uncertainties v small. We'll add dP hyperparameters to deal with the multichamber nature of the experiments.

                
                for k, idx in enumerate(chamber_indices):
                    
                    # Si, Al, FeT, [Mo], Mg, Ca, Na, [Re, no unc]
                    c = np.array([float(ds[idx][cidx]) for cidx in [7,9,11,15,17,19]])

                    sig_c = np.array([max(float(ds[idx][cidx]), 0.001)
                                      for cidx in [8,10,12,16,18,20]])
                    
                    cation_total = float(ds[idx][22])
                    f = float(ds[idx][23]) # Fe3+/sumFe
                    sig_f = float(ds[idx][24]) # sigma Fe3+/sumFe
                    
                    # Solution phases in this dataset are:
                    # fper, ol, wad, ring (Fe-Mg exchange only)
                    # opx, hpx (CFMASO)
                    # cpx, gt (NCFMASO)
                    if (assemblage.phases[k] is fper or
                        assemblage.phases[k] is ol or
                        assemblage.phases[k] is wad or
                        assemblage.phases[k] is child_solutions['ring']):
                
                        assemblage.phases[k].fitted_elements = ['Fe', 'Mg']
                        assemblage.phases[k].composition = c[2:4]
                        assemblage.phases[k].compositional_uncertainties = sig_c[2:4]
                        
                    elif (assemblage.phases[k] is opx_od or
                          assemblage.phases[k] is hpx_od): # CFMAS
                        
                        assemblage.phases[k].fitted_elements = ['Si', 'Al', 'Fe', 'Mg', 'Ca', 'Mga']
                        assemblage.phases[k].composition = np.zeros(6)
                        assemblage.phases[k].composition[0:5] = c[0:5]

                        assemblage.phases[k].compositional_uncertainties = np.zeros(6)
                        assemblage.phases[k].compositional_uncertainties[0:5] = sig_c[0:5]
                        assemblage.phases[k].compositional_uncertainties[5] = 0.1 # large uncertainty for Mg on A

                        
                        # The following adjusts compositions to reach equilibrium
                        a = burnman.Composite([assemblage.phases[k]])
                        burnman.processanalyses.compute_and_set_phase_compositions(a)
                        a.set_state(pressure, temperature)
                        equilibrium_order(assemblage.phases[k])

                        if assemblage.phases[k] is opx_od:
                            opx_od.composition[5] = opx_od.molar_fractions[opx_od.endmember_names.index('ofm')]
                        if assemblage.phases[k] is hpx_od:
                            hpx_od.composition[5] = hpx_od.molar_fractions[hpx_od.endmember_names.index('hfm')]

                    elif (assemblage.phases[k] is cpx_od): # NCFMASO
                        
                        assemblage.phases[k].fitted_elements = ['Si', 'Al', 'Fe',
                                                                'Mg', 'Ca', 'Na', 'Fef', 'Fea']
                        
                        cpx_od.composition = np.zeros(8)
                        cpx_od.composition[0:6] = c
                        
                        # Fudge Fe3+ and sigma for now
                        print('WARNING! Fudging Fe3+ (10%) in cpx in Fe-saturated runs')
                        f = 0.1
                        sig_f = 0.05 
                        cpx_od.composition[2] = c[2]*(1. - f) # Fe2+
                        cpx_od.composition[6] = c[2]*f # Fe3+
                        cpx_od.composition[7] = 0.1 # large uncertainty for Fe on A
                        
                        # Uncertainties
                        cpx_od.compositional_uncertainties = np.zeros((8, 8))
                        for s_idx, sig in enumerate(sig_c):
                            cpx_od.compositional_uncertainties[s_idx, s_idx] = np.max([sig*sig, 1.e-4])
                        
                        cpx_od.compositional_uncertainties[7,7] = 1.e-4
                            
                        # We want to convert FeT and Fe3+/sum(Fe) uncertainties into
                        # Fe2+ and Fe3+ uncertainties.
                        # Starting with the equations for the amounts of Fe2+ and Fe3+:
                        # Fe2+ = (1 - f)*FeT
                        # Fe3+ = f*FeT
                        # we can build the Jacobian and transform the uncertainties:
                        J = np.array([[1.-f, -c[2]],
                                      [f,     c[2]]])
                        
                        sig = np.array([[np.power(sig_c[2], 2.), 0.],
                                        [0., np.power(sig_f, 2.)]])
                        sig_prime = J.dot(sig).dot(J.T)
                        
                        cpx_od.compositional_uncertainties[2,2] = sig_prime[0][0]
                        cpx_od.compositional_uncertainties[2,6] = sig_prime[0][1]
                        cpx_od.compositional_uncertainties[6,2] = sig_prime[1][0]
                        cpx_od.compositional_uncertainties[6,6] = sig_prime[1][1]

                        # The following adjusts compositions to reach equilibrium
                        a = burnman.Composite([cpx_od])
                        burnman.processanalyses.compute_and_set_phase_compositions(a)
                        a.set_state(pressure, temperature)
                        equilibrium_order(cpx_od)
                        cpx_od.composition[7] = cpx_od.molar_fractions[cpx_od.endmember_names.index('cfs')]
                        
                    elif assemblage.phases[k] is gt:
                        gt.fitted_elements = ['Si', 'Al', 'Fe', 'Mg',
                                              'Ca', 'Na', 'Fef']

                        # Composition
                        gt.composition = np.zeros(7)
                        gt.composition[:6] = np.copy(c)
                        gt.composition[2] = c[2]*(1. - f) # Fe2+
                        gt.composition[6] = c[2]*f # Fe3+

                        # Uncertainties
                        gt.compositional_uncertainties = np.zeros((7, 7))
                        for s_idx, sig in enumerate(sig_c):
                            gt.compositional_uncertainties[s_idx, s_idx] = np.max([sig*sig, 1.e-4])
                            
                        # We want to convert FeT and Fe3+/sum(Fe) uncertainties into
                        # Fe2+ and Fe3+ uncertainties.
                        # Starting with the equations for the amounts of Fe2+ and Fe3+:
                        # Fe2+ = (1 - f)*FeT
                        # Fe3+ = f*FeT
                        # we can build the Jacobian and transform the uncertainties:
                        J = np.array([[1.-f, -c[2]],
                                      [f,     c[2]]])
                        sig = np.array([[np.power(sig_c[2], 2.), 0.],
                                        [0., np.power(sig_f, 2.)]])
                        sig_prime = J.dot(sig).dot(J.T)
                        
                        gt.compositional_uncertainties[2,2] = sig_prime[0][0]
                        gt.compositional_uncertainties[2,6] = sig_prime[0][1]
                        gt.compositional_uncertainties[6,2] = sig_prime[1][0]
                        gt.compositional_uncertainties[6,6] = sig_prime[1][1]

                    elif (assemblage.phases[k] is child_solutions['xna_gt']  or
                          assemblage.phases[k] is child_solutions['xmj_gt']):
                        
                        assemblage.phases[k].fitted_elements = ['Si', 'Al', 'Fe', 'Mg',
                                                                'Ca', 'Fef']

                        # Composition
                        assemblage.phases[k].composition = np.zeros(6)
                        assemblage.phases[k].composition[:5] = c[0:5]
                        assemblage.phases[k].composition[2] = c[2]*(1. - f) # Fe2+
                        assemblage.phases[k].composition[5] = c[2]*f # Fe3+
                        
                        # Uncertainties
                        assemblage.phases[k].compositional_uncertainties = np.zeros((6, 6))
                        for s_idx, sig in enumerate(sig_c[0:5]):
                            assemblage.phases[k].compositional_uncertainties[s_idx, s_idx] = np.max([sig*sig, 1.e-4])
                            
                        # We want to convert FeT and Fe3+/sum(Fe) uncertainties into
                        # Fe2+ and Fe3+ uncertainties.
                        # Starting with the equations for the amounts of Fe2+ and Fe3+:
                        # Fe2+ = (1 - f)*FeT
                        # Fe3+ = f*FeT
                        # we can build the Jacobian and transform the uncertainties:
                        J = np.array([[1.-f, -c[2]],
                                      [f,     c[2]]])
                        sig = np.array([[np.power(sig_c[2], 2.), 0.],
                                        [0., np.power(sig_f, 2.)]])
                        sig_prime = J.dot(sig).dot(J.T)
                        
                        assemblage.phases[k].compositional_uncertainties[2,2] = sig_prime[0][0]
                        assemblage.phases[k].compositional_uncertainties[2,5] = sig_prime[0][1]
                        assemblage.phases[k].compositional_uncertainties[5,2] = sig_prime[1][0]
                        assemblage.phases[k].compositional_uncertainties[5,5] = sig_prime[1][1]
                    elif isinstance(assemblage.phases[k], burnman.Mineral):
                        pass
                    else:
                        raise Exception('phase not recognised: {0}'.format(assemblage.phases[k].name))

                        

                burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

                """
                if child_solutions['xna_gt'] in assemblage.phases:
                    print(run, chamber, child_solutions['xna_gt'].molar_fractions)
                if child_solutions['xmj_gt'] in assemblage.phases:
                    print(run, chamber, child_solutions['xmj_gt'].molar_fractions) 
                if gt in assemblage.phases:
                    print(run, chamber, gt.molar_fractions)
                
                if cpx_od in assemblage.phases:
                    print(run, chamber, cpx_od.molar_fractions)
                """
                
                assemblage.stored_compositions = ['composition not assigned']*len(chamber_indices)
                for k in range(len(chamber_indices)):
                    try:
                        assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                             assemblage.phases[k].molar_fraction_covariances)
                    except:
                        pass

                Beyer_et_al_2019_NCFMASO_assemblages.append(assemblage)



# Change formulae back

# Garnet
gt.endmember_formulae[gt.endmember_names.index('andr')] = {'O':  12.0,
                                                           'Ca': 3.0,
                                                           'Fe': 2.0,
                                                           'Si': 3.0}
child_solutions['xna_gt'].endmember_formulae[child_solutions['xna_gt'].endmember_names.index('andr')] = {'O': 12.0, 'Ca': 3.0, 'Fe': 2.0, 'Si': 3.0}
child_solutions['xmj_gt'].endmember_formulae[child_solutions['xmj_gt'].endmember_names.index('andr')] = {'O': 12.0, 'Ca': 3.0, 'Fe': 2.0, 'Si': 3.0}

# Clinopyroxene
cpx_od.endmember_formulae[cpx_od.endmember_names.index('acm')] = {'O':  6.0,
                                                                  'Na': 1.0,
                                                                  'Fe': 1.0,
                                                                  'Si': 2.0}

    
# Ordered endmembers
opx_od.endmember_formulae[opx_od.endmember_names.index('ofm')] = {'O':  6.0,
                                                                  'Mg':  1.0,
                                                                  'Fe': 1.0,
                                                                  'Si': 2.0}

hpx_od.endmember_formulae[hpx_od.endmember_names.index('hfm')] = {'O':  6.0,
                                                                  'Mg': 1.0,
                                                                  'Fe': 1.0,
                                                                  'Si': 2.0}

cpx_od.endmember_formulae[cpx_od.endmember_names.index('cfs')] = {'O':  6.0,
                                                                  'Fe': 2.0,
                                                                  'Si': 2.0}
