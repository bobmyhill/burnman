import numpy as np

from input_dataset import *


# Garnet-clinopyroxene partitioning data
with open('data/Rohrbach_et_al_2007_NCFMASO_gt_cpx.dat', 'r') as f:
    expt_data = [line.split() for line in f if line.split() != [] and line[0] != '#']

set_runs = ['exp_15', 'zuII_4', 'zuIII_1', 'zu_8', 'zuII_3', 'zuIV_1']
#set_runs = ['zuIII_1', 'zuII_3']

print('WARNING: NOT READY YET!!')

# Temporarily change the formulae of the endmembers for fitting purposes!!
gt.endmember_formulae[gt.endmember_names.index('andr')] = {'O':  12.0,
                                                           'Ca':  3.0,
                                                           'Fef': 2.0,
                                                           'Si':  3.0}
cpx_od.endmember_formulae[cpx_od.endmember_names.index('acm')] = {'O':  6.0,
                                                                  'Na':  1.0,
                                                                  'Fef': 1.0,
                                                                  'Si': 2.0}


Rohrbach_et_al_2007_NCFMASO_assemblages = []
for i, run_id in enumerate(set_runs):
    # always garnet then cpx
    gt_idx, cpx_idx = [idx for idx, d in enumerate(expt_data) if d[0] == run_id]
    
    assemblage = burnman.Composite([gt, cpx_od, fcc_iron])
    assemblage.experiment_id = 'Rohrbach_et_al_2007_NCFMASO_{0}'.format(run_id)
    assemblage.nominal_state = np.array([float(expt_data[gt_idx][1]) * 1.e9, # GPa to Pa
                                         float(expt_data[gt_idx][2]) + 273.15]) # C to K
    assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

    c_gt = np.array(map(float, expt_data[gt_idx][5:]))
    c_px = np.array(map(float, expt_data[cpx_idx][5:]))

    gt_Si, gt_Ti, gt_Al, gt_Ca, gt_Fetot, gt_Mg, gt_Ca, gt_Na, gt_FoF, gt_FoF_unc = c_gt
    px_Si, px_Ti, px_Al, px_Ca, px_Fetot, px_Mg, px_Ca, px_Na, px_FoF, px_FoF_unc = c_px
    
    gt_Fe3 = gt_FoF * gt_Fetot
    gt_Fe2 = gt_Fetot - gt_Fe3
    
    px_Fe3 = px_FoF * px_Fetot
    px_Fe2 = px_Fetot - px_Fe3

    # Process garnet
    gt.fitted_elements = ['Na', 'Ca', 'Fe', 'Fef', 'Mg', 'Al', 'Si']
    gt.composition = np.array([gt_Na, gt_Ca, gt_Fe2, gt_Fe3, gt_Mg, gt_Al, gt_Si])
    gt.compositional_uncertainties = np.identity(7)*(0.01 + gt.composition*0.002) # unknown errors

    J = np.array([[1. - gt_FoF, -gt_Fetot],
                  [gt_FoF, gt_Fetot]])
    sig = np.array([[np.power(0.01 + gt_Fetot*0.002, 2.), 0.],
                    [0., np.power(gt_FoF_unc, 2.)]])
    sig_prime = J.dot(sig).dot(J.T)

    gt.compositional_uncertainties[2,2] = sig_prime[0][0]
    gt.compositional_uncertainties[2,3] = sig_prime[0][1]
    gt.compositional_uncertainties[3,2] = sig_prime[1][0]
    gt.compositional_uncertainties[3,3] = sig_prime[1][1]
    
    # Process pyroxene
    cpx_od.fitted_elements = ['Na', 'Ca', 'Fe', 'Fef', 'Mg', 'Al', 'Si']
    cpx_od.composition = np.array([px_Na, px_Ca, px_Fe2, px_Fe3, px_Mg, px_Al, px_Si])
    cpx_od.compositional_uncertainties = np.identity(7)*(0.01 + cpx_od.composition*0.0005) # unknown errors

    J = np.array([[1. - px_FoF, -px_Fetot],
                  [px_FoF, px_Fetot]])
    sig = np.array([[np.power(0.01 + px_Fetot*0.002, 2.), 0.],
                    [0., np.power(px_FoF_unc, 2.)]])
    sig_prime = J.dot(sig).dot(J.T)

    cpx_od.compositional_uncertainties[2,2] = sig_prime[0][0]
    cpx_od.compositional_uncertainties[2,3] = sig_prime[0][1]
    cpx_od.compositional_uncertainties[3,2] = sig_prime[1][0]
    cpx_od.compositional_uncertainties[3,3] = sig_prime[1][1]
    
    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

    # py, alm, gt, andr, dmaj, nagt
    # di, hed, cen, cats, jd, aeg, cfs
    print(run_id)
    print(gt.composition)
    print(cpx_od.composition)
    assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                       assemblage.phases[k].molar_fraction_covariances)
                                      for k in range(2)]
    print('gt', assemblage.phases[0].molar_fractions)
    print('cpx', assemblage.phases[1].molar_fractions)
        
    Rohrbach_et_al_2007_NCFMASO_assemblages.append(assemblage)
    

    
# Change formulae back
gt.endmember_formulae[gt.endmember_names.index('andr')] = {'O':  12.0,
                                                           'Ca':  3.0,
                                                           'Fe': 2.0,
                                                           'Si':  3.0}

cpx_od.endmember_formulae[cpx_od.endmember_names.index('acm')] = {'O':  6.0,
                                                                  'Na':  1.0,
                                                                  'Fe': 1.0,
                                                                  'Si': 2.0}

    
                
