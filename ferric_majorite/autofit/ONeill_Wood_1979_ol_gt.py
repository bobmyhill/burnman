import numpy as np

from input_dataset import *


# Garnet-olivine partitioning data
ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_ol_gt_KD.dat')

ONeill_Wood_1979_assemblages = []
run_id=0
for P, T, XMgOl, lnKD, lnKDerr in ol_gt_data:
    run_id+=1
    
    # KD is (XGtFe*XOlMg)/(XGtMg*XOlFe)
    KD = np.exp(lnKD)
    XMgGt = 1./( 1. + ((1. - XMgOl)/XMgOl)*KD)
    dXMgGtdlnKD = -(1. - XMgOl)*KD/(XMgOl * np.power( (1. - XMgOl)*KD/XMgOl + 1., 2. ))
    XMgGterr = dXMgGtdlnKD*lnKDerr


    assemblage = burnman.Composite([ol, gt])
    
    assemblage.experiment_id = 'ONeill_Wood_1979_{0}'.format(run_id)
    assemblage.nominal_state = np.array([P, T])
    assemblage.state_covariances = np.array([[1.e7*1.e7, 0.], [0., 100.]])

    ol.fitted_elements = ['Mg', 'Fe']  
    ol.composition = np.array([XMgOl, 1. - XMgOl])
    ol.compositional_uncertainties = np.array([XMgGterr, XMgGterr])
    
    gt.fitted_elements = ['Mg', 'Fe']  
    gt.composition = np.array([XMgGt, 1. - XMgGt])
    gt.compositional_uncertainties = np.array([XMgGterr, XMgGterr])
        
    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)
    
    assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                       assemblage.phases[k].molar_fraction_covariances)
                                      for k in range(2)]
    
        
    ONeill_Wood_1979_assemblages.append(assemblage)
    

print(ONeill_Wood_1979_assemblages[0].phases[1].molar_fractions)
