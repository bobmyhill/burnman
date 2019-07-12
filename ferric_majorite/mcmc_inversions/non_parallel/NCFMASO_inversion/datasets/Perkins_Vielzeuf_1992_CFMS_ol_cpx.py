import numpy as np

from input_dataset import *

print('Warning: Perkins and Vielzeuf cpx could have Fe3+. Uncertainties guessed')
print('N.B. Only using ol-cpx equilibria with Ca in ol < 0.02 and Ca in opx > 0.475 (near di-hed join)')

# Garnet-pyroxene partitioning data
cpx_ol_data = np.loadtxt('data/Perkins_Vielzeuf_1992_CFMS_ol_cpx.dat')

Perkins_Vielzeuf_1992_CFMS_assemblages = []

for run_id, (PGPa, TK, Cacpx, Mgcpx, Fecpx, Caol, Mgol, Feol) in enumerate(cpx_ol_data):


    if Cacpx > 0.475 and Caol < 0.02:
        assemblage = burnman.Composite([child_solutions['di_hed'], solutions['ol']])
        assemblage.experiment_id = 'Perkins_Vielzeuf_1992_CFMS_{0}'.format(run_id)
        assemblage.nominal_state = np.array([PGPa*1.e9, TK]) # CONVERT P TO PA
        assemblage.state_covariances = np.array([[5.e8*5.e8, 0.], [0., 100.]])

        solutions['ol'].fitted_elements = ['Mg', 'Fe']
        solutions['ol'].composition = np.array([Mgol, Feol])
        solutions['ol'].compositional_uncertainties = np.array([0.01, 0.01])

        child_solutions['di_hed'].fitted_elements = ['Mg', 'Fe']
        child_solutions['di_hed'].composition = np.array([Mgcpx, Fecpx])
        child_solutions['di_hed'].compositional_uncertainties = np.array([0.01, 0.01])

        burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

        assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                           assemblage.phases[k].molar_fraction_covariances)
                                          for k in range(2)]

        Perkins_Vielzeuf_1992_CFMS_assemblages.append(assemblage)
