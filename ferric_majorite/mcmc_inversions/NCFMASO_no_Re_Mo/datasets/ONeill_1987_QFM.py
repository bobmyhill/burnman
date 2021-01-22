import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    def mu_O2_Cu_Cu2O(T):
        if T < 750.:
            raise Exception('T too low')
        elif T < 1330.:
            return -347705. + 246.096*T - 12.9053*T*np.log(T)
        else:
            raise Exception('T too high')


    ONeill_1987_QFM_assemblages = []

    F = 96484.56 # value of Faraday constant from paper

    data = np.loadtxt('data/ONeill_1987_QFM_CuCu2O_electrode.dat')
    for i, (T, emfmV) in enumerate(data):
        if T > 1000.:
            emf = emfmV*1.e-3
            mu_O2_ref = mu_O2_Cu_Cu2O(T)
            mu_O2 = mu_O2_ref - 4.*F*emf # 4FE = mu_O2B - mu_O2A; reference electrode is electrode B

            assemblage = burnman.Composite([endmembers['qtz'], endmembers['fa'], endmembers['mt'],
                                            burnman.CombinedMineral([endmembers['O2']], [1.],
                                                                    [mu_O2, 0., 0.])])

            assemblage.experiment_id = 'QFM_CuCu2O_electrode'
            assemblage.nominal_state = np.array([1.e5, T])
            assemblage.state_covariances = np.array([[1., 0.],
                                                     [0., 100.]]) # 10 K uncertainty - this is actually a proxy for the uncertainty in the emf.


            ONeill_1987_QFM_assemblages.append(assemblage)

    return ONeill_1987_QFM_assemblages