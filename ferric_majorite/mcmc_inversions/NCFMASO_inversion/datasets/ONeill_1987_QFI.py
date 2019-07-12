import numpy as np
import burnman

def get_assemblages(mineral_dataset):
    endmembers = mineral_dataset['endmembers']
    solutions = mineral_dataset['solutions']
    child_solutions = mineral_dataset['child_solutions']

    def mu_O2_Fe_FeO(T):
        if T < 833.:
            raise Exception('T too low')
        elif T < 1042.:
            return -605812. + 1366.718*T - 182.7955*T*np.log(T) + 0.103592*T*T
        elif T < 1184.:
            return -519357. + 59.427*T + 8.9276*T*np.log(T)
        elif T < 1450.:
            return -551159. + 269.404*T - 16.9484*T*np.log(T)
        else:
            raise Exception('T too high')

    def mu_O2_Mo_MoO2(T):
        if T < 1000.:
            raise Exception('T too low')
        elif T < 1450.:
            return -603268. + 337.460*T - 20.6892*T*np.log(T)
        else:
            raise Exception('T too high')


    ONeill_1987_QFI_assemblages = []

    F = 96484.56 # value of Faraday constant from paper

    data = np.loadtxt('data/ONeill_1987_QFI_FeFeO_electrode.dat')
    for i, (T, emfmV) in enumerate(data):
        if T > 1000.:
            emf = emfmV*1.e-3
            mu_O2_ref = mu_O2_Fe_FeO(T)
            mu_O2 = mu_O2_ref - 4.*F*emf # 4FE = mu_O2B - mu_O2A; reference electrode is electrode B

            if T < 1184:
                iron = endmembers['bcc_iron']
            else:
                iron = endmembers['fcc_iron']

            assemblage = burnman.Composite([endmembers['fa'],
                                            iron,
                                            endmembers['qtz'],
                                            burnman.CombinedMineral([endmembers['O2']], [1.],
                                                                    [mu_O2, 0., 0.])])
            assemblage.experiment_id = 'QFI_FeFeO_electrode'

            assemblage.nominal_state = np.array([1.e5, T])
            assemblage.state_covariances = np.array([[1., 0.],
                                                     [0., 100.]]) # 10 K uncertainty - this is actually a proxy for the uncertainty in the emf.


            ONeill_1987_QFI_assemblages.append(assemblage)


    data = np.loadtxt('data/ONeill_1987_QFI_MoMoO2_electrode.dat')
    for i, (T, emfmV) in enumerate(data):
        if T > 1000.:
            emf = emfmV*1.e-3
            mu_O2_ref = mu_O2_Mo_MoO2(T)
            mu_O2 = mu_O2_ref - 4.*F*emf # 4FE = mu_O2B - mu_O2A; reference electrode is electrode B

            if T < 1184:
                iron = endmembers['bcc_iron']
            else:
                iron = endmembers['fcc_iron']

            assemblage = burnman.Composite([endmembers['fa'], iron,
                                            endmembers['qtz'],
                                            burnman.CombinedMineral([endmembers['O2']], [1.],
                                                                    [mu_O2, 0., 0.])])

            assemblage.experiment_id = 'QFI_MoMoO2_electrode'
            assemblage.nominal_state = np.array([1.e5, T])
            assemblage.state_covariances = np.array([[1., 0.],
                                                     [0., 100.]]) # 10 K uncertainty - this is actually a proxy for the uncertainty in the emf.


            ONeill_1987_QFI_assemblages.append(assemblage)

    return ONeill_1987_QFI_assemblages
