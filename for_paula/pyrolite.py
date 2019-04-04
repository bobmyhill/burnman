import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import burnman
from burnman import equilibrate
from burnman.minerals import SLB_2011

bdg = SLB_2011.mg_fe_perovskite()
fper = SLB_2011.ferropericlase()
cpv = SLB_2011.ca_perovskite()
cf = SLB_2011.ca_ferrite_structured_phase()

bdg.guess = np.array([0.9, 0.05, 0.05])
fper.guess = np.array([0.9, 0.1])
cf.guess = np.array([0.9, 0.1, 0.0])

pressures = np.linspace(40.e9, 100.e9, 61)
temperatures = np.linspace(1500., 2500., 21)

xSiO2 = 39.4
xAl2O3 = 2.0
xCaO = 3.3
xMgO = 49.5
xFeO = 5.2
xNa2O = 0.26

composition = {'Na': xNa2O*2., 'Fe': xFeO, 'Mg': xMgO, 'Si': xSiO2, 'Ca': xCaO, 'Al': xAl2O3*2.,
               'O': xSiO2*2. + xAl2O3*3. + xCaO + xMgO + xFeO + xNa2O}
assemblage = burnman.Composite([bdg, fper, cpv, cf])
equality_constraints = [('T', temperatures), ('P', pressures)]
sols, prm = equilibrate(composition, assemblage, equality_constraints, store_iterates=False, store_assemblage=True)

header = '# '
for p in prm.parameter_names:
    header += p+' '
    
np.savetxt(fname='pyrolite_assemblages.dat',
           X=np.array([sol.x for sol_list in sols for sol in sol_list]),
           fmt='%.4e',
           header=header)

        
for sol_list in sols:
    T = sol_list[0].assemblage.temperature
    pressures = np.empty(len(sol_list))
    p_wus = np.empty(len(sol_list))
    for i, sol in enumerate(sol_list):
        pressures[i] = sol.assemblage.pressure
        p_wus[i] = sol.assemblage.phases[1].molar_fractions[1]

    plt.plot(pressures/1.e9, p_wus, label='{0} K'.format(T))

plt.xlabel('Pressure (GPa)')
plt.ylabel('molar proportion of FeO in (Mg,Fe)O')
plt.legend()

plt.savefig('P_p_wus.pdf')
plt.show()
