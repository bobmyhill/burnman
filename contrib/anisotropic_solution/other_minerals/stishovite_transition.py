from burnman import ElasticSolution, Mineral
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class stishovite (Mineral):
    def __init__(self):
        self.params = {
            'name': 'Stishovite',
            'formula': {'Si': 1., 'O': 2.},
            'equation_of_state': 'slb3',
            'F_0': -818984.6,
            'V_0': 1.4017e-05,
            'K_0': 3.143352e+11,
            'Kprime_0': 3.75122,
            'Debye_0': 1107.824,
            'grueneisen_0': 1.37466,
            'q_0': 2.83517,
            'G_0': 2.2e+11,
            'Gprime_0': 1.93334,
            'eta_s_0': 4.60904,
            'n': 3.,
            'molar_mass': 0.06008}
        Mineral.__init__(self)


def F_xs(volume, temperature, molar_amounts):
    n_moles = sum(molar_amounts)
    molar_fractions = molar_amounts / n_moles
    x = molar_fractions[0]*molar_fractions[1]

    a = -1.8e3
    b = 0.
    c = -2.e9
    return n_moles * (a*x + b*x + c*x*x*(volume - 1.4017e-5))


class stv_solution(ElasticSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'stishovite'
        self.solution_type = 'function'
        self.endmembers = [[stishovite(), '[Si]O2'],
                           [stishovite(), '[Si]O2']]
        self.excess_helmholtz_function = F_xs

        ElasticSolution.__init__(self, molar_fractions=molar_fractions)


stv = stv_solution()
stv0 = stishovite()


def gibbs(x, pressure, temperature):

    stv.set_state(pressure, temperature)
    stv.set_composition([x[0], 1.-x[0]])
    return stv.gibbs


fig = plt.figure(figsize=(8, 8))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
pressures = np.linspace(1.e5, 100.e9, 51)
xs = np.empty_like(pressures)
Vs = np.empty_like(pressures)
dVs = np.empty_like(pressures)
dGs = np.empty_like(pressures)

for T in [300., 1000., 2000.]:
    for i, P in enumerate(pressures):
        popt = minimize(gibbs, x0=[0.501], args=(P, T), bounds=[(0.5, 1.)])
        xs[i] = popt.x[0]
        Vs[i] = stv.V
        G = stv.gibbs
        stv.set_composition([0.5, 0.5])
        dVs[i] = Vs[i] - stv.V
        dGs[i] = G - stv.gibbs

    ax[0].plot(pressures/1.e9, xs, label=f'{T} K')
    ax[1].plot(pressures/1.e9, Vs, label=f'{T} K')
    ax[2].plot(pressures/1.e9, dVs, label=f'{T} K')
    ax[3].plot(pressures/1.e9, dGs, label=f'{T} K')
ax[0].legend()
plt.show()
exit()
xs = np.linspace(-0.1, 1.1, 51)
gibbs = np.empty_like(xs)

fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

for P in [1.e5, 10.e9, 50.e9, 100.e9]:
    print(P/1.e9)
    stv.set_state(P, 300.)
    stv.set_composition([0, 1.])
    gibbs0 = stv.gibbs
    for i, x in enumerate(xs):
        stv.set_composition([x, 1.-x])
        gibbs[i] = stv.gibbs

    # plt.plot(xs, (gibbs - gibbs0)/stv.V, label=f'{P/1.e9} GPa')
    ax[0].plot(xs, (gibbs - gibbs0), label=f'{P/1.e9} GPa')

pressures = np.linspace(1.e5, 100.e9, 51)
temperatures = pressures * 0. + 300.

stv.set_composition([0.5, 0.5])
gibbs, volumes = stv.evaluate(['gibbs', 'V'], pressures, temperatures)
ax[1].plot(pressures, volumes, label='stv')
ax[2].plot(pressures, volumes-np.gradient(gibbs, pressures, edge_order=2),
           label='stv')

stv.set_composition([0., 1.])
gibbs, volumes = stv.evaluate(['gibbs', 'V'], pressures, temperatures)
ax[1].plot(pressures, volumes, label='post-stv')
ax[2].plot(pressures, volumes-np.gradient(gibbs, pressures, edge_order=2),
           label='post-stv')


gibbs, volumes = stv0.evaluate(['gibbs', 'V'], pressures, temperatures)
ax[2].plot(pressures, volumes-np.gradient(gibbs, pressures, edge_order=2),
           label='stv_endmember')

ax[0].legend()
ax[1].legend()
plt.show()
