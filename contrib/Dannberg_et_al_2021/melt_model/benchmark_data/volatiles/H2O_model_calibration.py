from __future__ import absolute_import
import numpy as np
import burnman
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

gas_constant = 8.31446
H2O_mantle_melt_params={'name': 'H2O melt mantle component; 10.1038/nature06918',
                        'equation_of_state': 'mod_hp_tmt',
                        
                        'Pref': 100.e9,
                        'T_0': 298.15,
                        
                        'V_0': 1.086e-05,
                        'a_0': 6.475e-05,
                        'K_0': 73.3e9,
                        'Kprime_0': 2.265,
                        'Kdprime_0': -1.04e-11,
                        
                        'T_einstein': 242., # 0.806*300
                        
                        'H_Pref': 0., # not used
                        'S_Pref': 0., # not used
                        'Cp_Pref': np.array([74.83, 0., 0., 0.]), # 3nR
                        
                        'molar_mass': 0.01801528,
                        'n': 3.,
                        'formula': {'H': 2., 'O': 1.}}



melt = burnman.Mineral(params = H2O_mantle_melt_params)


# DATA
fit = False

P, T, V, Verr = np.loadtxt('H2O_partial_molar_volumes.dat', unpack=True)
P = np.concatenate((P, [140, 140]))
T = np.concatenate((T, [3000., 6000.]))
V = np.concatenate((V, [4.9, 5.7]))
Verr = np.concatenate((Verr, [0.1, 0.1]))


mask = [i for i, p in enumerate(P) if p > 5.]
P = P[mask]
T = T[mask]
V = V[mask]
Verr = Verr[mask]


if fit:
    PTV = np.array([P*1.e9, T, V*1.e-6]).T
    nul = np.array([0.]*len(P))
    PTV_covariances = np.array([[nul, nul, nul], [nul, nul, nul], [nul, nul, Verr*1.e-6]]).T
    
    params = ['V_0', 'K_0', 'Kprime_0', 'Kdprime_0', 'a_0']
    fitted_eos = burnman.eos_fitting.fit_PTV_data(melt, params, PTV, PTV_covariances, verbose=False)

    print(fitted_eos.popt)


fig = plt.figure()
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

img_VH2O = mpimg.imread('H2O_melt_partial_volume.png')
ax[0].imshow(img_VH2O, extent=[0., 140., 5, 25], aspect='auto')

    
for T in np.linspace(0., 6000., 7):
    pressures = np.linspace(max(1.e5, (4.*T/6000 - 3.)*10.e9), 140.e9, 101)
    temperatures = pressures*0. + T
    ax[0].plot(pressures/1.e9, melt.evaluate(['V'], pressures, temperatures)[0]*1.e6, label='{0} K'.format(T))
    ax[1].plot(pressures/1.e9, melt.evaluate(['rho'], pressures, temperatures)[0]/1.e3, label='{0} K'.format(T))

ax[0].set_ylim(0., 25)
plt.show()
