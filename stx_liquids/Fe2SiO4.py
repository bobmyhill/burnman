import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import DKS_2013_liquids
from burnman.minerals import RS_2014_liquids
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dataset='SLB'
if dataset=='HP':
    fa = burnman.minerals.HHPH_2013.fa()
    frw = burnman.minerals.HHPH_2013.frw()
    wus = burnman.minerals.HHPH_2013.fper()
    stv = burnman.minerals.HHPH_2013.stv()
elif dataset=='SLB':
    fa = burnman.minerals.SLB_2011.fayalite()
    frw = burnman.minerals.SLB_2011.fe_ringwoodite()
    wus = burnman.minerals.SLB_2011.wuestite()
    stv = burnman.minerals.SLB_2011.stishovite()
else:
    exit()
    
fa_liq = RS_2014_liquids.Fe2SiO4_liquid()

# Plot data (1)
data = np.loadtxt(fname='data/fa_melting_Ohtani_1979.dat')
Ohtani_data = [[], [], [], []]
for datum in data:
    #print datum
    Ohtani_data[int(datum[3]-1.)].append((datum[0]*1.e8, datum[1]+273.15, datum[2]))

assemblages=['fa, Ohtani et al., 1979',
             'quench fa, Ohtani et al., 1979',
             'frw, Ohtani et al., 1979',
             'quench frw, Ohtani et al., 1979']
for i, assemblage in enumerate(Ohtani_data):
    P, T, Terr = np.array(zip(*assemblage))
    plt.errorbar(P/1.e9, T, yerr=Terr, marker='o', markersize=6, linestyle='None', label=assemblages[i])

# Plot data (2)
data = np.loadtxt(fname='data/frw_melting_Kato_2016.dat')
Kato_data = [[], []]
for datum in data:
    #print datum
    Kato_data[int(datum[3]-1.)].append((datum[1]*1.e9, datum[2], 100.))

assemblages=['melted, Kato et al., 2016',
             'not melted, Kato et al., 2016']
for i, assemblage in enumerate(Kato_data):
    P, T, Terr = np.array(zip(*assemblage))
    plt.errorbar(P/1.e9, T, yerr=Terr, marker='o', markersize=6, linestyle='None', label=assemblages[i])


    

# Hsu up to 40 kbar
# Akimoto et al., 1967
P = lambda T: 41.*(np.power(T/1478., 4.8) - 1.)*1.e8
T = np.linspace(1478., 1800., 101)
plt.plot(P(T)/1.e9, T, label='Akimoto et al., 1967')

# Let's set a few things to get the right answers:
# There seems to be agreement about the temperature of melting at ambient pressure
# and at the triple point

# This is enough for us to find the best fitting delta_E and delta_S:

T0 = 1205.+273.15
P0 = 1.e5

T_tp = 1520. + 273.15
P_tp = burnman.tools.equilibrium_pressure([fa, frw], [1.0, -1.0], T_tp)


for m in [fa, fa_liq]:
    m.set_state(P0, T0)

dG0 = fa.gibbs - fa_liq.gibbs

for m in [fa, fa_liq]:
    m.set_state(P_tp, T_tp)

dG_tp = fa.gibbs - fa_liq.gibbs

delta_V = 0.
delta_S = -(dG_tp - dG0)/(T_tp - T0)
delta_E = dG0 - (P0*delta_V - T0*delta_S)

'''
dPdT0 = P(T_tp + 1.) - P(T_tp) # using Akimoto
dPdT1 = (80.e8 - 62.e8)/(1700. - 1520.)


Vliq = ((frw.S - fa.S) - frw.V*dPdT1 + fa.V*dPdT0)/(dPdT0 - dPdT1)
Sliq = dPdT1*(Vliq - frw.V) + frw.S


for m in [fa, frw, fa_liq]:
    m.set_state(P, T)

delta_S = -90. # Sliq - fa_liq.S
delta_E = fa.gibbs - fa_liq.gibbs - (P*delta_V - T*delta_S)
'''

fa_liq.property_modifiers = [['linear', {'delta_E': delta_E, 'delta_S': delta_S, 'delta_V': delta_V}]]

'''
# Check
for m in [fa, frw, fa_liq]:
    m.set_state(P_tp, T_tp)
print fa.gibbs, frw.gibbs, fa_liq.gibbs
'''

P_tp, T_tp = burnman.tools.invariant_point([fa, frw], [1.0, -1.0],
                                           [fa, fa_liq], [1.0, -1.0])

P_tp2, T_tp2 = burnman.tools.invariant_point([wus, stv, frw], [2.0, 1.0, -1.0],
                                           [frw, fa_liq], [1.0, -1.0])

# Make sure gibbs is the same at the melting point
temperatures = np.linspace(800., T_tp, 21)
Pr = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    Pr[i] = burnman.tools.equilibrium_pressure([fa, frw], [1.0, -1.0], T)

plt.plot(Pr/1.e9, temperatures, label='fa-frw')



pressures = np.linspace(1.e5, 35.e9, 101)
Tm = np.empty_like(pressures)
Tm2 = np.empty_like(pressures)
for i, P in enumerate(pressures):
    if P < P_tp:
        Tm[i] = burnman.tools.equilibrium_temperature([fa, fa_liq], [1.0, -1.0], P)
    elif P < P_tp2:
        Tm[i] = burnman.tools.equilibrium_temperature([frw, fa_liq], [1.0, -1.0], P)
    else:
        Tm[i] = burnman.tools.equilibrium_temperature([wus, stv, fa_liq], [2.0, 1.0, -1.0], P, 3000.)
        
plt.plot(pressures/1.e9, Tm, label='modelled melting temperature')



pressures = np.linspace(1.e5, 39.e9, 101)
Tm = np.empty_like(pressures)
for i, P in enumerate(pressures):
    Tm[i] = burnman.tools.equilibrium_temperature([fa, fa_liq], [1.0, -1.0], P)

plt.plot(pressures/1.e9, Tm, linestyle='--', label='metastable fa melting temperature')


temperatures = np.linspace(800., T_tp2, 21)
Pr = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
     Pr[i] = burnman.tools.equilibrium_pressure([wus, stv, frw], [2.0, 1.0, -1.0], T)

plt.plot(Pr/1.e9, temperatures, linestyle='-.', label='frw -> wus + stv')





plt.xlabel('P (GPa)')
plt.ylabel('T (K)')
plt.xlim(0., 20.)
plt.ylim(1600., 2200.)
plt.legend(loc='lower right')
plt.show()

test=False
if test==True:
    TE = np.loadtxt(fname='Ramo.dat')

    '''
    for temperature, rho, Ee, Ei, Ex in TE:
        volume = fa_liq.params['molar_mass']/rho

        E_el = (fa_liq.method._F_el(temperature, volume, fa_liq.params)
                + temperature
                * fa_liq.method._S_el(temperature, volume, fa_liq.params))
        E_ig = (fa_liq.method._F_ig(temperature, volume, fa_liq.params)
                + temperature
                * fa_liq.method._S_ig(temperature, volume, fa_liq.params))

        E_xs = (fa_liq.method._F_xs(temperature, volume, fa_liq.params)
                + temperature
                * fa_liq.method._S_xs(temperature, volume, fa_liq.params))

        print(E_el/Ee, E_ig/Ei, E_xs/Ex)
    '''


    Pm = 1.e5
    Tm = 1573.


    dT = 1.
    dV = 1.e-8


    fa.set_state(Pm, Tm)
    fa_liq.set_state(Pm, Tm)

    print(fa.V, fa_liq.V, fa_liq.rho)

    Vm = fa_liq.V

    temperatures = np.linspace(Tm - dT, Tm + dT, 3)
    volumes = np.linspace(Vm - dV, Vm + dV, 3)

    F = np.empty_like(temperatures)
    S = np.empty_like(temperatures)
    C_v = np.empty_like(temperatures)
    P = np.empty_like(temperatures)
    K_T = np.empty_like(temperatures)
    alphaK_T = np.empty_like(temperatures)

    for i, T in enumerate(temperatures):
        F[i] = fa_liq.method._F_mag(T, Vm, fa_liq.params)
        S[i] = fa_liq.method._S_mag(T, Vm, fa_liq.params)
        C_v[i] = fa_liq.method._C_v_mag(T, Vm, fa_liq.params)

    F_mag = F[1]
    S_mag_analytic = S[1]
    S_mag_numeric = -np.gradient(F)[1]/dT
    C_v_mag_analytic = C_v[1]
    C_v_mag_numeric = Tm*np.gradient(S)[1]/dT

    '''
        def _alphaK_T_mag(self, temperature, volume, params):
            S_a, S_b, numerator, numerator_2, n_atoms = self._spin(temperature, volume, params)
            S = S_a*temperature + S_b
            d2FdVdT = (-2.*params['spin_b'][1]*temperature/(params['V_0']*np.power(VoverVx, 2.))
                       + numerator)/(2.*S + 1.) - 2.*temperature*S_a*numerator/(np.power((2.*S+1.), 2.))
            return -n_atoms*constants.gas_constant*d2FdVdT
    '''

    for i, V in enumerate(volumes):
        F[i] = fa_liq.method._F_mag(Tm, V, fa_liq.params)
        P[i] = fa_liq.method._P_mag(Tm, V, fa_liq.params)
        K_T[i] = fa_liq.method._K_T_mag(Tm, V, fa_liq.params)

        alphaK_T[i] = fa_liq.method._alphaK_T_mag(Tm, V, fa_liq.params)
        S[i] = fa_liq.method._S_mag(Tm, V, fa_liq.params)

    P_mag_analytic = P[1]
    P_mag_numeric = -np.gradient(F)[1]/dV
    K_T_mag_analytic = K_T[1]
    K_T_mag_numeric = -Vm*np.gradient(P)[1]/dV
    alphaK_T_mag_analytic = alphaK_T[1]
    alphaK_T_mag_numeric = np.gradient(S)[1]/dV

    print('P:', P_mag_analytic/P_mag_numeric)
    print('K_T:', K_T_mag_analytic/K_T_mag_numeric)
    print('alphaK_T:', alphaK_T_mag_analytic/alphaK_T_mag_numeric)
    print('S:', S_mag_analytic/S_mag_numeric)
    print('C_v:', C_v_mag_analytic/C_v_mag_numeric)


    pressures = np.linspace(1.e5, 200.e9, 51)
    rhos = np.empty_like(pressures)


    fig1 = mpimg.imread('figures/Fe2SiO4_liquid_PVT.png')
    plt.imshow(fig1, extent=[3.5, 7.5, 0., 200], aspect='auto')

    for T in [3000., 4000., 6000.]: #, 5000., 2000., 1000.]:
        for i, P in enumerate(pressures):
            fa_liq.set_state(P, T)

            #print(P, T, fa_liq.rho)
            rhos[i] = fa_liq.rho
        plt.plot(rhos/1.e3, pressures/1.e9, label=str(T)+' K')

    plt.legend(loc='upper left')
    plt.show()

    fig1 = mpimg.imread('figures/Fe2SiO4_liquid_hugoniot.png')
    plt.imshow(fig1, extent=[3.5, 7.5, 0, 200], aspect='auto')


    temperatures, volumes = burnman.tools.hugoniot(fa_liq, 1.e5, 1573., pressures)
    plt.plot(fa_liq.molar_mass/volumes/1.e3, pressures/1.e9)


    plt.show()
