from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman
from burnman.equilibrate import equilibrate
from burnman.minerals.boukare import bridgmanite_boukare, ferropericlase_boukare, stishovite_boukare, melt_boukare

fper = ferropericlase_boukare()
bdg = bridgmanite_boukare()
stv = stishovite_boukare()
liq = melt_boukare()

fitting=True
if fitting:
    pressure = 120.e9
    Xs = np.linspace(0.6, 0.99999, 21) # for fitting
else:
    pressure = 135.e9
    Xs = np.linspace(0.2, 0.99999, 21) # for plotting


solidi = np.empty_like(Xs)
liquidi = np.empty_like(Xs)
melting_entropy = np.empty_like(Xs)
melting_volume = np.empty_like(Xs)
density_solid = np.empty_like(Xs)
density_liquid = np.empty_like(Xs)
c_MgO = np.empty_like(Xs)
c_FeO = np.empty_like(Xs)
c_SiO2 = np.empty_like(Xs)
c_MgO2 = np.empty_like(Xs)
c_FeO2 = np.empty_like(Xs)
c_SiO22 = np.empty_like(Xs)
c_MgO3 = np.empty_like(Xs)
c_FeO3 = np.empty_like(Xs)
c_SiO23 = np.empty_like(Xs)

solid_assemblage = burnman.Composite([fper, bdg])
assemblage = burnman.Composite([fper, bdg, liq])


fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


#for P in np.linspace(110.e9, 140.e9, 4):
for P in [pressure]:
    for i, x in enumerate(Xs):


        # The composition should be fairly close to the cotectic,
        # but we calculate the cotectic composition here anyway
        composition = {'Mg': x, 'Fe': (1. - x), 'Si': 0.5, 'O': 2.}
        assemblage.set_state(P, 5000.)
        assemblage.set_fractions([0.5, 0.5, 0.])

        X_Si_guess = 0.4
        MgO = composition['Mg']/(composition['Mg'] + composition['Fe'])
        fper.set_composition([MgO, 1. - MgO])
        bdg.set_composition([MgO, 1. - MgO])
        liq.set_composition([(1. - x)*(1. - X_Si_guess), x*(1. - X_Si_guess), X_Si_guess])
        fper.guess = fper.molar_fractions
        bdg.guess = bdg.molar_fractions
        liq.guess = liq.molar_fractions

        # First, let's find the liquid composition and temperature at the cotectic
        equality_constraints = [('P', P), ('phase_proportion', (liq, np.array([0.])))]
        sol, prm = equilibrate(composition, assemblage,
                               equality_constraints,
                               initial_state_from_assemblage=True,
                               initial_composition_from_assemblage=True,
                               store_iterates=False)

        liquidus_T = sol.x[1]
        x_MgO, x_SiO2 = sol.x[-2:]
        x_FeO = 1. - x_MgO - x_SiO2

        # Now, let's find the solidus temperature at that composition:
        composition = {'Mg': x_MgO, 'Fe': x_FeO, 'Si': x_SiO2, 'O': 1. + x_SiO2}

        c_MgO[i] = x_MgO
        c_FeO[i] = x_FeO
        c_SiO2[i] = x_SiO2

        X_Mg = x_MgO/(x_MgO + x_FeO)
        X_Si_guess = 0.4
        fper.guess = np.array([X_Mg, 1. - X_Mg])
        bdg.guess = np.array([X_Mg, 1. - X_Mg])
        liq.guess = np.array([(1. - X_Mg)*(1. - X_Si_guess), X_Mg*(1. - X_Si_guess), X_Si_guess])

        equality_constraints = [('P', P), ('phase_proportion', (liq, np.array([0.])))]
        sol, prm = equilibrate(composition, assemblage,
                               equality_constraints,
                               initial_state_from_assemblage=True,
                               initial_composition_from_assemblage=False,
                               store_iterates=False)
        solidus_T = sol.x[1]

        solidi[i] = solidus_T
        liquidi[i] = liquidus_T


        # Now let's find the melting volume and entropy in the middle of the melt interval
        # (and hope that it is roughly linear!!!)
        mid_T = (solidus_T + liquidus_T)/2.
        equality_constraints = [('P', P), ('T', mid_T)]
        sol, prm = equilibrate(composition, solid_assemblage,
                               equality_constraints,
                               initial_composition_from_assemblage=False,
                               store_iterates=False)

        liq.set_composition([x_FeO, x_MgO, x_SiO2])
        liq.set_state(P, mid_T)

        melting_entropy[i] = liq.S - solid_assemblage.S*solid_assemblage.n_moles
        melting_volume[i] = liq.V - solid_assemblage.V*solid_assemblage.n_moles
        density_solid[i] = solid_assemblage.density
        density_liquid[i] = liq.density

        # And the bdg-stv-liq cotectic (just for ternary plotting)
        composition2 = {'Mg': X_Mg/4., 'Fe': (1. - X_Mg/4.), 'Si': 1.0, 'O': 2.5}
        X_Si_guess = 0.7
        bdg.set_composition([X_Mg/4., 1. - X_Mg/4.])
        liq.set_composition([(1. - X_Mg/4.)*(1. - X_Si_guess), X_Mg/4.*(1. - X_Si_guess), X_Si_guess])
        bdg.guess = bdg.molar_fractions
        liq.guess = liq.molar_fractions
        assemblage2 = burnman.Composite([bdg, stv, liq])
        assemblage2.set_state(P, 5000.)
        assemblage2.set_fractions([0.5, 0.5, 0.])

        # Find the liquid composition and temperature at the cotectic
        equality_constraints = [('P', P), ('phase_proportion', (liq, np.array([0.])))]
        sol, prm = equilibrate(composition2, assemblage2,
                               equality_constraints,
                               initial_state_from_assemblage=True,
                               initial_composition_from_assemblage=True,
                               store_iterates=False)


        x_MgO, x_SiO2 = sol.x[-2:]
        x_FeO = 1. - x_MgO - x_SiO2

        c_MgO2[i] = x_MgO
        c_FeO2[i] = x_FeO
        c_SiO22[i] = x_SiO2

        # And the fper-stv-liq cotectic (just for ternary plotting)
        composition2 = {'Mg': X_Mg, 'Fe': (1. - X_Mg), 'Si': 1., 'O': 2.}
        X_Si_guess = 0.6
        fper.set_composition([X_Mg, 1. - X_Mg])
        liq.set_composition([(1. - X_Mg)*(1. - X_Si_guess), X_Mg*(1. - X_Si_guess), X_Si_guess])
        fper.guess = fper.molar_fractions
        liq.guess = liq.molar_fractions
        assemblage2 = burnman.Composite([fper, stv, liq])
        assemblage2.set_state(P, 5000.)
        assemblage2.set_fractions([0.5, 0.5, 0.])

        # Find the liquid composition and temperature at the cotectic
        equality_constraints = [('P', P), ('phase_proportion', (liq, np.array([0.])))]
        sol, prm = equilibrate(composition2, assemblage2,
                               equality_constraints,
                               initial_state_from_assemblage=True,
                               initial_composition_from_assemblage=True,
                               store_iterates=False)


        x_MgO, x_SiO2 = sol.x[-2:]
        x_FeO = 1. - x_MgO - x_SiO2

        c_MgO3[i] = x_MgO
        c_FeO3[i] = x_FeO
        c_SiO23[i] = x_SiO2

    #Finally, plot the FeO-SiO2 eutectic point

    composition3 = {'Fe': 0.5, 'Si': 0.5, 'O': 1.}
    assemblage3 = burnman.Composite([fper.endmembers[1][0], stv, liq])
    assemblage3.set_state(P, 5000.)
    assemblage3.set_fractions([0.5, 0.5, 0.])
    equality_constraints = [('P', P), ('phase_proportion', (liq, np.array([0.])))]
    sol, prm = equilibrate(composition3, assemblage3,
                           equality_constraints,
                           initial_state_from_assemblage=True,
                           initial_composition_from_assemblage=True,
                           store_iterates=False)
    print(sol.x)
    FS_liq_SiO2 = sol.x[-1]

    # Plotting used to be here

for i in range(4):
    ax[i].set_xlim(0., 1.)
    ax[i].legend(loc='best')
    ax[i].set_xlabel('MgO/(MgO + FeO) in pyrolite')

ax[0].set_ylabel('Temperature (K)')
ax[1].set_ylabel('Melting entropy (J/K/mol)')
ax[2].set_ylabel('Melting volume (cm$^3$/mol)')
ax[3].set_ylabel('Density (kg/m$^3$)')


# Now we can do some linearisation of the problem:
mask = [i for i, x in enumerate(c_MgO) if x > 0.25]

from scipy.optimize import curve_fit
func_linear = lambda x, a, b: (1. - x)*a + x*b
func_quadratic = lambda x, c: c*x*(1. - x)

mbr_MgO, _ = curve_fit(func_linear, c_SiO2[mask], c_MgO[mask])
mbr_FeO, _ = curve_fit(func_linear, c_SiO2[mask], c_FeO[mask])
mbr_SiO2, _ = curve_fit(func_linear, c_SiO2[mask], c_SiO2[mask])
mbr_S, _ = curve_fit(func_linear, c_SiO2[mask], melting_entropy[mask])
mbr_V, _ = curve_fit(func_linear, c_SiO2[mask], melting_volume[mask])

# One end should have c_MgO = 0, the other should have c_FeO as zero
x = np.array([mbr_MgO[0]/(mbr_MgO[0] - mbr_MgO[1]),
              mbr_FeO[0]/(mbr_FeO[0] - mbr_FeO[1])])

mbr_MgO = func_linear(x, *mbr_MgO)
mbr_FeO = func_linear(x, *mbr_FeO)
mbr_SiO2 = func_linear(x, *mbr_SiO2)
mbr_S = func_linear(x, *mbr_S)
mbr_V = func_linear(x, *mbr_V)

Xs_out = c_MgO/mbr_MgO[1]

ax[0].plot(Xs_out, solidi, label='solidus: {0} GPa'.format(P/1.e9))
ax[0].plot(Xs_out, liquidi, label='liquidus: {0} GPa'.format(P/1.e9))

ax[1].plot(Xs_out, melting_entropy, label='liquidus: {0} GPa'.format(P/1.e9))
ax[2].plot(Xs_out, melting_volume*1.e6, label='liquidus: {0} GPa'.format(P/1.e9))
ax[3].plot(Xs_out, density_solid, label='solid at {0} GPa'.format(P/1.e9))
ax[3].plot(Xs_out, density_liquid, label='liquid at {0} GPa'.format(P/1.e9))



print(mbr_MgO)
print(mbr_FeO)
print(mbr_SiO2)

print(solidi[-1])

# Begin simple melting model parameterisation (to fit melting properties)
R = 8.31446
Pref = 120.e9
P = 120.e9

def misfit(n):
    """
    Fit the n mole reactants and melting T/1000. of the Fe endmember
    """
    mbr_melting_T = np.array([n[2]*1000., solidi[-1]])
    sig = 0
    for i, X in enumerate(Xs_out):

        if X > 0.7:
            T = solidi[i]
            dG = (mbr_melting_T - T) * mbr_S + (P - Pref) * mbr_V
            Xls = 1.0 - (1.0 - np.exp(dG[1]/(n[1]*R*T)))/(np.exp(dG[0]/(n[0]*R*T)) -
                                                          np.exp(dG[1]/(n[1]*R*T)))
            Xss = Xls * np.exp(dG[1]/(n[1]*R*T))
            sig += np.power(Xss - Xs_out[i], 2.)

        if X < 0.5:
            T = liquidi[i]
            dG = (mbr_melting_T - T) * mbr_S + (P - Pref) * mbr_V
            Xls = 1.0 - (1.0 - np.exp(dG[1]/(n[1]*R*T)))/(np.exp(dG[0]/(n[0]*R*T)) -
                                                          np.exp(dG[1]/(n[1]*R*T)))
            sig += np.power(Xls - Xs_out[i], 2.)
    return sig

from scipy.optimize import minimize
sol = minimize(misfit, [0.5, 0.5, 3.47]).x
n = sol[0:2]
mbr_melting_T = np.array([sol[2]*1000., solidi[-1]])

temperatures = np.linspace(mbr_melting_T[0], mbr_melting_T[1], 101)

Xls = np.empty_like(temperatures)
Xss = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    dG = (mbr_melting_T - T) * mbr_S + (P - Pref) * mbr_V
    Xls[i] = 1.0 - (1.0 - np.exp(dG[1]/(n[1]*R*T)))/(np.exp(dG[0]/(n[0]*R*T)) -
                                                     np.exp(dG[1]/(n[1]*R*T)))
    Xss[i] = Xls[i] * np.exp(dG[1]/(n[1]*R*T))

ax[0].plot(Xls, temperatures, linestyle='--')
ax[0].plot(Xss, temperatures, linestyle='--')

ax[0].legend()

xs = np.linspace(0., 1., 101)
ax[1].plot(xs, mbr_S.dot(np.array([1 - xs, xs])), linestyle='--')
ax[2].plot(xs, mbr_V.dot(np.array([1 - xs, xs]))*1.e6, linestyle='--')

print(mbr_MgO, mbr_FeO, mbr_SiO2, mbr_S, mbr_V, mbr_melting_T, n)



import ternary
fig, tax = ternary.figure()
# Draw boundary and gridlines
tax.ax.axis("off")
tax.gridlines(color="blue", multiple=0.1)
tax.ticks(multiple=0.2,offset=0.02, tick_formats="%.1f")

# Set axis labels
fontsize = 10
#tax.left_corner_label("FeO", fontsize=fontsize)
#tax.right_corner_label("MgO", fontsize=fontsize)
#tax.top_corner_label("SiO$_2$", fontsize=fontsize)

tax.bottom_axis_label("MgO", fontsize=fontsize, offset=0.14)
tax.right_axis_label("SiO$_2$", fontsize=fontsize, offset=0.14)
tax.left_axis_label("FeO", fontsize=fontsize, offset=0.14)

tax.plot(np.array([c_MgO, c_SiO2, c_FeO]).T, linewidth=2.0, label="bdg-stv cotectic")
tax.plot(np.array([c_MgO2, c_SiO22, c_FeO2]).T, linewidth=2.0, label="bdg-fper cotectic")
tax.plot(np.array([c_MgO3, c_SiO23, c_FeO3]).T, linewidth=2.0, label="fper-stv cotectic")

tax.plot(np.array([mbr_MgO, mbr_SiO2, mbr_FeO]).T, linewidth=2.0, linestyle='--', label="Linear cotectic")
fig.savefig('output_figures/FMS_ternary_{0:.0f}_GPa.pdf'.format(pressure/1.e9))
plt.show()
