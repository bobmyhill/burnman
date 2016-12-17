# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

'''
example_gibbs_minimization
--------------------

This example demonstrates how burnman may be used to calculate the
equilibrium phase proportions and compositions for an assemblage
of a fixed bulk composition.
It then explains how to perform the basic i/o of BurnMan in a format 
that can be read in by the ASPECT mantle convection code.

*Uses:*

* :doc:`mineral_database`
* :class:`burnman.composite.Composite`
* :func:`burnman.equilibriumassemblage.gibbs_minimizer`
'''
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman.minerals import HP_2011_ds62, SLB_2011
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy import interpolate
from burnman.equilibriumassemblage import *

if __name__ == "__main__":

    def three_phase_eqm(P, T, composition, guess_three_phase):
        def three_phase_solution(args, P, T, composition):
            n_ppv, x_al_ppv, x_fe_ppv, x_fe_bdg = args
            n_bdg = (composition['Si'] + 0.5*composition['Al']) - n_ppv
            n_per = composition['O'] - 3.*(n_ppv + n_bdg)
            
            x_mg_ppv = 1. - x_al_ppv - x_fe_ppv
        
            x_al_bdg = (0.5*composition['Al'] - n_ppv*x_al_ppv)/n_bdg
            x_mg_bdg = 1. - x_al_bdg - x_fe_bdg
                
            x_fe_per = (composition['Fe'] - n_ppv*x_fe_ppv - n_bdg*x_fe_bdg)/n_per
            x_mg_per = 1. - x_fe_per
    
            amounts = np.array([n_ppv, n_bdg, n_per])
            proportions = amounts/np.sum(amounts)
            
            ppv.set_composition([x_mg_ppv, x_fe_ppv, x_al_ppv])
            bdg.set_composition([x_mg_bdg, x_fe_bdg, x_al_bdg])
            fper.set_composition([x_mg_per, x_fe_per])
            ppv.set_state(P, T)
            bdg.set_state(P, T)
            fper.set_state(P, T)
    
            # Mg amount
            # Al2O3 activities
            # FeSiO3 activities (bdg, ppv)
            # Mg-Fe exchange (bdg, per)
            return [ppv.partial_gibbs[0] - bdg.partial_gibbs[0],
                    ppv.partial_gibbs[2] - bdg.partial_gibbs[2],
                    (ppv.partial_gibbs[1] + fper.partial_gibbs[0]) - (ppv.partial_gibbs[0] + fper.partial_gibbs[1]),
                    (bdg.partial_gibbs[1] + fper.partial_gibbs[0]) - (bdg.partial_gibbs[0] + fper.partial_gibbs[1])]
    
        sol = opt.fsolve(three_phase_solution, guess_three_phase, args=(P, T, composition), full_output=True)
        guess_three_phase = sol[0]
        n_ppv = sol[0][0]
        n_bdg = (composition['Si'] + 0.5*composition['Al']) - n_ppv
        n_per = composition['O'] - 3.*(n_ppv + n_bdg)
        amounts = np.abs(np.array([n_ppv, n_bdg, n_per])) # frustratingly, solve doesn't quite have the accuracy to make all amounts positive
        proportions = amounts/np.sum(amounts)
        if sol[2] == 1:
            return burnman.Composite([ppv, bdg, fper], proportions)
        else:
            raise Exception('Could not find three phase solution')

    # Let's solve the equations at a fixed temperature of 2000 K
    # and Mg-rich composition with a little bit of Al2O3
    # composition = { 'Mg': 1.775, 'Fe': 0.2, 'Al': 0.05, 'Si': 0.975, 'O': 4.}
    composition = { 'Mg': 1.679, 'Fe': 0.296, 'Al': 0.05, 'Si': 0.975, 'O': 4.}
    bdg = SLB_2011.mg_fe_bridgmanite() # Mg, Fe, Al
    ppv = SLB_2011.post_perovskite() # Mg, Fe, Al
    fper = SLB_2011.ferropericlase() # Mg, Fe
    assemblage1 = burnman.Composite([bdg, fper])
    assemblage2 = burnman.Composite([ppv, fper])
    
    # The next few lines do all the work, looping over lower mantle pressures
    # and finding the equilibrium composition at each P-T point.
    # pressures = np.linspace(20.0e9, 140.e9, 51)
    pressures = np.linspace(2.e9, 140.e9, 101)
    pstep = (pressures[1] - pressures[0])/1.e5

    # temperatures = np.linspace(1500., 3500., 51)
    temperatures = np.linspace(1500., 3773., 101)
    Tstep = temperatures[1] - temperatures[0]

    P_bdg = []
    x_bdg = []
    P_ppv = []
    x_ppv = []
    P_fper = []
    x_fper = []
    rock = []
    guess_three_phase = [0.001, 0.03, 0.03, 0.03]

    pressure_list = []
    temperature_list = []
    densities = []
    compressibilities = []
    s_velocities = []
    p_velocities = []
    thermal_expansivities = []
    heat_capacities = []
    enthalpies = []

    ppv_in = find_univariant(composition, [bdg, fper], ppv, 'T', temperatures, [100.e9, 2000., 0.0, 0.005, 0.025, 1.0, 0.005, 0.025, 1., 0.19525])
    pv_out = find_univariant(composition, [ppv, fper], bdg, 'T', temperatures, [100.e9, 2000., 0.0, 0.005, 0.025, 1.0, 0.005, 0.025, 1., 0.19525])

    for i, P in enumerate(pressures):

        for j, T in enumerate(temperatures):

            P0 = ppv_in[j][0]
            P1 = pv_out[j][0]
            found_solution = None

            try:
                if P < P0:
                    sol = gibbs_minimizer(composition, assemblage1, [['P', P], ['T', T]])
                    P_bdg.append(P)
                    x_bdg.append(sol['c'][1])
                    P_fper.append(P)
                    x_fper.append(sol['c'][4])
                    rock = assemblage1
            
                elif P > P1:
                    sol = gibbs_minimizer(composition, assemblage2, [['P', P], ['T', T]])
                    P_ppv.append(P)
                    x_ppv.append(sol['c'][1])
                    P_fper.append(P)
                    x_fper.append(sol['c'][4])
                    rock = assemblage2
            
                else:
                    rock = three_phase_eqm(P, T, composition, guess_three_phase)
                    P_ppv.append(P)
                    P_bdg.append(P)
                    P_fper.append(P)
                    x_ppv.append(ppv.molar_fractions[1])
                    x_bdg.append(bdg.molar_fractions[1])
                    x_fper.append(fper.molar_fractions[1])

                found_solution = True

            except:
                print('No solution found at P={:.3e} Pa, T={:.1f} K'.format(P,T))

            rock.set_state(P, T)
            pressure_list.append(P/1.e5) # pressure in bar
            temperature_list.append(T)

            if (found_solution):
                rock.debug_print()
                print(P, ' ', T, '\n')

                densities.append(rock.density)
                compressibilities.append(rock.isothermal_compressibility)
                s_velocities.append(rock.shear_wave_velocity)
                p_velocities.append(rock.p_wave_velocity)
                thermal_expansivities.append(rock.thermal_expansivity)
                heat_capacities.append(rock.heat_capacity_p/rock.molar_mass)
                enthalpies.append(rock.molar_enthalpy)

            else:
                densities.append(np.nan)
                compressibilities.append(np.nan)
                s_velocities.append(np.nan)
                p_velocities.append(np.nan)
                thermal_expansivities.append(np.nan)
                heat_capacities.append(np.nan)
                enthalpies.append(np.nan)

    # write to file:
    output_filename = "test_iron.txt"
    f = open(output_filename, 'wb')

    # write the header of the file:
    f.write("BurnMan\nComposition:")
    f.write(str(composition))
    f.write("\n\nT(K)\n   ")
    f.write(str(temperature_list[0]))
    f.write("\n   ")
    f.write(str(Tstep))
    f.write("\n         ")
    f.write(str(temperatures.size))
    f.write("\nP(bar)\n   ")
    f.write(str(pressure_list[0]))
    f.write("\n   ")
    f.write(str(pstep))
    f.write("\n         ")
    f.write(str(pressures.size))
    f.write("\n\n")

    header = "T(K)           P(bar)         rho,kg/m3      alpha,1/K      cp,J/K/kg      vp,km/s        vs,km/s        h,J/kg\n"
    f.write(header.encode('utf-8'))

    data = list(
        zip(temperature_list, pressure_list, densities, thermal_expansivities, heat_capacities, p_velocities, s_velocities, enthalpies))
    np.savetxt(f, data, fmt='%.10e', delimiter='\t')

    print("\nYour data has been saved as: ", output_filename)
    
    
