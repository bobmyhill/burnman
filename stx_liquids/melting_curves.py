import numpy as np
from scipy.optimize import fsolve, brentq, root
import matplotlib.pyplot as plt


import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from burnman import constants
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from silicate_models import *

def MS_liquidus_temperature(temperature, pressure, solid, n_cations_solid, SiO2_fraction_solid, SiO2_fraction):
    # Find liquidus curves from stoichiometric phases
    
    solid.set_state(pressure, temperature)
    FMS.molar_fractions = np.array([0.001, (1. - SiO2_fraction),  SiO2_fraction - 0.001])
    FMS.set_state(pressure, temperature)
    mu_phase_liq = ( FMS.partial_gibbs[1]*( 1. - SiO2_fraction_solid ) +
                     FMS.partial_gibbs[2]*SiO2_fraction_solid ) * n_cations_solid
    return solid.gibbs - mu_phase_liq


mpv = SLB_2011.mg_perovskite()
mpv.property_modifiers = [['linear', {'delta_E': -2.6e3, 'delta_S': 0., 'delta_V': 0.}]]

curves = [[per, 1., 0., 0.001, 0.40, 21],
          #[mpv, 2., 1./2., 0.35, 0.8, 21],
          [stv, 1., 1., 0.70, 0.999, 21]]

for P in [100.e9, 24.e9]:
    for curve in curves:
        solid, nc_solid, c_solid, X_min, X_max, n = curve
        X_SiO2 = np.linspace(X_min, X_max, n)
    
        temperatures = np.empty_like(X_SiO2)
        
    
        Tmin = 500.
        Tmax = 3680. + P/1.e9*60.
                
        for i, X in enumerate(X_SiO2):
            sol = brentq(MS_liquidus_temperature, Tmin, Tmax, args=(P, solid, nc_solid, c_solid, X), full_output=True)
            if sol[1].converged == True:
                temperatures[i] = sol[0]
                Tmin = temperatures[i] - 300.
                Tmax = temperatures[i] + 300.
                print X, temperatures[i]
            else:
                temperatures[i] = 2000.
        plt.plot(X_SiO2, temperatures)
plt.ylabel('Temperature (K)')
plt.xlabel('X SiO2')
plt.show()
