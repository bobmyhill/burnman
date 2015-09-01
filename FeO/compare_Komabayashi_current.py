import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import Komabayashi_2014, Myhill_calibration_iron, Fe_Si_O
import numpy as np
from fitting_functions import *
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass

atomic_masses=read_masses()

Fe_fcc = Myhill_calibration_iron.fcc_iron_HP()
Fe_hcp = Myhill_calibration_iron.hcp_iron_HP()
Fe_liq = Myhill_calibration_iron.liquid_iron_HP()
FeO_sol = Fe_Si_O.FeO_solid()
FeO_liq = Fe_Si_O.FeO_liquid()

Fe_fcc_L = Myhill_calibration_iron.fcc_iron()
Fe_hcp_L = Myhill_calibration_iron.hcp_iron()
Fe_liq_L = Myhill_calibration_iron.liquid_iron()
FeO_sol_L = Fe_Si_O.FeO_solid_HP()
FeO_liq_L = Fe_Si_O.FeO_liquid_HP()

Fe_fcc_K = Komabayashi_2014.fcc_iron()
Fe_hcp_K = Komabayashi_2014.hcp_iron()
Fe_liq_K = Komabayashi_2014.liquid_iron()
FeO_sol_K = Komabayashi_2014.FeO_solid()
FeO_liq_K = Komabayashi_2014.FeO_liquid()



P = 100.e9
T = 2400.
Fe_hcp.set_state(P, T)
Fe_hcp_L.set_state(P, T)
Fe_hcp_K.set_state(P, T)
print Fe_hcp.V, Fe_hcp_L.V, Fe_hcp_K.V
print Fe_hcp.gibbs, Fe_hcp_L.gibbs, Fe_hcp_K.gibbs

Fe_fcc.set_state(P, T)
Fe_fcc_L.set_state(P, T)
Fe_fcc_K.set_state(P, T)
print Fe_fcc.V, Fe_fcc_L.V, Fe_fcc_K.V
print Fe_fcc.gibbs, Fe_fcc_L.gibbs, Fe_fcc_K.gibbs

Fe_liq.set_state(P, T)
Fe_liq_L.set_state(P, T)
Fe_liq_K.set_state(P, T)
print Fe_liq.V, Fe_liq_L.V, Fe_liq_K.V
print Fe_liq.gibbs, Fe_liq_L.gibbs, Fe_liq_K.gibbs


FeO_liq.set_state(P, T)
FeO_liq_L.set_state(P, T)
FeO_liq_K.set_state(P, T)
print FeO_liq.V, FeO_liq_L.V, FeO_liq_K.V
print FeO_liq.gibbs, FeO_liq_L.gibbs, FeO_liq_K.gibbs

FeO_sol.set_state(P, T)
FeO_sol_L.set_state(P, T)
FeO_sol_K.set_state(P, T)
print FeO_sol.V, FeO_sol_L.V, FeO_sol_K.V
print FeO_sol.gibbs, FeO_sol_L.gibbs, FeO_sol_K.gibbs
