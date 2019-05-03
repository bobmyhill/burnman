from fitting_functions import equilibrium_order
from input_dataset import *

input_fractions = [0.11, 0.11, 0.2, 0.1, 0.48]
hpx_od.set_composition(input_fractions)
hpx_od.set_state(1.e9, 1000.)

equilibrium_order(hpx_od)

print('{0} -> {1}'.format(input_fractions,
                          [float('{0:.4f}'.format(f)) for f in hpx_od.molar_fractions]))

