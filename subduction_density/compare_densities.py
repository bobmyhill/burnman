# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

"""
example_perplex
---------------

This minimal example demonstrates how burnman can be used to read and interrogate
a PerpleX tab file (as produced by burnman/misc/create_burnman_readable_perplex_table.py 
It also demonstrates how we can smooth a given property on a given P-T grid.

*Uses:*

* :doc:`PerplexMaterial`
* :func:`burnman.material.Material.evaluate`
* :func:`burnman.tools.smooth_array`

*Demonstrates:*

* Use of PerplexMaterial
* Smoothing gridded properties


"""
import sys
import os
import numpy as np

sys.path.insert(1, os.path.abspath('..'))

import burnman
import matplotlib.pyplot as plt


from burnman.minerals.JH_2015 import *

from scipy.special import erfc

plt.style.use('ggplot')


def dimensionless_temperature_contrast(t_trench, t_sub, velocity, dip_degrees, depth):
    """
    Returns maximum relative temperature contrast between the slab and convecting mantle
    0 is no contrast, 1 is the (potential) temperature contrast between the surface and mantle
    """
    phi = velocity*np.sin(dip_degrees/180.*np.pi)*t_sub # thermal parameter
    f = phi/depth # nondimensionalised thermal parameter

    return erfc(np.sqrt(0.5/f*np.log(1.+f))) - erfc(np.sqrt(0.5*((1.+f)/f)*np.log(1. + f)))

# Thermal parameters
thermal_parameters = {'Aleutians': 2500.e3,
                      'Tonga': 17000.e3}

depths = np.linspace(1., 650.e3, 101)
temperature_mantle = 1553.15 + 0.45e-3*depths # Tp ~ 1280 C
temperature_ocean = 300. + 0.3e-3/5.*depths

temperatures = {}
for name, thermal_parameter in thermal_parameters.iteritems():
    T_contrast = dimensionless_temperature_contrast(t_trench=60.e6*np.pi*1.e7,
                                                    t_sub=thermal_parameter, # thermal parameter
                                                    velocity=1.,
                                                    dip_degrees=90.,
                                                    depth=depths)

    temperatures[name] = temperature_mantle + T_contrast*(temperature_ocean - temperature_mantle)
    #plt.plot(depths/1.e3, temperatures[name]-273.15, label=name)
    
#plt.plot(depths/1.e3, temperature_mantle-273.15, label='Mantle temperature')

#temperature_anderson = np.array([T if T>300. else 300. for T in burnman.geotherm.anderson(depths)])
#plt.plot(depths/1.e3, temperature_anderson-273.15, label='Mantle temperature')

#plt.xlabel('Depth (km)')
#plt.ylabel('Minimum temperature ($^{\\circ}$C)')
#plt.legend(loc='best')
#plt.show()
      
    

seismic_model = burnman.seismic.PREM()
pressures = seismic_model.evaluate(['pressure'], depths)[0] # only approximate, but it'll do.
pressures = np.array([P if P>1.1e5 else 1.1e5 for P in pressures])
                         


# Metastable basalt calculated by meemum using the Jennings and Holland database:
cpx = clinopyroxene([0.75802, 0.15676, 0., 0., 0., 0.03148, 0.05367, 0.00006]) # di, cfs, cats, crdi, cess, jd, cen, cfm
ol = olivine([0.62778, 0.37222]) # fo, fa
pl = plagioclase([0.65930, 0.34070]) # an, abh
opx = orthopyroxene([0.47495, 0.157, 0.30324, 0.03704, 0.02778, 0., 0.]) # en, fs, fm, odi, mgts, cren, mess

metastable_morb = burnman.Composite(phases=[cpx, ol, pl, opx],
                                fractions=[0.2937, 0.1875, 0.4594, 0.0594],
                                fraction_type='molar')
metastable_morb.name = 'metastable basalt'

print('Reading PerpleX tables...\r')
pyrolite=burnman.PerplexMaterial('pyrolite.out')
morb=burnman.PerplexMaterial('morb.out')
harzburgite=burnman.PerplexMaterial('harzburgite.out')
print('PerpleX tables successfully read')

pyrolite.name = 'pyrolite, mantle temperature'
morb.name = 'basalt'
harzburgite.name = 'harzburgite'

fig = plt.figure()
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]


rocks = [metastable_morb, morb, harzburgite]
for i, (name, thermal_parameter) in enumerate(thermal_parameters.iteritems()):

    ax[i+1].set_title(name)
    for rock in rocks:
        rock_rhos_max = rock.evaluate(['density'], pressures, temperatures[name])[0]
        rock_rhos_min = rock.evaluate(['density'], pressures, temperature_mantle)[0]
        ax[i+1].fill_between(depths/1.e3, rock_rhos_min/1.e3, rock_rhos_max/1.e3, label=rock.name)

    pyrolite_rhos = pyrolite.evaluate(['density'], pressures, temperature_mantle)[0]
    ax[i+1].plot(depths/1.e3, pyrolite_rhos/1.e3, linewidth=3., color='black', label=pyrolite.name)
    
    ax[0].plot(depths/1.e3, temperatures[name]-273.15, linewidth=2., label=name)

ax[0].plot(depths/1.e3, temperature_mantle-273.15, linewidth=3., color='black',  label='mantle temperature')

for i in range(3):
    ax[i].set_xlabel('Depth (km)')
    ax[i].legend(loc='best')
    ax[i].set_xlim(0, 650)

ax[0].set_ylim(0., 1600.)
ax[1].set_ylim(3., 4.4)
ax[2].set_ylim(3., 4.4)
ax[0].set_title('Temperature structure')
ax[0].set_ylabel('Temperature ($^{\\circ}$C)')
ax[1].set_ylabel('Density (kg/m$^3$)')
ax[2].set_ylabel('Density (kg/m$^3$)')

plt.show()

