# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.
"""
example_build_planet
--------------------

For Earth we have well-constrained one-dimensional density models.  This allows us to
calculate pressure as a function of depth.  Furthermore, petrologic data and assumptions
regarding the convective state of the planet allow us to estimate the temperature.

For planets other than Earth we have much less information, and in particular we
know almost nothing about the pressure and temperature in the interior.  Instead, we tend
to have measurements of things like mass, radius, and moment-of-inertia.  We would like
to be able to make a model of the planet's interior that is consistent with those
measurements.

However, there is a difficulty with this.  In order to know the density of the planetary
material, we need to know the pressure and temperature.  In order to know the pressure,
we need to know the gravity profile.  And in order to the the gravity profile, we need
to know the density.  This is a nonlinear problem which requires us to iterate to find
a self-consistent solution.

This example allows the user to define layers of planets of known outer radius and self-
consistently solve for the density, pressure and gravity profiles. The calculation will
iterate until the difference between central pressure calculations are less than 1e-5.
The planet class in BurnMan (../burnman/planet.py) allows users to call multiple
properties of the model planet after calculations, such as the mass of an individual layer,
the total mass of the planet and the moment if inertia. See planets.py for information
on each of the parameters which can be called.

*Uses:*

* :doc:`mineral_database`
* :class:`burnman.planet.Planet`
* :class: `burnman.layer.Layer`

*Demonstrates:*

* setting up a planet
* computing its self-consistent state
* computing various parameters for the planet
* seismic comparison
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import matplotlib.pyplot as plt
import numpy as np

import burnman
from burnman import planet
import burnman.mineral_helpers as helpers

if __name__ == "__main__":



    # FIRST: we must define the composition of the planet as individual layers.
    # A layer is defined by 4 parameters: Name, min_depth, max_depth,and number of slices within the layer.
    # Separately the composition and the temperature_mode need to set.
    radius_planet = 6371.e3
    # inner_core
    inner_core = planet.Planet.Layer("inner core", radii = np.linspace(0,1220.e3,10))
    inner_core.set_material(burnman.minerals.other.Fe_Dewaele())
    
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will set the temperature at 300 K.
    inner_core.set_temperature_mode( 'user-defined',
        300.*np.ones_like(inner_core.radii))

    # outer_core
    outer_core = planet.Planet.Layer( "outer core", radii = np.linspace(1220.e3,3480.e3,10))
    outer_core.set_material(burnman.minerals.other.Liquid_Fe_Anderson())
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will define the temperature at 300 K.
    outer_core.set_temperature_mode('user-defined',
        300.*np.ones_like(outer_core.radii))

    # Next the Mantle.
    lower_mantle = planet.Planet.Layer( "lower_mantle", radii = np.linspace(3480.e3, 5711.e3, 10))
    lower_mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite())
    lower_mantle.set_temperature_mode('adiabat')
    upper_mantle = planet.Planet.Layer( "upper_mantle", radii = np.linspace(5711.e3, 6371e3, 10))
    upper_mantle.set_material(burnman.minerals.SLB_2011.forsterite())
    upper_mantle.set_temperature_mode('adiabat', temperature_top = 1200.)



    # Now we calculate the planet.
    Plan = burnman.Planet('earth_like',
                          [inner_core, outer_core, lower_mantle, upper_mantle], verbose=True)
    print(Plan)
    
    # Here we compute its state. Go BurnMan Go!
    # (If we were to change composition of one of the layers, we would have to
    # recompute the state)
    Plan.make()

    # Now we output the mass of the planet and moment of inertia
    print()
    print("mass/Earth= %.3f, moment of inertia= %.3f" %
          (Plan.mass / 5.97e24, Plan.moment_of_inertia_factor))

    # And here's the mass of the individual layers:
    for layer in Plan.layers:
        print("%s mass fraction of planet %.3f" %
              (layer.name, layer.mass / Plan.mass))
    print()


    # Let's get PREM to compare everything to as we are trying
    # to imitate Earth
    prem = burnman.seismic.PREM()
    premradii = 6371.e3 - prem.internal_depth_list()
    [premdensity, prempressure, premgravity,premvs,premvp] = prem.evaluate(
        ['density', 'pressure', 'gravity', 'v_s','v_p'])

    # Now let's plot everything up

    # Do you like pretty, but slow plotting figures? Try Burnman's patented pretty_plot package.
    # otherwise this line can be commented out
    #burnman.tools.pretty_plot()

    # Come up with axes for the final plot
    figure = plt.figure(figsize=(11, 7))
    plt.subplot(111, projection = 'polar')
    
    angels = np.linspace(0.,2.*np.pi,360)
    xx,yy =np.meshgrid(angels, premradii/1.3)
    xx,zz =np.meshgrid(angels, premdensity/1.3)
    plt.pcolor(xx,yy,zz, cmap='cubehelix_r')
    plt.gca().set_xticklabels('')
    plt.gca().set_yticklabels('')
    #plt.colorbar()
    plt.savefig('logotest.png')
    plt.show()
