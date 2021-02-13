#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    # Reads aspect output and computes seismic velocities using burnman
    # Aspect output is assumed to be stored under output/#model_name/solution/
    # Creates and additional h5 file with the computed vs, vp, and density in the output directory as
    'output/'+model_name+'/solution/seismic_velocities_res'+str(resample)+'.h5'
    # Creates a plot with four subplots of temperature, melt_fraction, Vs and Vp saved as "plots_"+model_name+".pdf"

"""
import sys,os
# hack to allow scripts to be placed in subdirectories next to burnman:
sys.path.insert(1, os.path.abspath('../'))
sys.path.insert(1, os.path.abspath('../../../burnman/'))
sys.path.insert(1, os.path.abspath('../melt_model/'))


import glob
import ulvz_melt_model
from burnman import averaging_schemes
from burnman import minerals
import burnman
from burnman import Mineral

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import average_solid_melt
from model_parameters import *

'''
input
'''
# Model to load and plot
model_name = 'output-permeability-5e-12-melting-reduction-650-volume-1.7e-5-weakening-20-hdf5'
# Preset molar_volume value for this specific run
volume =1.7e-5

# Resample nodes
# For quick exploration, I resample every 10 nodes.
# I change this to 1 to make high quality plots, but computation takes a
# while...
resample = 1

'''
create melt endmember
'''
# Creating melt endmember for which the volume is set self-consistent to what is used in the geodynamical model

class Fe_endmember_melt(Mineral):
    """
    FeO-SiO2 endmember
    """
    def __init__(self):
        name = 'FeO-SiO2 endmember'
        self.params = fe_mantle_melt_params
        Mineral.__init__(self)

class Mg_endmember_melt(Mineral):
    """
    MgO-SiO2 endmember
    """
    def __init__(self):
        name = 'MgO-SiO2 endmember'
        self.params = mg_mantle_melt_params
        Mineral.__init__(self)

Fe_endmember_melt_mod = Fe_endmember_melt()
Fe_endmember_melt_mod.params['V_0'] = volume



'''
load aspect output
'''
# aspect files
print(glob.glob('hdf5/' + model_name + '/solution/mesh-*.h5'))
mesh_file = glob.glob('hdf5/' + model_name + '/solution/mesh-*.h5')[0]
solution_file = glob.glob(
    'hdf5/' +
    model_name +
    '/solution/solution-*.h5')[0]

# Load the mesh
mesh = h5.File(mesh_file, 'r')
nodes = np.array(mesh['nodes'])
solution = h5.File(solution_file, 'r')
print('keys in solution', list(solution.keys()))


# identify duplicate nodes - note that this is a bit sloppy because it
# presumes that the nodal values are also duplicated, which may not be
# true for DG fields
unique_nodes, unique_indices, unique_inverse, unique_counts = np.unique(
    nodes, axis=0, return_index=True, return_inverse=True, return_counts=True)

x = unique_nodes[:, 0]
y = unique_nodes[:, 1]

# load T and P
temperatures = np.array(solution['T'])[unique_indices][:, 0]
# 'p' goes negative, 'p_f' does not
pressures = np.array(solution['p_f'])[unique_indices][:, 0]

#viscosity and density
viscosity = np.array(solution['viscosity'])[unique_indices][:, 0]
density = np.array(solution['density'])[unique_indices][:, 0]

# load melt fraction
# porosity can go slightly negative
porosity = np.array(solution['porosity'])[unique_indices][:, 0]
melt_frac = np.array(solution['melt_fraction'])[
    unique_indices][:, 0]  # runs from 0-1

# load composition
bulk_composition = np.array(solution['bulk_composition'])[unique_indices][:, 0]
melt_fe = np.array(solution['molar_Fe_in_melt'])[unique_indices][:, 0]
solid_fe = np.array(solution['molar_Fe_in_solid'])[unique_indices][:, 0]


'''
Plot temperature and melt fraction
'''
# plot temperature
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.tricontourf(
    x / 1.e3,
    y / 1.e3,
    temperatures,
    100,
    cmap='hot',
    extend='both')
plt.colorbar()
#plt.tricontour(x/1.e3,y/1.e3,temperatures,levels=[ 3700,3900], colors='k', linewidth =1)
plt.ylabel('height above CMB (km)')
plt.xlabel('(km)')

plt.title('Temperature (K)')

# plot melt fraction
plt.subplot(2, 2, 2)
plot_val = np.linspace(0, .1, 30, endpoint=True)
pl = plt.tricontourf(
    x / 1.e3,
    y / 1.e3,
    melt_frac,
    plot_val,
    cmap='copper',
    vmin=0,
    vmax=0.1,
    extend='both')
pl.set_clim(0., 0.1)
plt.colorbar()
plt.ylabel('height above CMB (km)')
plt.xlabel('(km)')
plt.title(r'melt fraction')


###################
'''
Compute velocities
'''
c_mantle = [{'MgO': 0.581, 'SiO2': 0.419},
            {'FeO': 0.908, 'SiO2': 0.092}]  # molar_proportions


# Initiate solid solutions
mg_fe_perovskite = minerals.SLB_2011.mg_fe_perovskite()
mg_fe_periclase = minerals.SLB_2011.ferropericlase()





# Loop through all points
vs = []
rho = []
vp = []


for i in range(0, len(pressures), resample):
    print(i)
    print(pressures[i], temperatures[i], solid_fe[i], melt_fe[i], melt_frac[i])

    # construct molar compositions using boukare model
    molar_fractions_in_composite, molar_volumes, molar_masses = ulvz_melt_model.calculate_endmember_proportions_volumes_masses(
        pressures[i], temperatures[i], solid_fe[i], melt_fe[i], melt_frac[i], c_mantle)

    # initialize melt
    melt = burnman.CombinedMineral([Mg_endmember_melt(),Fe_endmember_melt()], [1-melt_fe[i], melt_fe[i]])

    if melt_frac[i] < 1.:
        # construct solid composition
        # fractions of bridgmanite vs periclase
        tot_pv = molar_fractions_in_composite['mpv'] + \
            molar_fractions_in_composite['fpv']
        tot_per = molar_fractions_in_composite['per'] + \
            molar_fractions_in_composite['wus']
        # Set solid solutions
        mg_fe_perovskite.set_composition(
            [
                molar_fractions_in_composite['mpv'] /
                tot_pv,
                molar_fractions_in_composite['fpv'] /
                tot_pv,
                0.])
        mg_fe_periclase.set_composition(
            [molar_fractions_in_composite['per'] / tot_per, molar_fractions_in_composite['wus'] / tot_per])

        # Set solid composite
        molar_frac_solid = tot_pv + tot_per
        solid = burnman.Composite([mg_fe_perovskite, mg_fe_periclase], [
                                  tot_pv / molar_frac_solid, tot_per / molar_frac_solid])

    if melt_frac[i] > 0.:
        # construct liquid solid solution
        molar_frac_melt = molar_fractions_in_composite['mg_melt'] + \
            molar_fractions_in_composite['fe_melt']
        frac_mg_melt = molar_fractions_in_composite['mg_melt'] / \
            molar_frac_melt
        frac_fe_melt = molar_fractions_in_composite['fe_melt'] / \
            molar_frac_melt

        # set melt composite
        #melt.set_composition([frac_fe_melt, frac_mg_melt, 0.])

    # set composite including melt and solid
    if melt_frac[i] == 1.:
        mix = melt
    elif melt_frac[i] == 0.:
        mix = solid
    else:
        mix = burnman.Composite(
            [solid, melt], [molar_frac_solid, molar_frac_melt])
        # choose averaging scheme between melt and solid
        # currently set to a linear approximation for a dihedral angle of 20
        # and a poisson ratio of 0.25 (roughly fitted to Figure 2 in Takei
        # 2002).
        mix.set_averaging_scheme(
            average_solid_melt.contiguity_model_linear_approximation())

    # evaluate velocities and density
    [vs_i, vp_i, rho_i] = mix.evaluate(
        ['v_s', 'v_p', 'density'], pressures[i], temperatures[i])

    vs.append(vs_i)
    rho.append(rho_i)
    vp.append(vp_i)

print(np.max(vs), np.min(vs))

dvs = (np.array(vs) / np.max(vs) - 1.) * 100.
rho = np.array(rho)
dvp = (np.array(vp) / np.max(vp) - 1.) * 100.

plot_val = np.linspace(-20, 0, 30, endpoint=True)


'''
Plot Vs and Vp
'''
# Plot  shear wave velocity
plt.subplot(2, 2, 3)
pl = plt.tricontourf(x[::resample] /
                     1.e3, y[::resample] /
                     1.e3, dvs, plot_val, cmap='OrRd_r', vmin=-
                     20., vmax=0, extend='both')
bar = plt.colorbar()
pl.set_clim(-20., 0)
#plt.tricontour(x[::resample]/1.e3,y[::resample]/1.e3,dvs, levels= [-20, -10], colours='r')
plt.ylabel('height above CMB (km)')
plt.xlabel('(km)')

plt.title(r'computed dVs ')

# Plot P wave velocity
plt.subplot(2, 2, 4)
pl = plt.tricontourf(x[::resample] /
                     1.e3, y[::resample] /
                     1.e3, dvp, plot_val, cmap='OrRd_r', vmin=-
                     20., vmax=0., extend='both')
bar = plt.colorbar()

plt.ylabel('height above CMB (km)')
plt.xlabel('(km)')
plt.title(r'computed dVp')

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.ylim([0, 50])
    plt.xlim([200, 400])
    plt.gca().set_aspect('equal', adjustable='box')

plt.savefig("plots_" + model_name + ".pdf")
plt.show()

'''
write out velocities and densities
'''
# Save computed velocities and density
if resample == 1:
    outfile = 'hdf5/' + model_name + '/solution/seismic_velocities_correcteddensity.h5'
else:
    outfile = 'hdf5/' + model_name + \
        '/solution/seismic_velocities_res' + str(resample) + '_correcteddensity.h5'
with h5.File(outfile, "w") as f:
    f.create_dataset("x", data=x[::resample])
    f.create_dataset("y", data=y[::resample])
    f.create_dataset("vs", data=vs)
    f.create_dataset("vp", data=vp)
    f.create_dataset("rho", data=rho)
####
