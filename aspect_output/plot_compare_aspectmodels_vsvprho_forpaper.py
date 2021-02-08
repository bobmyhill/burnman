#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    # Reads aspect output and/or computed seismic velocities for up to four models and plots these
    # Aspect output file and h5 file with computed seismic velocities (computed using compute_velocities_aspectoutput.py) are expected in output/#model_name/solution/
    # Figures are saved as 'all_models_'+value_to_plot+'.pdf'
"""

import sys,os

# hack to allow scripts to be placed in subdirectories next to burnman:
sys.path.insert(1, os.path.abspath('../'))
sys.path.insert(1, os.path.abspath('../boukare/'))

import glob
import ulvz_melt_model
from burnman import averaging_schemes
from burnman import minerals
import burnman
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import average_solid_melt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from matplotlib import rc


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

color_array = plt.get_cmap('binary_r')(range(2))

# change alpha values
color_array[:,-1] = np.linspace(1.0,0.0,2)
print(color_array)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='binary_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)



'''
input
'''
# Models to plot
model_names = [
    'output-permeability-5e-12-melting-reduction-650-volume-1.6e-5-weakening-20-hdf5',
    'output-permeability-5e-12-melting-reduction-650-volume-1.7e-5-weakening-20-hdf5',
    'output-permeability-0-melting-reduction-650-volume-reference-weakening-20-hdf5',
    'output-permeability-5e-12-melting-reduction-650-volume-1.8e-5-weakening-20-hdf5',
    #'output-permeability-5e-12-melting-reduction-650-volume-1.9e-5-weakening-20-hdf5',
    'output-permeability-5e-12-melting-reduction-650-volume-2e-5-weakening-20-hdf5']
    
titles = [
    r'$\mathrm{\Delta \rho: 260 \  kg/m^3}$',
    r'$\mathrm{\Delta \rho: 60 \ kg/m^3}$',
    r'$\mathrm{no \ percolation}$',
    r'$\mathrm{\Delta \rho: -100 \ kg/m^3}$',
    r'$\mathrm{\Delta \rho: -400 \ kg/m^3}$']

# Plot value from aspect output across model
plot_aspect_output = False
value_to_plot = 'melt'  # implemented are 'temperature', 'melt', 'solid_fe'

# Plot computed Vs across models
# This plot is used in the paper without the density column
plot_seismic_velocity = True


'''
'''

fig, ax = plt.subplots(figsize=(15,12))


if plot_aspect_output:
    for m, model_name in enumerate(model_names):
        
        # Define mesh and solution file
        mesh_file = glob.glob(
            'hdf5/' +
            model_name +
            '/solution/mesh-*.h5')[0]

            

        # Load the mesh
        mesh = h5.File(mesh_file, 'r')
        nodes = np.array(mesh['nodes'])

        # Load aspct output
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
        bulk_composition = np.array(solution['bulk_composition'])[
            unique_indices][:, 0]
        melt_fe = np.array(solution['molar_Fe_in_melt'])[unique_indices][:, 0]
        solid_fe = np.array(solution['molar_Fe_in_solid'])[
            unique_indices][:, 0]

        if value_to_plot == 'temperature':
            # plot temperature
            plt.subplot(5,1, m + 1)
            plt.tricontourf(
                x / 1.e3,
                y / 1.e3,
                temperatures,
                100,
                cmap='hot',
                extend='both')
            plt.colorbar()
            #plt.tricontour(x/1.e3,y/1.e3,temperatures,levels=[ 0.92*3900,3900], colors='k')
            plt.ylabel('height above CMB (km)')
            plt.xlabel('(km)')

        if value_to_plot == 'melt':
            # plot melt
            plt.subplot(5,1, m + 1)
            plot_val = np.linspace(0, .1, 21, endpoint=True)
            pl = plt.tricontourf(
                x / 1.e3,
                y / 1.e3,
                melt_frac,
                plot_val,
                cmap='copper',
                vmin=0,
                vmax=0.1,
                extend='both')
            plt.colorbar()
            #plt.tricontour(x/1.e3,y/1.e3,temperatures,levels=[ 0.92*3900,3900], colors='k')
            plt.ylabel('height above CMB (km)')
            plt.xlabel('(km)')

        if value_to_plot == 'solid_fe':
            # plot iron content in nsolid
            plt.subplot(5,1, m + 1)
            plot_val = np.linspace(0, .2, 21, endpoint=True)
            pl = plt.tricontourf(
                x / 1.e3,
                y / 1.e3,
                solid_fe,
                plot_val,
                cmap='copper_r',
                vmin=0,
                vmax=0.2,
                extend='both')
            plt.colorbar()
            #plt.tricontour(x/1.e3,y/1.e3,temperatures,levels=[ 0.92*3900,3900], colors='k')
            plt.ylabel('height above CMB (km)')
            plt.xlabel('(km)')

        plt.gca().set_aspect('equal', adjustable='box')
        

    #plt.savefig('all_models_' + value_to_plot + '.pdf')
    #plt.show()


if plot_seismic_velocity:

    for m, model_name in enumerate(model_names):
        print(m,model_name)
        # Load and plot seismic velocities
        #plt.figure(figsize=(10, 6))

        solution_file = glob.glob(
                'hdf5/' +
                model_name +
                '/solution/solution-*.h5')[0]
        if m==2:
        
            vel_file = glob.glob(
                'hdf5/' +
                model_name +
                '/solution/seismic_velocities.h5')[0]
        else:
            vel_file = glob.glob(
                'hdf5/' +
                model_name +
                '/solution/seismic_velocities*sity.h5')[0]
                
        vels = h5.File(vel_file, 'r')
        vs = np.array(vels['vs'])
        x = np.array(vels['x'])
        y = np.array(vels['y'])
        # dVs/Vs with reference to 7.1 km/s
        dvs = (np.array(vs) / 7100 - 1.) * 100.
        
        vp = np.array(vels['vp'])
        #dVp/Vp with reference to 13.9 km/s
        dvp = (np.array(vp) / 13900 - 1.) * 100.
        
        rho = np.array(vels['rho'])
        #drho/rho with reference to 5.485 kg/m^3
        drho = (np.array(rho) / 5485 - 1.) * 100.

        plot_val = np.linspace(-40, 0, 41, endpoint=True)
        print(np.max(vs), np.max(vp), np.min(dvs), np.min(dvp))
        
        # Plot velocity
        ax = plt.subplot(5,3, m*3 + 1)
    
        pl1 = plt.tricontourf(
            x / 1.e3,
            y / 1.e3,
            dvs,
            plot_val,
            cmap='OrRd_r',
            vmin=-40.,
            vmax=0,
            extend='both')

        pl1.set_clim(-40., 0)
        pl2 = plt.tricontour(
        x / 1.e3,
        y / 1.e3,
        dvs,
        [-20.,],
        colors='r',
        linewidths = 1,
        linestyles = 'dashed',
        extend='both')
        pl3 = plt.tricontourf(
            x / 1.e3,
            y / 1.e3,
            dvs,
            [-100.,-99.9],
            cmap='binary_alpha',
            vmin=-100.,
            vmax=0,
            extend='both')
        
        plt.xlim([250,350])
        plt.ylim([0,50])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        #plt.text(.02,0.88,titles[m],fontsize=12, transform=ax.transAxes)
        plt.ylabel(titles[m], fontsize = 14)
        if m == 0:
            plt.title('S wave velocity', fontsize =14)
            ax.annotate('', xy=(-0.1,1.0), xycoords='axes fraction', xytext=(-.1,-4.5),
            arrowprops=dict(arrowstyle="<-", color='k'))
            plt.text(0.095, 0.4, 'Decreasing melt density', fontsize=14, transform=plt.gcf().transFigure, rotation=90)
        if m==4:
                #divider = make_axes_locatable(ax)
                #cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
                #fig.add_axes(cax)
                #fig.colorbar(pl1, cax=cax, orientation="horizontal")
                fig.subplots_adjust(bottom=0.1)
                cbar_ax = fig.add_axes([0.13, 0.08, 0.22, 0.01])
                cb = fig.colorbar(pl1, cax=cbar_ax, orientation="horizontal")
                cb.ax.set_xlabel('relative S wave velocity (\%)', fontsize =14)

                
        # P wave velocity
        ax = plt.subplot(5,3, m*3 + 2)
        
        pl1 = plt.tricontourf(
            x / 1.e3,
            y / 1.e3,
            dvp,
            plot_val,
            cmap='OrRd_r',
            vmin=-40.,
            vmax=0,
            extend='both')

        pl1.set_clim(-40., 0)
        pl2 = plt.tricontour(
        x / 1.e3,
        y / 1.e3,
        dvp,
        [-20.,],
        colors='r',
        linewidths = 1,
        linestyles = 'dashed',
        extend='both')

        
        plt.xlim([250,350])
        plt.ylim([0,50])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        #plt.text(.02,0.88,titles[m],fontsize=12, transform=ax.transAxes)
        if m==0:
            plt.title('P wave velocity', fontsize =14)
        if m==4:
                #divider = make_axes_locatable(ax)
                #cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
                #fig.add_axes(cax)
                #fig.colorbar(pl1, cax=cax, orientation="horizontal")
                #fig.subplots_adjust(bottom=0.1)
                cbar_ax = fig.add_axes([0.405, 0.08, 0.22, 0.01])
                cb = fig.colorbar(pl1, cax=cbar_ax, orientation="horizontal")
                cb.ax.set_xlabel('relative P wave velocity (\%)', fontsize =14)
        
                
    #
    #density - not shown in paper
        
        ax = plt.subplot(5,3, m*3 + 3)
        plot_val = np.linspace(-10, 10, 40, endpoint=True)
        pl1 = plt.tricontourf(
            x / 1.e3,
            y / 1.e3,
            drho,
            plot_val,
            cmap='seismic_r',
            vmin=-10.,
            vmax=10.,
            extend='both')

        pl1.set_clim(-10., 10.)

        plt.xlim([250,350])
        plt.ylim([0,50])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        plt.gca().set_aspect('equal', adjustable='box')
        #plt.text(.02,0.88,titles[m],fontsize=12, transform=ax.transAxes)
        if m==0:
            plt.title('density', fontsize =14)
        if m==4:
                #divider = make_axes_locatable(ax)
                #cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
                #fig.add_axes(cax)
                #fig.colorbar(pl1, cax=cax, orientation="horizontal")
                #fig.subplots_adjust(bottom=0.1)
                cbar_ax = fig.add_axes([0.68, 0.08, 0.22, 0.01])
                cb = fig.colorbar(pl1, cax=cbar_ax, orientation="horizontal", ticks = [-10,-5,0,5,10])
                cb.ax.set_xlabel('relative density (\%)', fontsize =14)
                
                cb.ax.set_xticklabels([-10,-5,0,5,10])
        

    

    plt.savefig("all_models_vsvp.pdf")
    plt.show()


    ####
