from __future__ import absolute_import
import numpy as np

from model_parameters import *
from eos_and_averaging import thermodynamic_properties, average_solid_properties, average_melt_properties, average_composite_properties


pressures = np.linspace(100.e9, 150.e9, 11)
temperatures = np.linspace(2000., 5000., 7)

Pv, Tv = np.meshgrid(pressures, temperatures)

pressures = Pv.flatten()
temperatures = Tv.flatten()

full_pty_list=['gibbs', 'S', 'V', 'rho', 'alpha', 'beta_T', 'molar_C_p', 'C_p_per_kilogram']
derivative_pty_list=['V', 'rho', 'alpha', 'beta_T', 'molar_C_p', 'C_p_per_kilogram']


full_header = 'Pressure Temperature'
for pty in full_pty_list:
    full_header += ' ' + pty

derivative_header = 'Pressure Temperature'
for subsystem in ['SOLID', 'MELT', 'COMPOSITE']:
    derivative_header += ' ' + subsystem
    for pty in derivative_pty_list:
        derivative_header += ' ' + pty


# 1) Mg bridgmanite table -> mpv_properties.dat
# 2) Mixed solid and liquid table -> composite_properties.dat

phi = 0.2
mass_fraction_pv = 0.25
p_wus = 0.41
p_fpv = 0.42
p_feliq = 0.43

mpv_table = []
composite_table = []
for i in range(len(pressures)):
    pty = thermodynamic_properties(pressures[i], temperatures[i], mpv_params)
    mpv_table.append([pressures[i], temperatures[i]])
    mpv_table[-1].extend([pty[p] for p in full_pty_list])

    solid_pty = average_solid_properties(pressures[i], temperatures[i],
                                         p_fpv, p_wus, mass_fraction_pv)
    melt_pty = average_melt_properties(pressures[i], temperatures[i],
                                       p_feliq)
    composite_pty = average_composite_properties(pressures[i], temperatures[i],
                                                 p_fpv, p_wus, p_feliq,
                                                 mass_fraction_pv, phi)


    composite_table.append([pressures[i], temperatures[i]])
    composite_table[-1].extend([solid_pty[p] for p in derivative_pty_list])
    composite_table[-1].extend([melt_pty[p] for p in derivative_pty_list])
    composite_table[-1].extend([composite_pty[p] for p in derivative_pty_list])


np.savetxt('output_data/PT_tables/mpv_properties.dat', mpv_table, header=full_header)
np.savetxt('output_data/PT_tables/composite_properties.dat', composite_table, header=derivative_header)
