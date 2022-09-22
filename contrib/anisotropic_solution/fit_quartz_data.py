import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def tilt_regular(theta):
    # theta is the Si-O-Si bond angle (degrees)
    return np.degrees(np.arccos(np.sqrt(0.75 - np.cos(np.radians(theta))) - 1./(2.*np.sqrt(3.))))


def tilt_distorted(c_over_a, x, z):
    # c_over_a is the c/a ratio
    # x and z are the 6c positions (Oxygen atom positions (x,y,z)) of space group P3_121
    # typically x~0.41, z~0.22
    return np.degrees(np.arctan(2.*np.sqrt(3.)/9.*c_over_a*(6.*z - 1.)/x))


Antao = pd.read_csv('data/Antao_2016_quartz_structure_1bar.dat',
                    delim_whitespace=True, comment='#')
Bachheimer = pd.read_csv('data/Bachheimer_Dolino_1975_quartz_Q.dat',
                         delim_whitespace=True, comment='#')
Hazen = pd.read_csv('data/Hazen_et_al_1989_quartz_cell.dat',
                    delim_whitespace=True, comment='#')
Gronvold = pd.read_csv('data/Gronvold_et_al_1989_quartz_Cp.dat',
                       delim_whitespace=True, comment='#')
Richet = pd.read_csv('data/Richet_et_al_1992_quartz_Cp.dat',
                     delim_whitespace=True, comment='#')
Jorgensen = pd.read_csv('data/Jorgensen_1978_quartz_tilts_high_pressure.dat',
                        delim_whitespace=True, comment='#')
Scheidl = pd.read_csv('data/Scheidl_et_al_2016_quartz_cell.dat',
                      delim_whitespace=True, comment='#')
print(Jorgensen)

fig = plt.figure(figsize=(6, 12))
ax = [fig.add_subplot(4, 2, i) for i in range(1, 8)]

i = {'T_V': 0,
     'V_P': 1,
     'T_tilt': 2,
     'V_tilt': 3,
     'T_ac': 4,
     'V_ac': 5,
     'T_CP': 6}


ax[i['T_tilt']].scatter(Antao['T_K'], Antao['tilt'])
# ax[i['T_tilt']].scatter(Antao['T_K'], tilt_regular(Antao['Si-O-Si']))
ax[i['T_tilt']].scatter(Bachheimer['T_K'], Bachheimer['Q_norm']*Antao['tilt'][i['T_tilt']])

ax[i['V_tilt']].errorbar(Jorgensen['V'], Jorgensen['tilt'],
               yerr=Jorgensen['unc_tilt'], linestyle='None')
ax[i['V_tilt']].scatter(Jorgensen['V'], Jorgensen['tilt'])

#ax[i['V_tilt']].scatter(Antao['V'], Antao['tilt'])

Hazen_tilt = tilt_distorted(Hazen['c']/Hazen['a'], Hazen['x'], Hazen['z'])
ax[i['V_tilt']].scatter(Hazen['V'], Hazen_tilt)

ax[i['T_V']].errorbar(Antao['T_K'], Antao['V'],
                      yerr=Antao['unc_V'], linestyle='None')
ax[i['T_V']].scatter(Antao['T_K'], Antao['V'])


for axis in ['a', 'c']:
    ax[i['T_ac']].errorbar(Antao['T_K'], Antao[axis],
                           yerr=Antao[f'unc_{axis}'], linestyle='None')
    ax[i['T_ac']].scatter(Antao['T_K'], Antao[axis])

ax[i['T_CP']].scatter(Gronvold['T_K'], Gronvold['CP'])
ax[i['T_CP']].scatter(Richet['T_K'], Richet['CP'])


ax[i['V_P']].errorbar(Jorgensen['V'], Jorgensen['P_kbar']/10.,
                      xerr=Jorgensen['unc_V'], linestyle='None')
ax[i['V_P']].scatter(Jorgensen['V'], Jorgensen['P_kbar']/10.)

ax[i['V_P']].errorbar(Scheidl['V'], Scheidl['P_GPa'],
                      yerr=Scheidl['unc_P'], xerr=Scheidl['unc_V'], linestyle='None')
ax[i['V_P']].scatter(Scheidl['V'], Scheidl['P_GPa'])

ax[i['V_P']].errorbar(Hazen['V'], Hazen['P_GPa'],
                      xerr=Hazen['unc_V'], linestyle='None')
ax[i['V_P']].scatter(Hazen['V'], Hazen['P_GPa'])

ax[i['V_ac']].errorbar(Jorgensen['V'], Jorgensen['a'],
               yerr=Jorgensen['unc_a'], linestyle='None')
ax[i['V_ac']].scatter(Jorgensen['V'], Jorgensen['a'])

ax[i['V_ac']].errorbar(Scheidl['V'], Scheidl['a'],
               xerr=Scheidl['unc_V'], yerr=Scheidl['unc_a'], linestyle='None')
ax[i['V_ac']].scatter(Scheidl['V'], Scheidl['a'])

ax[i['V_ac']].errorbar(Hazen['V'], Hazen['a'],
               yerr=Hazen['unc_a'], linestyle='None')
ax[i['V_ac']].scatter(Hazen['V'], Hazen['a'])

ax[i['V_ac']].errorbar(Jorgensen['V'], Jorgensen['c'],
               yerr=Jorgensen['unc_c'], linestyle='None')
ax[i['V_ac']].scatter(Jorgensen['V'], Jorgensen['c'])

ax[i['V_ac']].errorbar(Scheidl['V'], Scheidl['c'],
               xerr=Scheidl['unc_V'], yerr=Scheidl['unc_c'], linestyle='None')
ax[i['V_ac']].scatter(Scheidl['V'], Scheidl['c'])

ax[i['V_ac']].errorbar(Hazen['V'], Hazen['c'],
               yerr=Hazen['unc_c'], linestyle='None')
ax[i['V_ac']].scatter(Hazen['V'], Hazen['c'])

for i in [0, 2, 4, 6]:
    ax[i].set_xlim(250., 1250.)
for i in [1, 3, 5]:
    ax[i].set_xlim(85., 115.)
plt.show()