from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
# import matplotlib.image as mpimg
# from matplotlib import cm
# from scipy.optimize import minimize, fsolve

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.solutionbases import transform_solution_to_new_basis


from input_buffers import rhenium, rhenium_dioxide, molybdenum, molybdenum_dioxide
from load_inversion import load_inversion

mcmc_params, dataset, storage, special_constraints = load_inversion(show_correlations=False)

comps = {'Na': ('Na2O', 2., 1.),
         'Ca': ('CaO', 1., 1.),
         'Al': ('Al2O3', 2., 3.),
         'Mg': ('MgO', 1., 1.),
         'Si': ('SiO2', 1., 2.),
         'Fe': ('Fe', 1., 0.),
         'Re': ('Re', 1., 0.),
         'Mo': ('Mo', 1., 0.),
         'O': ('O2', 0., 2.)}


notes = ''

print('')
print('begin_makes')
print('')

for name, m in dataset['endmembers'].items():
    if isinstance(m, burnman.CombinedMineral):

        cnames = [list(dataset['endmembers'].keys())[list(dataset['endmembers'].values()).index(e[0])]
                  for e in m.mixture.endmembers]
        mm = [(m.mixture.molar_fractions[i], cnames[i]) for i in range(len(m.mixture.endmembers))]
        cstr = f'{name} = '
        for f, n in mm:
            cstr += f'{f} {n} + '
        cstr = cstr[:-2]

        nT = -m.property_modifiers[0][1]["delta_S"]
        nP = m.property_modifiers[0][1]["delta_V"]*1e5
        dstr = f'       DQF(J/mol) = {m.property_modifiers[0][1]["delta_E"]:.1f}'
        if nT >= 0.:
            dstr += f' + {nT}*T_K'
        else:
            dstr += f' - {-nT}*T_K'
        if nP >= 0.:
            dstr += f' + {nP}*p_bar'
        else:
            dstr += f' - {-nP}*p_bar'

        print(cstr)
        print(dstr)

print('rro = 1.0 reox - 1.0 re')
print('       DQF(J/mol) = 0')
print('')

print('mmo = 1.0 moox - 1.0 mo')
print('       DQF(J/mol) = 0')
print('')

print('')
print('end_makes')
print('')


for nm, m in dataset['endmembers'].items():
    name = nm
    if name == 'Re':
        name = 're'
    if name == 'ReO2':
        name = 'reox'

    if name == 'Mo':
        name = 'mo'
    if name == 'MoO2':
        name = 'moox'
    if not isinstance(m, burnman.CombinedMineral):

        try:
            if name != "O2":
                print(f'{name:6s}       EoS = 8 | H= {int(m.params["H_0"])}')
                form = ''
                for el in ['Na', 'Ca', 'Mg', 'Al', 'Si', 'Fe', 'Re', 'Mo', 'O']:
                    try:
                        m.params["formula"][el] = m.params["formula"][el] + 0.
                    except:
                        m.params["formula"][el] = 0.

                nfreeO = m.params["formula"]['O']
                for el in ['Na', 'Ca', 'Mg', 'Al', 'Si', 'Fe', 'Re', 'Mo']:
                    nfreeO -= m.params["formula"][el]*comps[el][2]/comps[el][1]
                    if m.params["formula"][el] > 0.:
                        form += f'{comps[el][0]}({m.params["formula"][el]/comps[el][1]})'

                if nfreeO > 1.e-5:
                    form += f'O2({nfreeO/2.})'

                print(form)
                print(f'GH = {m.params["H_0"] - 298.15*m.params["S_0"]:.1f} S0 = {m.params["S_0"]:.3f} V0 = {m.params["V_0"]*1.e5:.3f}')
                print(f'c1 = {m.params["Cp"][0]:.1f} c2 = {m.params["Cp"][1]:.6f} c3 = {m.params["Cp"][2]:.1f} c5 = {m.params["Cp"][3]:.1f}')
                print(f'b1 = {m.params["a_0"]:.3e} b5 = {m.params["T_einstein"]:.3f} b6 = {int(m.params["K_0"]/1.e5)} b7 = {m.params["Kdprime_0"]*1.e5:.3e} b8 = {m.params["Kprime_0"]}')

                if len(m.property_modifiers) > 0:
                    for pm in m.property_modifiers:
                        if pm[0] == 'bragg_williams':
                            print(f'transition = 1  type = 5  t1 = {int(pm[1]["deltaH"])}  t2 = {pm[1]["deltaV"]*1.e5}  t3 = {int(pm[1]["Wh"])}  t4 = {pm[1]["Wv"]*1.e5}  t5 = {pm[1]["n"]}  t6 = {pm[1]["factor"]:.5f}')
                        elif pm[0] == 'landau_hp':
                            print(f'transition = 1  type = 4  t1 = {int(pm[1]["Tc_0"])}  t2 = {pm[1]["S_D"]}  t3 = {pm[1]["V_D"]*1.e5}')
                        else:
                            raise Exception('property modifier type not yet implemented.')


                print(f'end')

                print('')
                print('')
        except:
            notes += f'\nPlease add {name} manually'


for name, m in dataset['solutions'].items():
    print(name)
    # ensure that endmember names are the same as given in the original dataset
    e_names = []
    for e in m.endmembers:
        if e[0] in dataset['endmembers'].values():
            e_names.append(list(dataset['endmembers'].keys())[list(dataset['endmembers'].values()).index(e[0])])
        else:
            e_names.append(list(dataset['combined_endmembers'].keys())[list(dataset['combined_endmembers'].values()).index(e[0])])


    #e_names = [list(dataset['endmembers'].keys())[list(dataset['endmembers'].values()).index(e[0])]
    #           if e[0] in dataset['endmembers'].values()
    #           else list(dataset['combined_endmembers'].keys())[list(dataset['combined_endmembers'].values()).index(e[0])]
    #           for e in m.endmembers]
    #e_names = [e[0].params['name'] for e in m.endmembers]

    for i, n0 in enumerate(e_names):
        for j, n1 in enumerate(e_names[i+1:]):
            if (m.energy_interaction[i][j] != 0.):

                dstr = f'w({n0} {n1}) {m.energy_interaction[i][j]:.1f}'

                if m.entropy_interaction is not None:
                    if np.abs(m.entropy_interaction[i][j]) > 1.e-4:
                        nT = -m.entropy_interaction[i][j]
                        if nT >= 0.:
                            dstr += f' + {nT:.3f}*T_K'
                        else:
                            dstr += f' - {-nT:.3f}*T_K'

                if m.volume_interaction is not None:
                    if np.abs(m.volume_interaction[i][j]) > 1.e-8:
                        nP = m.volume_interaction[i][j]*1.e5
                        if nP >= 0.:
                            dstr += f' + {nP:.3f}*p_bar'
                        else:
                            dstr += f' - {-nP:.3f}*p_bar'
                print(dstr)

print('')
print(notes)
