# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2019 by the BurnMan team, released under the GNU
# GPL v2 or later.


# This is a standalone program that converts the Holland and Powell data format
# into the standard burnman format (printed to file)
# It only outputs properties of solid endmembers - other endmembers are
# currently ignored.


import sys
import os.path
import pprint
from collections import OrderedDict
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman
from burnman.processchemistry import dictionarize_formula, formula_mass


if os.path.isfile('tc-ds633.txt') == False:
    print('This code requires the data file tc-ds633.txt.')
    print(
        'This file is bundled with the software package THERMOCALC, which can be found here:')
    print(
        'https://www.esc.cam.ac.uk/research/research-groups/research-projects/tim-hollands-software-pages/thermocalc')
    print('')
    print('Please download the file and place it in this directory.')
    exit()

# Components
components = ['Si', 'Ti', 'Al', 'Fe', 'Mg', 'Mn', 'Ca', 'Na',
              'K', 'O', 'H', 'C', 'Cl', 'e-', 'Ni', 'Zr', 'S', 'Cu', 'Cr']


class Endmember:

    def __init__(self, name, atoms, formula, sites, comp, H, S, V, Cp, a, k, flag, od):
        if flag != -1 and flag != -2 and k[0] > 0:
            formula = dictionarize_formula(formula)
            self.params = OrderedDict([('name', name),
                                       ('formula', formula),
                                       ('equation_of_state', 'hp_tmt'),
                                       ('H_0', round(H * 1e3, 10)),
                                       ('S_0', round(S * 1e3, 10)),
                                       ('V_0', round(V * 1e-5, 15)),
                                       ('Cp', [round(Cp[0] * 1e3, 10),
                                               round(Cp[1] * 1e3, 10),
                                               round(Cp[2] * 1e3, 10),
                                               round(Cp[3] * 1e3, 10)]),
                                       ('a_0', a),
                                       ('K_0', round(k[0] * 1e8, 10)),
                                       ('Kprime_0', k[1]),
                                       ('Kdprime_0', round(k[2] * 1e-8, 15)),
                                       ('n', sum(formula.values())),
                                       ('molar_mass', round(formula_mass(formula), 10))])
        if flag == 1:
            self.landau_hp = OrderedDict([('P_0', 1e5),
                                          ('T_0', 298.15),
                                          ('Tc_0', od[0]),
                                          ('S_D', round(od[1] * 1e3, 10)),
                                          ('V_D', round(od[2] * 1e-5, 10))])

        if flag == 2:
            self.bragg_williams = OrderedDict([('deltaH', round(od[0] * 1e3, 10)),
                                               ('deltaV', round(od[1] * 1e-5, 15)),
                                               ('Wh', round(od[2] * 1e3, 10)),
                                               ('Wv', round(od[3] * 1e-5, 15)),
                                               ('n', od[4]),
                                               ('factor', od[5])])




# Read dataset
with open('tc-ds633.txt', 'r') as f:
    ds = [line.split() for line in f]

def getmbr(ds, mbr):
    mbrarray = []
    for i in range(0, int(ds[0][0])):
        if ds[i * 4 + 3][0] == mbr:
            atoms = 0.0
            formula = ''
            for j in range(3, len(ds[i * 4 + 3]) - 1, 2):
                atoms += float(ds[i * 4 + 3][j])
                formula = formula + \
                    components[int(ds[i * 4 + 3][j - 1]) - 1] + str(
                        round(float(ds[i * 4 + 3][j]), 10))
            if mbr.endswith('L'):
                flag = -2
                od = [0]
            else:
                flag = int(ds[i * 4 + 6][4])
            endmember = Endmember(mbr, atoms, formula, int(ds[i * 4 + 3][1]),
                                  list(map(float, ds[i * 4 + 3][2:(len(ds[i * 4 + 3]) - 1)])),
                                  float(ds[i * 4 + 4][0]), float(ds[i * 4 + 4][1]),
                                  float(ds[i * 4 + 4][2]), map(float, ds[i * 4 + 5]),
                                  float(ds[i * 4 + 6][0]), list(map(float, ds[i * 4 + 6][1:4])),
                                  flag, list(map(float, ds[i * 4 + 6][5:])))
            return endmember

with open('HGP_2018_ds633.py', 'wb') as outfile:
    outfile.write('# This file is part of BurnMan - a thermoelastic and '
                  'thermodynamic toolkit for the Earth and Planetary Sciences\n'
                  '# Copyright (C) 2012 - 2019 by the BurnMan team, '
                  'released under the GNU \n# GPL v2 or later.\n\n\n'
                  '"""\n'
                  'HGP_2018 (ds-62)\n'
                  'Endmember minerals from Holland, Green and Powell (2018) and references therein\n'
                  'Dataset version 6.33\n'
                  'The values in this document are all in S.I. units,\n'
                  'unlike those in the original tc-ds633.txt\n'
                  'File autogenerated using HGP633data_to_burnman.py\n'
                  '"""\n\n'
                  'from ..mineral import Mineral\n'
                  'from ..processchemistry import dictionarize_formula, formula_mass\n\n')

    outfile.write('"""\n'
                  'ENDMEMBERS\n'
                  '"""\n\n')

    def pprint_ordered_dict(d, leading_string, extra_whitespace=0):
        whitespace = ' ' * (len(leading_string)+2+extra_whitespace)
        s = pprint.pformat(d)
        s = s.replace('), ', ',\n{0}\''.format(whitespace))
        s = s.replace('\', ', '\': ').replace(' \'(\'', '\'')
        s = s.replace('OrderedDict([(', leading_string+'{').replace(')])', '}')
        return s

    formula = '0'
    for i in range(int(ds[0][0])):
        mbr = ds[i * 4 + 3][0]
        M = getmbr(ds, mbr)
        if mbr == 'and':  # change silly abbreviation
            mbr = 'andalusite'

        # Print parameters
        if hasattr(M, 'params'):
            outfile.write('class {0} (Mineral):\n'.format(mbr)+
                          '    def __init__(self):\n')

            s = pprint_ordered_dict(M.params, leading_string = '        self.params = ')
            s = s.replace('000000.0', 'e6')
            outfile.write(s)
            outfile.write('\n')

            # Print property modifiers (if they exist)
            if hasattr(M, 'landau_hp'):
                outfile.write('        self.property_modifiers = [[')
                s = pprint_ordered_dict(M.landau_hp, leading_string = '\'landau_hp\', ', extra_whitespace = 36)
                outfile.write(s)
                outfile.write(']]\n')

            if hasattr(M, 'bragg_williams') and M.bragg_williams['factor'] > 0:
                outfile.write('        self.property_modifiers = [[')
                s = pprint_ordered_dict(M.bragg_williams, leading_string = '\'bragg_williams\', ', extra_whitespace = 36)
                outfile.write(s)
                outfile.write(']]\n')


            outfile.write('        Mineral.__init__(self)\n\n')



    outfile.write('def cov():\n'
                  '    \"\"\"\n'
                  '    A function which loads and returns the variance-covariance matrix of the\n'
                  '    zero-point energies of all the endmembers in the dataset.\n\n'
                  '    Returns\n'
                  '    -------\n'
                  '    cov : dictionary\n'
                  '        Dictionary keys are:\n'
                  '        - endmember_names: a list of endmember names, and\n'
                  '        - covariance_matrix: a 2D variance-covariance array for the\n'
                  '          endmember zero-point energies of formation\n'
                  '    \"\"\"\n\n'
                  '    from .HGP_2018_ds633_cov import cov\n'
                  '    return cov\n\n')

# Process uncertainties
with open('HGP_2018_ds633_cov.py', 'wb') as outfile:

    outfile.write('# This file is part of BurnMan - a thermoelastic and '
                  'thermodynamic toolkit for the Earth and Planetary Sciences\n'
                  '# Copyright (C) 2012 - 2019 by the BurnMan team, '
                  'released under the GNU \n# GPL v2 or later.\n\n\n'
                  '"""\n'
                  'HGP_2018 (ds-633) zero-point energy covariance matrix\n'
                  'Derived from Holland, Green and Powell (2018) and references therein\n'
                  'Dataset version 6.33\n'
                  'The values in this document are all in S.I. units,\n'
                  'unlike those in the original tc-ds633.txt\n'
                  'File autogenerated using HGP633data_to_burnman.py\n'
                  '"""\n\n'
                  'from numpy import array\n\n'
                  'cov = ')


    import numpy as np
    n_mbrs = int(ds[0][0])

    names = []
    for i in range(n_mbrs):
        names.append(ds[i*4+3][0])

    cov = []
    for i in range(n_mbrs*4+4, len(ds)-2):
        cov.extend(map(float, ds[i]))

    i_utr = np.triu_indices(n_mbrs)
    i_ltr = np.tril_indices(n_mbrs)
    M = np.zeros((n_mbrs, n_mbrs))

    M[i_utr] = cov[1:]
    M[i_ltr] = M.T[i_ltr]

    M = M*1.e6 # (kJ/mol)^2 -> (J/mol)^2

    d = {'endmember_names':names,
         'covariance_matrix':M}

    np.set_printoptions(threshold='nan')

    import pprint
    pp = pprint.PrettyPrinter(indent=0, width=160, depth=3, stream=outfile)
    pp.pprint(d)

    outfile.write('\n')