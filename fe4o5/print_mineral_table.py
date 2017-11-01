""" Generates a text table with mineral properties. Run 'python table.py latex' to write a tex version of the table to mytable.tex """
from __future__ import absolute_import
from __future__ import print_function


import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
import inspect
from mineral_models_new import *
from burnman.minerals.Metal_Metal_oxides import *

from burnman import tools

def create_list(name, mineral):
    ownname = mineral.to_string().replace(
        "'", "").replace("burnman.minerals.", "")
    row = [name]
    for param in params:
        if param in mineral.params:
            if type(mineral.params[param]) is float:
                row.append('{0:.6e}'.format(mineral.params[param]).replace('0e','e').replace('0e','e').replace('0e','e').replace('0e','e').replace('0e','e').replace('0e','e').replace('0.e+00','0'))
            else:
                row.append(str(mineral.params[param]))
    return row
    
minerals = [Fe4O5, Fe5O6, wustite, defect_wustite, Mg2Fe2O5]
previous = [fcc_iron, high_mt]
metals = [Mo, MoO2, Re, ReO2]

for mineral_list in [minerals, metals, previous]:
    phasenames = [(x().name, x()) for x in mineral_list]
    params = ['H_0', 'S_0', 'V_0', 'K_0', 'Kprime_0', 'a_0', 'Cp', 'Cp', 'Cp', 'Cp']
    table = [['Name'] + params]
    tablel = []
    #sortedlist = sorted(phasenames, key=lambda x: x[0])
    
    
    for (name, p) in phasenames:
        p.set_state(1e9, 300)
        row = create_list(name, p)
        table.append(row)
        tablel.append(row)

    table = zip(*table)
    
    tools.pretty_print_table(table, use_tabs=False, latex=True)
    print('')
