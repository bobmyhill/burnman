# Benchmarks for the solid solution class
import os.path, sys
sys.path.insert(1,os.path.abspath('../..'))

import numpy as np


print '# BurnMan - a lower mantle toolkit'
print '# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.'
print '# Released under GPL v2 or later.'
print ''
print '"""'
print 'DKS_2013'
print 'Liquids from de Koker and Stixrude (2013) FPMD simulations'
print '"""'
print ''
print 'from burnman.mineral import Mineral'
print 'from burnman.solidsolution import SolidSolution'
print 'from burnman.solutionmodel import *'
print ''
print '# Vector parsing for DKS liquid equation of state'
print 'def vector_to_array(a, Of, Otheta):'
print '    array=np.empty([Of+1, Otheta+1])'
print '    for i in range(Of+1):'
print '        for j in range(Otheta+1):'
print '            n=int((i+j)*((i+j)+1.)/2. + j)'
print '            array[i][j]=a[n]'
print '    return array'
print ''

filemin=[['dks_l', './helm_inp.siliq', 'SiO2_liquid'], 
         ['dks_l', './helm_inp.j11lq', 'MgSiO3_liquid'], 
         ['dks_l', './helm_inp.j12lq', 'MgSi2O5_liquid'],  
         ['dks_l', './helm_inp.j13lq', 'MgSi3O7_liquid'], 
         ['dks_l', './helm_inp.j15lq', 'MgSi5O11_liquid'],  
         ['dks_l', './helm_inp.j21lq', 'Mg2SiO4_liquid'], 
         ['dks_l', './helm_inp.j32lq', 'Mg3Si2O7_liquid'],  
         ['dks_l', './helm_inp.j51lq', 'Mg5SiO7_liquid'], 
         ['dks_l', './helm_inp.peliq', 'MgO_liquid'], 
         ]

for database, f, mineral in filemin:
    f = open(f, 'r')
    datalines = [ str.split(line.strip()) for idx, line in enumerate(f.read().split('\n')) if line.strip()]

    elements=map(int,datalines[5])
    composition=map(float,datalines[4])
    n_Si=0
    n_Mg=0 
    n_O=0
    for i, e in enumerate(elements):
        if e == 14:
            n_Si = composition[i]
        elif e == 12:
            n_Mg = composition[i]
        elif e == 8:
            n_O = composition[i]
        else:
            print 'Element not in MSO system'
            exit

    V_0, T_0, E_0, S_0 = map(float,datalines[datalines.index(['F_cl:']) + 1])
    m = map(float,datalines[datalines.index(['F_cl:']) + 5])[0]
    a = map(float,datalines[datalines.index(['F_cl:']) + 7])
    zeta_0, xi, Tel_0, eta, el_V_0 = \
        map(float,datalines[datalines.index(['F_el:']) + 1])
 
    n_a = int(datalines[datalines.index(['F_cl:']) + 6][0])
    O_theta = 2
    if n_a == 36:
        O_f = 5
    if n_a == 28:
        O_f = 4
    elif n_a == 21:
        O_f = 3
    else:
        print 'Array number not consistent with'
        print ' O_theta = 2 and O_f = 3-5 (21, 28, 36)'
        exit

    print 'class', mineral+'(Mineral):'
    print '    def __init__(self):'
    print '        self.params = {'
    print '            \'name\': \''+mineral+'\','


    print '            \'formula\': {\'Mg\':', n_Mg, ', \'Si\':', n_Si, ', \'O\':', n_O, '},'
    print '            \'equation_of_state\': \'dks_l\','

    print '            \'V_0\':', V_0*1e-6, ','
    print '            \'T_0\':', T_0, ','
    print '            \'E_0\':', E_0*1e3, ','
    print '            \'S_0\':', S_0, ','


    print '            \'O_theta\':', O_theta, ','
    print '            \'O_f\':', O_f,','
    print '            \'m\':', m, ','
    print '            \'a\':',  a, ','

    print '            \'zeta_0\':', zeta_0, ','
    print '            \'xi\':',xi, ','
    print '            \'Tel_0\':',Tel_0, ','
    print '            \'eta\':',eta, ','
    print '            \'el_V_0\':', el_V_0 * 1e-6 
    print '            }'
    print '        self.params[\'a\'] = vector_to_array(self.params[\'a\'], self.params[\'O_f\'], self.params[\'O_theta\'])*1e3 # [J/mol]'
    print '        Mineral.__init__(self)'

    print ''
    print ''
