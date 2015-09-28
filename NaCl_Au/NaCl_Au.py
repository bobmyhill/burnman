import os, sys, argparse, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import fsolve
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--Au_hkl', nargs='+', type=float, help='hkl for gold', required=True)
parser.add_argument('--NaCl_hkl', nargs='+', type=float, help='hkl for halite', required=True)
parser.add_argument('--Au_d', nargs='+', type=float, help='d spacing for gold (Angstrom)', required=True)
parser.add_argument('--NaCl_d', nargs='+', type=float, help='d spacing for halite (Angstrom)', required=True)
parser.add_argument('--Au_a0', nargs='+', type=float, help='Measured unit cell length a0 for gold (Angstrom)', required=True)
parser.add_argument('--NaCl_a0', nargs='+', type=float, help='Measured unit cell length a0 for halite (Angstrom)', required=True)


args = parser.parse_args()
atomic_masses=burnman.processchemistry.read_masses()
Z_NaCl = 4.
Z_Au = 4.
Atom3=1.e-30

class gold (burnman.Mineral):
    def __init__(self):
        formula='Au'
        formula = burnman.processchemistry.dictionarize_formula(formula)
        self.params = {
            'name': 'Gold',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': 0.0 ,
            'V_0': 67.850*burnman.constants.Avogadro/Z_Au*Atom3 ,
            'K_0': 167.e9 ,
            'Kprime_0': 5.0 ,
            'Debye_0': 170.0 ,
            'grueneisen_0': 2.97 ,
            'q_0': 1.0 ,
            'G_0': 40000000000.0 , # not fixed
            'Gprime_0': 1.1 , # not fixed
            'eta_s_0': 1.6 , # not fixed
            'n': sum(formula.values()),
            'molar_mass': burnman.processchemistry.formula_mass(formula, atomic_masses)}

        self.uncertainties = {
            'err_F_0': 4000.0 ,
            'err_V_0': 0.004 ,
            'err_K_0': 3.e9 ,
            'err_K_prime_0': 0.2 ,
            'err_Debye_0': 2.0 ,
            'err_grueneisen_0': 0.05 ,
            'err_q_0': 0.1 ,
            'err_G_0': 3000000000.0 , # not fixed
            'err_Gprime_0': 0.5 , # not fixed
            'err_eta_s_0': 1.0 } # not fixed
        burnman.Mineral.__init__(self)


Au = gold()
NaCl = burnman.minerals.HP_2011_ds62.hlt()

'''
# Check HP2011 EOS for NaCl (ref Birch 1986)
NaCl.set_state(297.10e8, 500.+273.15)
print NaCl.V/NaCl.params['V_0'], 'should be 0.65 (ca. 1 GPa difference)'
# ref Decker 1971
NaCl.set_state(110.44e8, 800.+273.15)
print (NaCl.V-NaCl.params['V_0'])/NaCl.params['V_0'], 'should be -0.1956 (ca. 0.1 GPa difference)'
NaCl.set_state(216.73e8, 800.+273.15)
print (NaCl.V-NaCl.params['V_0'])/NaCl.params['V_0'], 'should be -0.2950, (ca. 0.3 GPa difference)'
'''

'''
# Check Shim EoS for gold
Au.set_state(198.07e9, 300.)
print 1.-Au.V/Au.params['V_0'], 'should be 0.34'
Au.set_state(210.02e9, 2000.)
print 1.-Au.V/Au.params['V_0'], 'should be 0.34'
Au.set_state(217.17e9, 3000.)
print 1.-Au.V/Au.params['V_0'], 'should be 0.34'
'''


def calcP_T(PT, NaCl_V, Au_V):
    P, T = PT
    NaCl.set_state(P,T)
    Au.set_state(P,T)
    return [NaCl_V - NaCl.V, Au_V - Au.V]


guesses = [1.e9, 1000.15]

NaCl_V0_over_V0_measured = NaCl.params['V_0']/(np.power(args.NaCl_a0[0], 3.)/Z_NaCl*burnman.constants.Avogadro*Atom3)
Au_V0_over_V0_measured = Au.params['V_0']/(np.power(args.Au_a0[0], 3.)/Z_Au*burnman.constants.Avogadro*Atom3)

print np.power((NaCl.params['V_0']*Z_NaCl/burnman.constants.Avogadro/Atom3), 1./3.)
print np.power((Au.params['V_0']*Z_Au/burnman.constants.Avogadro/Atom3), 1./3.)


NaCl_V = np.power(args.NaCl_d[0]*np.linalg.norm(args.NaCl_hkl), 3.) \
         * NaCl_V0_over_V0_measured / Z_NaCl*burnman.constants.Avogadro*Atom3
Au_V   = np.power(args.Au_d[0]*np.linalg.norm(args.Au_hkl), 3.) \
         * Au_V0_over_V0_measured / Z_Au*burnman.constants.Avogadro*Atom3


P, T = fsolve(calcP_T, guesses, args=(NaCl_V, Au_V))
print "{0:.3f}".format(round(P/1.e9,3)), 'GPa', "{0:.1f}".format(round(T,1)), 'K ('+str("{0:.1f}".format(round(T-273.15,1))), 'C)'
