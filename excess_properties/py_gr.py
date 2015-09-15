import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals

'''
Finally, the full subregular model is extremely flexible.
The compositional variability in excess properties at any given 
P and T is given by (Helffrich and Wood, 1989), as in the
standard subregular model.
The difference is in the construction of the intermediates. 
Instead of defining excess properties based on the endmembers and
constant H_ex, S_ex and V_ex, each binary interaction term is 
described via an intermediate compound
'''


# First let's define a simple symmetric garnet for comparison
class symmetric_garnet(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Symmetric pyrope-grossular garnet'
        self.type='symmetric'
        self.endmembers = [[minerals.HP_2011_ds62.py(), '[Mg]3[Al]2Si3O12'],
                           [minerals.HP_2011_ds62.gr(), '[Ca]3[Al]2Si3O12']]
        self.enthalpy_interaction=[[H_ex]]
        self.entropy_interaction=[[S_ex]]
        self.volume_interaction=[[V_ex]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)
        
        
# Now let's define a full subregular garnet model
# First, we need to create an intermediate compound
# We can do this by using experimental data augmented by some reasonable heuristics
        
# Here are the excess properties we want to use
H_ex = 15000.*3. # J/mol; making a symmetric version of Ganguly et al., 1996.
S_ex = 5.78*3. # m^3/mol; making a symmetric version of Ganguly et al., 1996.
V_ex = 4.e-6 # m^3/mol, see Du et al, 2015


# Now we define properties relative to the binary
py = minerals.HP_2011_ds62.py()
gr = minerals.HP_2011_ds62.gr()

V_pygr = (py.params['V_0'] + gr.params['V_0'])*0.5 + V_ex/4.

# Heuristics
KV_py = py.params['K_0']*py.params['V_0']
KV_gr = gr.params['K_0']*gr.params['V_0']

aK_py = py.params['K_0']*py.params['a_0']
aK_gr = gr.params['K_0']*gr.params['a_0']

KV_pygr = 0.5*(KV_py + KV_gr)
aK_pygr = 0.5*(aK_py + aK_gr)

K_pygr = KV_pygr / V_pygr


a_pygr = 0.5*(py.params['a_0']/py.params['V_0'] + gr.params['a_0']/gr.params['V_0']) * V_pygr

Kprime_pygr = V_pygr*2./(py.params['V_0']/(py.params['Kprime_0'] + 1.) \
                             + gr.params['V_0']/(gr.params['Kprime_0'] + 1.)) - 1.

Sconf = 2.*burnman.constants.gas_constant*0.5*3.*np.log(0.5)
S_pygr = (py.params['S_0'] + gr.params['S_0'])*0.5 - Sconf + S_ex/4.
H_pygr = (py.params['H_0'] + gr.params['H_0'])*0.5 + H_ex/4.

# Cp_scaling = ((py.params['S_0'] + gr.params['S_0'])*0.5 + S_ex/4)/((py.params['S_0'] + gr.params['S_0'])*0.5)
# Haselton and Westrum (1980) show that the excess entropy is primarily a result of a low temperature spike in Cp
# Therefore, at >=298.15 K, Cp is well approximated by a linear combination of pyrope and grossular 
Cp_scaling = 1. 

Cp_pygr = [(py.params['Cp'][0] + gr.params['Cp'][0])*0.5*Cp_scaling,
           (py.params['Cp'][1] + gr.params['Cp'][1])*0.5*Cp_scaling,
           (py.params['Cp'][2] + gr.params['Cp'][2])*0.5*Cp_scaling,
           (py.params['Cp'][3] + gr.params['Cp'][3])*0.5*Cp_scaling] 

# Overloaded heuristics
K_pygr = 168.e9
a_pygr = 2.3e-5

# Creating the intermediate endmember is just like creating any other endmember
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
atomic_masses=read_masses()

class pygr (burnman.Mineral):
    def __init__(self):
        formula='Mg1.5Ca1.5Al2.0Si3.0O12.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'py-gr intermediate',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': H_pygr,
            'S_0': S_pygr ,
            'V_0': V_pygr,
            'Cp': Cp_pygr ,
            'a_0': a_pygr, 
            'K_0': K_pygr,
            'Kprime_0': Kprime_pygr ,
            'Kdprime_0': -Kprime_pygr/K_pygr ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula, atomic_masses)}
        burnman.Mineral.__init__(self)
        
# Finally, here's the solid solution class
class full_garnet(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Subregular pyrope-almandine-grossular garnet'
        self.type='full_subregular'
        self.endmembers = [[minerals.HP_2011_ds62.py(), '[Mg]3[Al]2Si3O12'],
                           [minerals.HP_2011_ds62.gr(), '[Ca]3[Al]2Si3O12']]
        self.intermediates=[[[pygr(), pygr()]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)


# Now, let's see what the model looks like in practise
g6=full_garnet()
g7=symmetric_garnet()


# Check KV rule of thumb
g6.set_composition([0.5, 0.5])
g6.set_state(1.e5, 298.15)
K_T = g6.K_T
V = g6.V

KV_py = py.params['K_0']*py.params['V_0']
KV_gr = gr.params['K_0']*gr.params['V_0']

K_average = 0.5*(py.params['K_0'] + gr.params['K_0'])
KV_average = 0.5*(KV_py + KV_gr)
  
print (K_T - K_average)/(KV_average/V - K_average)


pressures = np.linspace(1.e5, 25.e9, 25)
py_volumes = np.empty_like(pressures)
gr_volumes = np.empty_like(pressures)
g6_volumes = np.empty_like(pressures)
g7_volumes = np.empty_like(pressures)
g6_excess_volumes = np.empty_like(pressures)
g7_excess_volumes = np.empty_like(pressures)

g6_excess_gibbs = np.empty_like(pressures)
g7_excess_gibbs = np.empty_like(pressures)

T = 298.15
molar_fractions=[0.5, 0.5]
for i,P in enumerate(pressures):
    py.set_state(P,T)
    py_volumes[i] = py.V
    gr.set_state(P,T)
    gr_volumes[i] = gr.V
    
    g6.set_composition(molar_fractions)
    g6.set_state(P,T)
    g6_volumes[i] = g6.V
    g6_excess_volumes[i] = g6.excess_volume
    g6_excess_gibbs[i] = g6.excess_gibbs
    
    g7.set_composition(molar_fractions)
    g7.set_state(P,T)
    g7_volumes[i] = g7.V
    g7_excess_volumes[i] = g7.excess_volume
    g7_excess_gibbs[i] = g7.excess_gibbs
    
    
plt.plot( pressures/1.e9, py_volumes*1.e6, 'r-', linewidth=1., label='Py volume')
plt.plot( pressures/1.e9, gr_volumes*1.e6, 'r-', linewidth=1., label='Gr volume')
plt.plot( pressures/1.e9, g6_volumes*1.e6, 'g-', linewidth=1., label='Py50Gr50 (full)')
plt.plot( pressures/1.e9, g7_volumes*1.e6, 'b-', linewidth=1., label='Py50Gr50 (simple)')
plt.title("Room temperature py-gr equations of state")
plt.ylabel("Volume (cm^3/mole)")
plt.xlabel("Pressure (GPa)")
plt.legend(loc='lower left')
plt.show()

plt.plot( pressures/1.e9, g6_excess_volumes*1.e6, 'g-', linewidth=1., label='Py50Gr50 (full)')
plt.plot( pressures/1.e9, g7_excess_volumes*1.e6, 'b-', linewidth=1., label='Py50Gr50 (simple)')
plt.title("Py-gr volume excesses")
plt.ylabel("Excess volume (cm^3/mole)")
plt.xlabel("Pressure (GPa)")
plt.legend(loc='lower left')
plt.show()


plt.plot( pressures/1.e9, g6_excess_gibbs*1.e-3, 'g-', linewidth=1., label='Py50Gr50 (full)')
plt.plot( pressures/1.e9, g7_excess_gibbs*1.e-3, 'b-', linewidth=1., label='Py50Gr50 (simple)')
plt.title("Py-gr gibbs excesses")
plt.ylabel("Excess gibbs (kJ/mole)")
plt.xlabel("Pressure (GPa)")
plt.legend(loc='lower left')
plt.show()

dT = 1.
comp = np.linspace(0.0, 1.0, 101)
g6_K_T = np.empty_like(comp)
g7_K_T = np.empty_like(comp)

g6_Cp = np.empty_like(comp)
g7_Cp = np.empty_like(comp)

g6_a = np.empty_like(comp)
g7_a = np.empty_like(comp)

for i,c in enumerate(comp):
    molar_fractions=[1.0-c, c]
    g6.set_composition(molar_fractions)
    g6.set_state(1.e5,298.15)
    g6_K_T[i] = g6.K_T
    g6_a[i] = g6.alpha
    g6_Cp[i] = g6.C_p
    
    g7.set_composition(molar_fractions)
    g7.set_state(1.e5,298.15)
    g7_K_T[i] = g7.K_T
    g7_a[i] = g7.alpha
    g7_Cp[i] = g7.C_p
    
plt.plot( comp, g6_K_T, 'g-', linewidth=1., label='Py-Gr (full)')
plt.plot( comp, g7_K_T, 'b-', linewidth=1., label='Py-Gr (simple)')
plt.title("Bulk modulus along the py-gr join")
plt.ylabel("Bulk modulus (GPa)")
plt.xlabel("Pyrope fraction")
plt.legend(loc='lower left')
plt.show()

plt.plot( comp, g6_Cp, 'g-', linewidth=1., label='Py-Gr (full)')
plt.plot( comp, g7_Cp, 'b-', linewidth=1., label='Py-Gr (simple)')
plt.title("Isobaric heat capacity along the py-gr join")
plt.ylabel("Cp")
plt.xlabel("Pyrope fraction")
plt.legend(loc='lower left')
plt.show()

plt.plot( comp, g6_a, 'g-', linewidth=1., label='Py-Gr (full)')
plt.plot( comp, g7_a, 'b-', linewidth=1., label='Py-Gr (simple)')
plt.title("Thermal expansion along the py-gr join")
plt.ylabel("alpha (/K)")
plt.xlabel("Pyrope fraction")
plt.legend(loc='lower left')
plt.show()

