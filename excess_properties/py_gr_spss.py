import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve, curve_fit
from make_intermediate import make_intermediate


# Read in data from Ganguly

f = open('data/Ganguly_CT_1996_GASP.dat')
growth_data = []
breakdown_data = []

growth_22 = []
growth_41 = []
growth_68 = []

breakdown_22 = []
breakdown_41 = []
breakdown_68 = []

datalines = [ line.strip().split() for idx, line in enumerate(f.read().split('\n')) if line.strip() ]
for content in datalines:
    if content[0] != '%':
        Mg_end = float(content[-2])
        if content[-1] == 'g':
            growth_data.append(map(float,content[1:8]))
            if Mg_end < 0.3:
                growth_22.append(map(float,content[1:8]))
            elif Mg_end < 0.5:
                growth_41.append(map(float,content[1:8]))
            else:
                growth_68.append(map(float,content[1:8]))
                
        if content[-1] == 'b':
            breakdown_data.append(map(float, content[1:8]))
            if Mg_end < 0.3:
                breakdown_22.append(map(float,content[1:8]))
            elif Mg_end < 0.5:
                breakdown_41.append(map(float,content[1:8]))
            else:
                breakdown_68.append(map(float,content[1:8]))
                
P_g, T_g, t_g, gr_start_g, mg_start_g, gr_end_g, mg_end_g = zip(*growth_data)
P_b, T_b, t_b, gr_start_b, mg_start_b, gr_end_b, mg_end_b = zip(*breakdown_data)
P_g = np.array(P_g)*1.e8
P_b = np.array(P_b)*1.e8


P_g2, T_g2, t_g2, gr_start_g2, mg_start_g2, gr_end_g2, mg_end_g2 = zip(*growth_22)
P_b2, T_b2, t_b2, gr_start_b2, mg_start_b2, gr_end_b2, mg_end_b2 = zip(*breakdown_22)
P_g2 = np.array(P_g2)*1.e8
P_b2 = np.array(P_b2)*1.e8
P_g4, T_g4, t_g4, gr_start_g4, mg_start_g4, gr_end_g4, mg_end_g4 = zip(*growth_41)
P_b4, T_b4, t_b4, gr_start_b4, mg_start_b4, gr_end_b4, mg_end_b4 = zip(*breakdown_41)
P_g4 = np.array(P_g4)*1.e8
P_b4 = np.array(P_b4)*1.e8
P_g6, T_g6, t_g6, gr_start_g6, mg_start_g6, gr_end_g6, mg_end_g6 = zip(*growth_68)
P_b6, T_b6, t_b6, gr_start_b6, mg_start_b6, gr_end_b6, mg_end_b6 = zip(*breakdown_68)
P_g6 = np.array(P_g6)*1.e8
P_b6 = np.array(P_b6)*1.e8




# First, let's define the endmembers
pyrope = minerals.HP_2011_ds62.py()
grossular = minerals.HP_2011_ds62.gr()
spessartine = minerals.HP_2011_ds62.spss()

# Rodehorst et al (2004) suggest mixing values for
# py-alm (~0 kJ/mol), sp-gr(14.4kJ/mol), alm-gr(~21 kJ/mol), py-gr (Wh=~32 kJ/mol)

# H_ex, S_ex, Sconf, V_ex, K_ex, a_ex
Sconf = -2.*burnman.constants.gas_constant*3.*0.5*np.log(0.5)

# Gr-Spss params (Dachs et al., 2014, Rodehorst et al., 2004)
grsp_params=[3.2e3*3./4., 3.8*3./4., Sconf, 0.58e-6/4., 0., 0.] # Dachs Wh
grsp_params=[14.4e3/4., 3.8*3./4., Sconf, 0.58e-6/4., 0., 0.] # Rodehorst Wh
grsp = make_intermediate(grossular, spessartine, grsp_params)()
spgr = make_intermediate(grossular, spessartine, grsp_params)()

# Py-Spss params (Wood et al., 1994)
pysp_params=[1.5e3*3./4., 0., Sconf, 0., 0., 0.] 
pysp = make_intermediate(pyrope, spessartine, pysp_params)()
sppy = make_intermediate(pyrope, spessartine, pysp_params)()

pygr_params=[8500., 1.186, Sconf, 7.83e-7, -10.93e9, -7.04e-7]
pygr = make_intermediate(pyrope, grossular, pygr_params)()

grpy_params=[8500., 1.186, Sconf, 7.83e-7, -10.93e9, -7.04e-7]
grpy = make_intermediate(pyrope, grossular, grpy_params)()

        
# Now, let's set up the solid solution model
class pyrope_grossular_spessartine(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Pyrope-grossular-spessartine garnet'
        self.type='full_subregular'
        self.endmembers = [[pyrope,  '[Mg]3Al2Si3O12'],
                           [grossular, '[Ca]3Al2Si3O12'],
                           [spessartine, '[Mn]3Al2Si3O12']]
        self.intermediates=[[[pygr, grpy], [pysp, sppy]],
                            [[grsp,spgr]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

garnet = pyrope_grossular_spessartine()




# GASP thermometer (Ganguly et al., 1996)
# We're looking for the composition of garnets in equilibrium with quartz, kyanite and anorthite
# The three endmembers fix the grossular activity
quartz = minerals.HP_2011_ds62.q()
anorthite = minerals.HP_2011_ds62.an()
kyanite = minerals.HP_2011_ds62.ky()

def GASP_equilibrium(p_gr, P, T, pyoverpyandsp):
    p_py = (1.-p_gr)*pyoverpyandsp
    p_sp = 1. - p_gr - p_py

    garnet.set_composition([p_py, p_gr[0], p_sp])
    garnet.set_state(P, T)

    quartz.set_state(P, T)
    kyanite.set_state(P, T)
    anorthite.set_state(P, T)

    ASP = [quartz, kyanite, anorthite]
    mu_gr = burnman.chemicalpotentials.chemical_potentials(ASP, [grossular.params['formula']])
    
    return garnet.partial_gibbs[1] - mu_gr


T = 1000. + 273.15
pyoverpyandsp = [0.23, 0.41, 0.68]
pressures = np.linspace(0.6e9, 2.e9, 21)
compositions = np.empty_like(pressures)
for Mgnumber in pyoverpyandsp:
    for i, P in enumerate(pressures):
        compositions[i] = fsolve(GASP_equilibrium, 0.2, args=(P, T, Mgnumber))[0]
    plt.plot(compositions, pressures/1.e9, label=str(Mgnumber))


plt.plot(gr_end_g6, P_g6/1.e9, marker='.', linestyle='None')
plt.plot(gr_end_b6, P_b6/1.e9, marker='o', linestyle='None')
plt.plot(gr_end_g2, P_g2/1.e9, marker='.', linestyle='None')
plt.plot(gr_end_b2, P_b2/1.e9, marker='o', linestyle='None')

plt.legend(loc='lower right')
plt.show()
