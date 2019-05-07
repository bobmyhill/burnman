from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import minimize, fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.solutionbases import transform_solution_to_new_basis
from input_buffers import rhenium, rhenium_dioxide, molybdenum, molybdenum_dioxide

#########################
# ENDMEMBER DEFINITIONS #
#########################

# Buffer phases
Re = rhenium()
ReO2 = rhenium_dioxide()
Mo = molybdenum()
MoO2 = molybdenum_dioxide()
O2 = burnman.minerals.HP_2011_fluids.O2()

# Iron endmembers
bcc_iron = burnman.minerals.SE_2015.bcc_iron()
fcc_iron = burnman.minerals.SE_2015.fcc_iron()
hcp_iron = burnman.minerals.SE_2015.hcp_iron()

# Oxides
cor = burnman.minerals.HP_2011_ds62.cor()
hem = burnman.minerals.HP_2011_ds62.hem()

# MgO and FeO
per = burnman.minerals.HHPH_2013.per()
wus = burnman.minerals.HHPH_2013.fper()

# Olivine endmembers
fo = burnman.minerals.HHPH_2013.fo()
fa = burnman.minerals.HHPH_2013.fa()

# Wadsleyite endmembers
mwd = burnman.minerals.HHPH_2013.mwd()
fwd = burnman.minerals.HHPH_2013.fwd()

# Spinel endmembers
mrw = burnman.minerals.HHPH_2013.mrw()
frw = burnman.minerals.HHPH_2013.frw()
sp = burnman.minerals.HP_2011_ds62.sp()
herc = burnman.minerals.HP_2011_ds62.herc()
mt = burnman.minerals.HP_2011_ds62.mt()

# Clinopyroxene endmembers
di = burnman.minerals.HP_2011_ds62.di()
hed = burnman.minerals.HP_2011_ds62.hed()
cen = burnman.minerals.HP_2011_ds62.cen()
cats = burnman.minerals.HP_2011_ds62.cats()
jd = burnman.minerals.HP_2011_ds62.jd()
aeg = burnman.minerals.HP_2011_ds62.acm() # aegirine is also known as acmite

cfs = burnman.minerals.HP_2011_ds62.fs()
cfs.params['name'] = 'cfs'
cfs.name = 'cfs'
cfs.params['H_0'] += 2.1e3
cfs.params['S_0'] += 2. # N.B: implies fs->cfs transition at ~1000 K at room pressure!
cfs.params['V_0'] += 0.045e-5 # Clapeyron slope of 225 K/GPa

# Orthopyroxene endmembers
oen = burnman.minerals.HHPH_2013.en()
ofs = burnman.minerals.HHPH_2013.fs()
mgts = burnman.minerals.HP_2011_ds62.mgts()
odi = burnman.minerals.HP_2011_ds62.di()
odi.params['H_0'] += -0.1e3
odi.params['S_0'] += -0.211
odi.params['V_0'] += 0.005e-5

ofm = burnman.CombinedMineral([oen, ofs], [0.5, 0.5], [-6.6e3, 0., 0.], name='ofm')


# High pressure clinopyroxene endmembers
hen = burnman.minerals.HHPH_2013.hen()
hfs = burnman.minerals.HHPH_2013.hfs()

hfm = burnman.CombinedMineral([hen, hfs], [0.5, 0.5], [-6.6e3, 0., 0.], name='hfm')

# Garnet endmembers
py = burnman.minerals.HHPH_2013.py()
alm = burnman.minerals.HHPH_2013.alm()
gr = burnman.minerals.HHPH_2013.gr()
dmaj = burnman.minerals.HHPH_2013.maj()
andr = burnman.minerals.HP_2011_ds62.andr()
nagt = burnman.minerals.HHPH_2013.nagt()

# xpy, VA^3, err, cubic py-maj from Heinemann et al., 1997
Heinemann_data = np.array([[0.2592, 0.4985, 0.7510, 0.9998],
                           [1511.167, 1509.386, 1506.802, 1503.654],
                           [0.024, 0.044, 0.047, 0.052]])
quad = lambda x, a, b, c: a*x + b*(1. - x) + c*x*(1. - x)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(quad,
                       xdata=Heinemann_data[0],
                       ydata=Heinemann_data[1]*1.e-30/8.*burnman.constants.Avogadro,
                       sigma=Heinemann_data[2])

py.params['V_0'] = popt[0]
dmaj.params['V_0'] = popt[1]
py_dmaj_Vex = popt[2]

"""
xs = np.linspace(0., 1., 101)
plt.errorbar(Heinemann_data[0],
             Heinemann_data[1]*1.e-30/8.*burnman.constants.Avogadro,
             yerr=Heinemann_data[2]*1.e-30/8.*burnman.constants.Avogadro,
             fmt='none', color='r')
plt.scatter(Heinemann_data[0],
         Heinemann_data[1]*1.e-30/8.*burnman.constants.Avogadro, color='r')
plt.plot(xs, quad(xs, *popt), color='r')
plt.errorbar([0., 0., 0.0659, 0.1285, 0.1906],
             np.array([1515.258, 1515.183, 1515.669, 1514.330, 1512.063])*1.e-30/8.*burnman.constants.Avogadro,
             yerr=np.array([0.051, 0.088, 0.419, 0.309, 0.426])*1.e-30/8.*burnman.constants.Avogadro, fmt='none', color='b')

plt.scatter([0., 0., 0.0659, 0.1285, 0.1906],
             np.array([1515.258, 1515.183, 1515.669, 1514.330, 1512.063])*1.e-30/8.*burnman.constants.Avogadro, color='b')
plt.ylim(11.3e-5, 11.42e-5)
plt.show()
"""



# Bridgmanite
mbdg = burnman.minerals.HHPH_2013.mpv()
fbdg = burnman.minerals.HHPH_2013.fpv()
abdg = burnman.minerals.HHPH_2013.apv()
fefbdg = burnman.minerals.HP_2011_ds62.hem()

fabdg = burnman.CombinedMineral([fefbdg, abdg], [0.5, 0.5], [-6.6e3, 0., 0.], name='fabdg')

# Akimotoite
mak = burnman.minerals.HHPH_2013.mak()
fak = burnman.minerals.HHPH_2013.fak()

# High pressure corundum phase
mcor = burnman.minerals.HHPH_2013.mcor()

facor = burnman.CombinedMineral([cor, hem], [0.5, 0.5], [-6.6e3, 0., 0.], name='facor')
fcor = burnman.CombinedMineral([mcor, mak, fak], [1., -1., 1.], [-15e3, 0., 0.], name='fcor')


# SiO2 polymorphs
qtz = burnman.minerals.HP_2011_ds62.q()
coe = burnman.minerals.HP_2011_ds62.coe()
stv = burnman.minerals.HP_2011_ds62.stv()




###############################
# MODIFY ENDMEMBER PROPERTIES #
###############################

gr.params['S_0_orig'] = [gr.params['S_0'], 1.] # from HP
dmaj.params['S_0_orig'] = [dmaj.params['S_0'], 1.] # from HP
andr.params['S_0_orig'] = [andr.params['S_0'], 1.] # from HP


wus.params['H_0'] = -2.65453e+05
wus.params['S_0'] = 59.82
wus.params['V_0'] = 12.239e-06
wus.params['a_0'] = 3.22e-05
wus.params['K_0'] = 162.e9
wus.params['Kprime_0'] = 4.9
wus.params['Cp'] = np.array([42.638, 0.00897102, -260780.8, 196.6])

# Metastable ferrowadsleyite
fwd.params['V_0'] = 4.31e-5 # original for HP is 4.321e-5
fwd.params['K_0'] = 160.e9 # 169.e9 is SLB
fwd.params['a_0'] = 2.48e-5 # 2.31e-5 is SLB

# Katsura et al., 2004
mrw.params['V_0'] = 3.95e-5
mrw.params['a_0'] = 2.13e-5
mrw.params['K_0'] = 182.e9
mrw.params['Kprime_0'] = 4.59

# Armentrout and Kavner, 2011
frw.params['V_0'] = 42.03e-6
frw.params['K_0'] = 202.e9
frw.params['Kprime_0'] = 4.
frw.params['a_0'] = 1.95e-5


# PRIORS FOR PARAMETERS
per.params['S_0_orig'] = [26.9, 0.1] # exp paper reported in Jacobs et al., 2017
wus.params['S_0_orig'] = [60.45, 1.] # Stolen et al., 1996 (in Jacobs et al., 2019)

fo.params['S_0_orig'] = [94.0, 0.1] # Dachs et al., 2007
fa.params['S_0_orig'] = [151.4, 0.1] # Dachs et al., 2007

mwd.params['S_0_orig'] = [86.4, 0.4] # exp paper reported in Jacobs et al., 2017
fwd.params['S_0_orig'] = [144.2, 3.] # similar relationship to fa and frw as the Mg-endmembers

mrw.params['S_0_orig'] = [82.7, 0.5] # exp paper reported in Jacobs et al., 2017
frw.params['S_0_orig'] = [140.2, 1.] # Yong et al., 2007; formal error is 0.4

oen.params['S_0_orig'] = [66.27*2., 0.1*2.] # reported in Jacobs et al., 2017 (Krupka et al., 1985)
ofs.params['S_0_orig'] = [186.5, 0.5] # Cemic and Dachs, 2006

py.params['S_0_orig'] = [265.94, 1.] # Dachs and Geiger, 2006; nominally 0.23, but sample dependent. HP2011_ds62 has 269.5, SLB has 244.55 (yeah, this what happens when you use a Debye model)
alm.params['S_0_orig'] = [342.6, 2.] # Anovitz et al., 1993

mbdg.params['S_0_orig'] = [57.9, 0.3] # Akaogi et al. (2008), formal error is 0.3
fbdg.params['S_0_orig'] = [fbdg.params['S_0'], 15.] # v. large uncertainties

sp.params['S_0_orig'] = [80.9, 0.6] # Klemme and Ahrens, 2007; 10.1007/s00269-006-0128-4

di.params['S_0_orig'] = [142.7, 0.2] # Krupke et al., 1985
hed.params['S_0_orig'] = [174.3, 0.3] # Haselton et al., 1987

# woll.params['S_0_orig'] = [81.69, 0.12] # Krupke et al., 1985

mins = [per, wus, fo, fa, mwd, fwd, mrw, frw, py, alm, gr, andr, dmaj, mbdg, fbdg, sp, di, hed]

print()
for m in mins:
    # Set entropies
    m.params['S_0'] = m.params['S_0_orig'][0]

    # Get thermal expansivities
    m.params['a_0_orig'] = m.params['a_0']



    
"""
# Entropies for fo-fa solid solutions from Dachs (2007). Almost a perfectly straight line.
plt.plot([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0],
         [151.4, 144.9, 138.9, 134.2,
         128.9,
         121.7,
         116.1,
         110.8,
         99.5,
          94.0])
plt.show()
exit()
"""

from scipy.optimize import curve_fit
func_Cp = lambda T, *c: c[0] + c[1]*T + c[2]/T/T + c[3]/np.sqrt(T)

"""
# mwd from Jahn et al., 2013
T, Cp, sigma = np.loadtxt('data/mwd_Cp_Jahn_2013.dat', unpack=True)
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = Cp, p0=mwd.params['Cp'])
mwd.params['Cp'] = popt
"""

"""
Cp -= 8./(2500.*2500.)*T*T
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = Cp, p0=mrw.params['Cp'])
mrw.params['Cp'] = popt
"""

# mrw from Kojitani et al., 2012
func_Cp_mrw_Kojitani = lambda T: 164.30 + 1.0216e-2*T + 7.6665e3/T - 1.1595e7/T/T + 1.3807e9/T/T/T 
tweak = lambda T: 5./(2500.*2500.)*T*T
T = np.linspace(306., 2500., 101)
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = func_Cp_mrw_Kojitani(T) + tweak(T), p0=mrw.params['Cp'])
mrw.params['Cp'] = popt


# WARNING, THIS IS NOT THE CP OF MWD, ITS JUST AN EASY WAY TO CREATE A SMOOTH MWD-MRW TRANSITION!!
tweak = lambda T: 2./(2500.*2500.)*T*T
T = np.linspace(306., 2500., 101)
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = func_Cp_mrw_Kojitani(T) + tweak(T), p0=mrw.params['Cp'])
mwd.params['Cp'] = popt


# fa from Benisek et al., 2012 (also used for fwd, frw)
T, Cp, sigma = np.loadtxt('data/fa_Cp_Benisek_2012.dat', unpack=True)
T = list(T)
T.extend([900., 1000., 1100., 1200., 1300., 1500., 1700., 2000., 2200.])
T = np.array(T)
P = 1.e5 * 0.*T
# This doesn't use the Benisek data...
Cp = fa.evaluate(['C_p'], P, T)[0]

"""
# Use Benisek data
fa.set_state(1.e5, 1673.15)
fa_gibbs = fa.gibbs
Cp = list(Cp)
Cp.extend([187.3, 190.9, 189, 180., 200., 205., 211.7, 219.6, 230.5])
sigma = list(sigma)
sigma.extend([1., 1., 1., 1., 1., 1., 1, 1])
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = Cp, p0=fa.params['Cp'])
fa.params['Cp'] = popt

fa.set_state(1.e5, 1673.15)
fa.params['H_0'] += fa_gibbs - fa.gibbs
"""

# NOTE: making the heat capacity of frw lower than fa results in a convex-down-pressure boundary in poor agreement with the experimental data.
# NOTE: making the heat capacity of frw the same as fa results in a nearly-linear curve
# NOTE: making the heat capacity of frw the same as fa+10./(np.sqrt(2500.))*np.sqrt(T) results in a curve that is too convex-up-pressure

# NOTE: fwd should have a slightly lower heat capacity than frw
tweak = lambda T: -2./(np.sqrt(2500.))*np.sqrt(T)
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = np.array(Cp) + tweak(np.array(T)), p0=fa.params['Cp'])
fwd.params['Cp'] = popt

tweak = lambda T: 2./(np.sqrt(2500.))*np.sqrt(T)
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = np.array(Cp) + tweak(np.array(T)), p0=fa.params['Cp'])
frw.params['Cp'] = popt

"""
# Plot heat capacities of mwd and mrw
temperatures = np.linspace(300., 2400., 101)
pressures = 1.e5 + temperatures*0.

fig = plt.figure(figsize=(20., 15.))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
ax[0].imshow(mwd_Cp_diagram, extent=[0., 800., 0., 180.], aspect='auto')
ax[0].plot(temperatures, mwd.evaluate(['C_p'], pressures, temperatures)[0])

ax[1].imshow(mrw_Cp_diagram, extent=[0., 1000., 0., 180.], aspect='auto')
ax[1].plot(temperatures, mrw.evaluate(['C_p'], pressures, temperatures)[0])

ax[2].plot(temperatures, fo.evaluate(['C_p'], pressures, temperatures)[0], label='fo', linewidth=3.)
ax[2].plot(temperatures, mwd.evaluate(['C_p'], pressures, temperatures)[0], label='mwd')
ax[2].plot(temperatures, mrw.evaluate(['C_p'], pressures, temperatures)[0], label='mrw')
ax[2].legend()

#ax[3].imshow(fa_Cp_diagram, extent=[250., 2200., 100., 240.], aspect='auto')
ax[3].imshow(frw_Cp_diagram, extent=[0., 700., 0., 200.], aspect='auto')

ax[3].plot(temperatures, fa.evaluate(['C_p'], pressures, temperatures)[0], label='fa', linewidth=3.)
ax[3].plot(temperatures, fwd.evaluate(['C_p'], pressures, temperatures)[0], label='fwd')
ax[3].plot(temperatures, frw.evaluate(['C_p'], pressures, temperatures)[0], label='frw')
ax[3].legend()

for i in range(4):
    ax[i].set_xlabel('T (K)')
    ax[i].set_ylabel('C_p (J/K/mol)')

plt.show()
"""

for (P, m1, m2) in [[14.25e9, fo, mwd],
                    [19.7e9, mwd, mrw],
                    [6.2e9, fa, frw],
                    [6.23e9, fa, fwd]]:
    m1.set_state(P, 1673.15)
    m2.set_state(P, 1673.15)
    m2.params['H_0'] += m1.gibbs - m2.gibbs
    


###################
# SOLID SOLUTIONS #
###################

bdg = Solution(name = 'bridgmanite',
               solution_type ='symmetric',
               endmembers=[[mbdg, '[Mg][Si]O3'],
                           [fbdg, '[Fe][Si]O3'],
                           [abdg, '[Al][Al]O3'],
                           [fefbdg, '[Fef][Fef]O3'],
                           [fabdg, '[Fef][Al]O3']],
               energy_interaction=[[6.e3, 0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0.],
                                   [0.]],
               volume_interaction=[[0., 0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0.],
                                   [0.]])

cor = Solution(name = 'corundum',
               solution_type ='symmetric',
               endmembers=[[cor, '[Al][Al]O3'],
                           [hem, '[Fef][Fef]O3'],
                           [facor, '[Fef][Al]O3'],
                           [mcor, '[Mg][Si]O3'],
                           [fcor, '[Fe][Si]O3']],
               energy_interaction=[[0., 0., 6.e3, 0.],
                                   [0., 0., 0.],
                                   [0., 0.],
                                   [0.]],
               volume_interaction=[[0., 0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0.],
                                   [0.]])
               
fper = Solution(name = 'ferropericlase',
                solution_type ='symmetric',
                endmembers=[[per, '[Mg]O'], [wus, '[Fe]O']],
                energy_interaction=[[11.1e3]],
                volume_interaction=[[1.1e-7]]) 
ol = Solution(name = 'olivine',
              solution_type ='symmetric',
              endmembers=[[fo, '[Mg]2SiO4'], [fa, '[Fe]2SiO4']],
              energy_interaction=[[6.37e3]],
              volume_interaction=[[0.e-7]]) # O'Neill et al., 2003
wad = Solution(name = 'wadsleyite',
               solution_type ='symmetric',
               endmembers=[[mwd, '[Mg]2SiO4'], [fwd, '[Fe]2SiO4']],
               energy_interaction=[[16.7e3]],
               volume_interaction=[[0.e-7]])

spinel = Solution(name = 'spinel',
                  solution_type ='symmetric',
                  endmembers=[[sp, '[Al7/8Mg1/8]2[Mg3/4Al1/4]O4'],
                              [herc, '[Al7/8Fe1/8]2[Fe3/4Al1/4]O4'],
                              [mt, '[Fef7/8Fe1/8]2[Fe3/4Fef1/4]O4'],
                              [mrw, '[Mg]2[Si]O4'],
                              [frw, '[Fe]2[Si]O4']],
                  energy_interaction=[[0.e3, 0.e3, 0.e3, 0.e3],
                                      [0.e3, 0.e3, 0.e3],
                                      [0.e3, 0.e3],
                                      [7.6e3]],
                  volume_interaction=[[0.e-7, 0.e-7, 0.e-7, 0.e-7],
                                      [0.e-7, 0.e-7, 0.e-7],
                                      [0.e-7, 0.e-7],
                                      [0.e-7]])

"""
spinel_od = Solution(name = 'spinel with order-disorder',
                     solution_type ='symmetric', # fake multiplicity of 1 (should be 2)
                     endmembers=[[nsp,   '[Mg][Al]AlO4'],
                                 [nherc, '[Fe][Al]AlO4'],
                                 [nmt,   '[Fe][Fef]FefO4'],
                                 [mrw,   '[Si][Mg]MgO4'],
                                 [frw,   '[Si][Fe]FeO4'],
                                 [isp,   '[Al][Mg1/2Al1/2]Mg1/2Al1/2O4'],
                                 [iherc, '[Al][Fe1/2Al1/2]Fe1/2Al1/2O4'],
                                 [imt,   '[Fef][Fe1/2Fef1/2]Fe1/2Fef1/2O4']], # 3 ordered endmembers
                     energy_interaction=[[0.e3, 0.e3, 0.e3, 0.e3, 0.e3, 0.e3, 0.e3],
                                         [0.e3, 0.e3, 0.e3, 0.e3, 0.e3, 0.e3,],
                                         [0.e3, 0.e3, 0.e3, 0.e3, 0.e3,],
                                         [0.e3, 0.e3, 0.e3, 0.e3],
                                         [0.e3, 0.e3, 0.e3],
                                         [0.e3, 0.e3],
                                         [7.6e3]],
                     volume_interaction=[[0.e-7, 0.e-7, 0.e-7, 0.e-7, 0.e-7, 0.e-7, 0.e-7],
                                         [0.e-7, 0.e-7, 0.e-7, 0.e-7, 0.e-7, 0.e-7],
                                         [0.e-7, 0.e-7, 0.e-7, 0.e-7, 0.e-7],
                                         [0.e-7, 0.e-7, 0.e-7, 0.e-7],
                                         [0.e-7, 0.e-7, 0.e-7],
                                         [0.e-7, 0.e-7],
                                         [0.e-7]])
"""

cpx_od = Solution(name = 'clinopyroxene with order-disorder',
                  solution_type = 'asymmetric', # fake multiplicity of 1/2 (should be 2)
                  alphas = [1.2, 1.2, 1.0, 1.0, 1.9, 1.2, 1.2],
                  endmembers = [[di,   '[Ca][Mg][Si]1/2O6'],
                                [hed,  '[Ca][Fe][Si]1/2O6'],
                                [cen,  '[Mg][Mg][Si]1/2O6'],
                                [cfs,  '[Fe][Fe][Si]1/2O6'], # order-disorder
                                [cats, '[Ca][Al][Si1/2Al1/2]1/2O6'],
                                [jd,   '[Na][Al][Si]1/2O6'],
                                [aeg,  '[Na][Fef][Si]1/2O6']],
                  energy_interaction=[[ 2.9e3, 29.8e3, 25.8e3, 13.0e3, 26.0e3, 26.7e3],
                                      [26.6e3, 20.9e3,  8.9e3,  9.6e3, 10.4e3],
                                      [ 2.3e3, 45.2e3, 40.0e3, 60.8e3],
                                      [25.0e3, 24.0e3, 52.3e3],
                                      [ 6.0e3, 17.4e3],
                                      [ 3.2e3]],
                  volume_interaction=[[ 0.0,  -3e-7,   -3e-7,  -6e-7,    0.0,    7.35e-7],
                                      [-3e-7, -3e-7,   -6e-7,    0.0,    7.35e-7],
                                      [ 0.0, -3.5e-06, 0.0,    4.2e-06],
                                      [-1e-6,  0.0,    1.2e-6],
                                      [ 0.0,   0.0],
                                      [ 0.0]])


opx_od = Solution(name = 'orthopyroxene with order-disorder',
                  solution_type ='asymmetric', # fake multiplicity of 1/2 (should be 2)
                  alphas = [1., 1., 1., 1.2, 1.],
                  endmembers=[[oen,  '[Mg][Mg][Si]1/2Si3/2O6'],
                              [ofs,  '[Fe][Fe][Si]1/2Si3/2O6'],
                              [mgts, '[Mg][Al][Al1/2Si1/2]1/2Al3/4Si3/4O6'], 
                              [odi,  '[Ca][Mg][Si]1/2Si3/2O6'], 
                              [ofm,  '[Fe][Mg][Si]1/2Si3/2O6']], # Fe-Mg o-d with Mg on the Al site
                  energy_interaction=[[7.e3, 12.5e3, 32.2e3, 4.e3],
                                      [11.e3, 25.54e3, 4.e3],
                                      [75.5e3, 15.e3],
                                      [22.54e3]],
                  volume_interaction=[[0., -0.04e-5, 0.12e-5, 0.],
                                      [-0.15e-5, 0.084e-5, 0.],
                                      [-0.84e-5, -0.15e-5],
                                      [0.084e-5]])

hpx_od = Solution(name = 'high pressure clinopyroxene with order-disorder',
                  solution_type ='asymmetric', # fake multiplicity of 1/2 (should be 2)
                  alphas = [1., 1., 1., 1.2, 1.],
                  endmembers=[[hen,  '[Mg][Mg][Si]1/2Si3/2O6'],
                              [hfs,  '[Fe][Fe][Si]1/2Si3/2O6'],
                              [mgts, '[Mg][Al][Al1/2Si1/2]1/2Al3/4Si3/4O6'], 
                              [odi,  '[Ca][Mg][Si]1/2Si3/2O6'], 
                              [hfm,  '[Fe][Mg][Si]1/2Si3/2O6']], # Fe-Mg o-d with Mg on the Al site
                  energy_interaction=[[7.e3, 12.5e3, 32.2e3, 4.e3],
                                      [11.e3, 25.54e3, 4.e3],
                                      [75.5e3, 15.e3],
                                      [22.54e3]],
                  volume_interaction=[[0., -0.04e-5, 0.12e-5, 0.],
                                      [-0.15e-5, 0.084e-5, 0.],
                                      [-0.84e-5, -0.15e-5],
                                      [0.084e-5]])

gt = Solution(name = 'garnet',
              solution_type = 'symmetric',
              endmembers = [[py, '[Mg]3[Al]2Si3O12'],
                            [alm, '[Fe]3[Al]2Si3O12'],
                            [gr, '[Ca]3[Al]2Si3O12'],
                            [andr, '[Ca]3[Fe]2Si3O12'],
                            [dmaj, '[Mg]3[Mg1/2Si1/2]2Si3O12'],
                            [nagt, '[Na1/3Mg2/3]3[Al1/2Si1/2]2Si3O12']],
              energy_interaction=[[0.e3, 30.e3, 90.e3, 0., 0.], # py-.....
                                  [0.e3, 60.e3, 0., 0.], # alm-....
                                  [5.e3, 0., 0.], # gr-...
                                  [0., 0.], # andr-..
                                  [0.]], # dmaj-namaj
              volume_interaction=[[0., 0., 0., py_dmaj_Vex, 0.], # py_dmaj from Heinemann
                                  [0., 0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0.],
                                  [0.]])

print('Warning! oen_ofs still doesn\'t have o-d')

# Child solutions *must* be in dictionary to be reset properly
child_solutions = {'mg_fe_bdg': transform_solution_to_new_basis(bdg,
                                                                np.array([[1., 0., 0., 0., 0.],
                                                                          [0., 1., 0., 0., 0.]]),
                                                                solution_name='mg-fe bridgmanite'),
                   'mg_al_bdg': transform_solution_to_new_basis(bdg,
                                                                np.array([[1., 0., 0., 0., 0.],
                                                                          [0., 0., 1., 0., 0.]]),
                                                                solution_name='mg-al bridgmanite'),
                   'al_mg_cor': transform_solution_to_new_basis(cor,
                                                                np.array([[1., 0., 0., 0., 0.],
                                                                          [0., 0., 0., 1., 0.]]),
                                                                solution_name='al-mg corundum'),
                   'py_alm_gt': transform_solution_to_new_basis(gt,
                                                                np.array([[1., 0., 0., 0., 0., 0.],
                                                                          [0., 1., 0., 0., 0., 0.]]),
                                                                solution_name='py-alm garnet'),
                   
                   'py_gr_gt': transform_solution_to_new_basis(gt,
                                                               np.array([[1., 0., 0., 0., 0., 0.],
                                                                         [0., 0., 1., 0., 0., 0.]]),
                                                               solution_name='py-gr garnet'),
                   
                   'py_alm_gr_gt': transform_solution_to_new_basis(gt,
                                                                   np.array([[1., 0., 0., 0., 0., 0.],
                                                                             [0., 1., 0., 0., 0., 0.],
                                                                             [0., 0., 1., 0., 0., 0.]]),
                                                                   solution_name='py-alm-gr garnet'),
                   
                   'alm_sk_gt': transform_solution_to_new_basis(gt,
                                                                np.array([[0., 1.,  0., 0., 0., 0.],
                                                                          [0., 1., -1., 1., 0., 0.]]),
                                                                solution_name='alm-sk garnet'),
                   'lp_gt': transform_solution_to_new_basis(gt,
                                                            np.array([[1., 0., 0., 0., 0., 0.],
                                                                      [0., 1., 0., 0., 0., 0.],
                                                                      [0., 0., 1., 0., 0., 0.],
                                                                      [0., 0., 0., 1., 0., 0.]]),
                                                            solution_name='py-alm-gr-andr garnet'),
                   'lp_FMASO_gt': transform_solution_to_new_basis(gt,
                                                                  np.array([[1., 0., 0., 0., 0., 0.],
                                                                            [0., 1., 0., 0., 0., 0.],
                                                                            [1., 0., -1., 1., 0., 0.]]),
                                                                  solution_name='py-alm-kho garnet'),
                   'FMASO_gt': transform_solution_to_new_basis(gt,
                                                               np.array([[1., 0., 0., 0., 0., 0.],
                                                                         [0., 1., 0., 0., 0., 0.],
                                                                         [1., 0., -1., 1., 0., 0.],
                                                                         [0., 0., 0., 0., 1., 0.]]),
                                                               solution_name='py-alm-kho-dmaj garnet'),
                   'FMAS_gt': transform_solution_to_new_basis(gt,
                                                              np.array([[1., 0., 0., 0., 0., 0.],
                                                                        [0., 1., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 1., 0.]]),
                                                              solution_name='py-alm-dmaj garnet'),
                   'xna_gt': transform_solution_to_new_basis(gt,
                                                             np.array([[1., 0., 0., 0., 0., 0.],
                                                                       [0., 1., 0., 0., 0., 0.],
                                                                       [0., 0., 1., 0., 0., 0.],
                                                                       [0., 0., 0., 1., 0., 0.],
                                                                       [0., 0., 0., 0., 1., 0.]]),
                                                             solution_name='py-alm-gr-andr-dmaj garnet'),
                   'xmj_gt': transform_solution_to_new_basis(gt,
                                                             np.array([[1., 0., 0., 0., 0., 0.],
                                                                       [0., 1., 0., 0., 0., 0.],
                                                                       [0., 0., 1., 0., 0., 0.],
                                                                       [0., 0., 0., 1., 0., 0.],
                                                                       [0., 0., 0., 0., 0., 1.]]),
                                                             solution_name='py-alm-gr-andr-nagt garnet'),
                   
                   'sk_gt': transform_solution_to_new_basis(gt, np.array([[0., 1., -1., 1., 0., 0.]]),
                                                                solution_name='skiagite'),
                   
                   'py_dmaj_gt': transform_solution_to_new_basis(gt,
                                                                 np.array([[1., 0., 0., 0., 0., 0.],
                                                                           [0., 0., 0., 0., 1., 0.]]),
                                                                 solution_name='py-dmaj garnet'),
                   
                   'ring': transform_solution_to_new_basis(spinel,
                                                           np.array([[0., 0., 0., 1., 0.],
                                                                     [0., 0., 0., 0., 1.]]),
                                                           solution_name='ringwoodite'),
                   
                   'herc_mt_frw': transform_solution_to_new_basis(spinel,
                                                                  np.array([[0., 1., 0., 0., 0.],
                                                                            [0., 0., 1., 0., 0.],
                                                                            [0., 0., 0., 0., 1.]]),
                                                                  solution_name='herc-mt-frw spinel'),
                   'mt_frw': transform_solution_to_new_basis(spinel,
                                                             np.array([[0., 0., 1., 0., 0.],
                                                                       [0., 0., 0., 0., 1.]]),
                                                             solution_name='mt-frw spinel'),
                   
                   'oen_ofs': transform_solution_to_new_basis(opx_od,
                                                              np.array([[1., 0., 0., 0., 0.],
                                                                        [0., 1., 0., 0., 0.]]),
                                                              solution_name='Mg-Fe orthopyroxene'),
                   
                   'oen_mgts': transform_solution_to_new_basis(opx_od,
                                                               np.array([[1., 0., 0., 0., 0.],
                                                                         [0., 0., 1., 0., 0.]]),
                                                               solution_name='MAS orthopyroxene'),
                   
                   'oen_mgts_odi': transform_solution_to_new_basis(opx_od,
                                                                   np.array([[1., 0., 0., 0., 0.],
                                                                             [0., 0., 1., 0., 0.],
                                                                             [0., 0., 0., 1., 0.]]),
                                                                   solution_name='CMAS orthopyroxene'),
                   
                   'oen_odi': transform_solution_to_new_basis(opx_od,
                                                              np.array([[1., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 1., 0.]]),
                                                              solution_name='CMS orthopyroxene'),
                   
                   'ofs_fets': transform_solution_to_new_basis(opx_od,
                                                               np.array([[0., 1., 0., 0., 0.],    # ofs
                                                                         [-1., 0., 1., 0., 1.]]), # fets = - oen + mgts + ofm
                                                               solution_name='FAS orthopyroxene'), 
                   
                   'di_cen': transform_solution_to_new_basis(cpx_od,
                                                             np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                                       [0., 0., 1., 0., 0., 0., 0.]]),
                                                             solution_name='CMS clinopyroxene'),
                   
                   'di_hed': transform_solution_to_new_basis(cpx_od,
                                                             np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                                       [0., 1., 0., 0., 0., 0., 0.]]),
                                                             solution_name='di-hed clinopyroxene'),
                   
                   'di_cen_cats': transform_solution_to_new_basis(cpx_od,
                                                                  np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                                            [0., 0., 1., 0., 0., 0., 0.],
                                                                            [0., 0., 0., 0., 1., 0., 0.]]),
                                                                  solution_name='CMAS orthopyroxene')}


solutions = {'mw': fper,
             'ol': ol,
             'wad': wad,
             'sp': spinel,
             'gt': gt,
             'cpx': cpx_od,
             'opx': opx_od,
             'hpx': hpx_od,
             'bdg': bdg}

endmembers = {'Re':       Re,       # buffer materials
              'ReO2':     ReO2,
              'Mo':       Mo,
              'MoO2':     MoO2,
              'bcc_iron': bcc_iron, # iron polymorphs
              'fcc_iron': fcc_iron,
              'hcp_iron': hcp_iron,
              'per':      per,      # MgO / FeO
              'wus':      wus,
              'fo':       fo,       # olivine
              'fa':       fa,
              'mwd':      mwd,      # wadsleyite
              'fwd':      fwd,
              'mrw':      mrw,      # spinel / ringwoodite
              'frw':      frw,
              'sp':       sp,
              'herc':     herc,
              'mt':       mt,
              'py':       py,       # garnet
              'alm':      alm,
              'gr':       gr,
              'andr':     andr,
              'dmaj':     dmaj,
              'nagt':     nagt,
              'di':       di,       # clinopyroxene
              'hed':      hed,
              'cen':      cen,
              'cfs':      cfs,
              'cats':     cats,
              'jd':       jd,
              'aeg':      aeg,
              'oen':      oen,      # orthopyroxene
              'ofs':      ofs,
              'mgts':     mgts,
              'odi':      odi,
              'hen':      hen,      # high pressure (C2/c) clinopyroxene
              'hfs':      hfs,
              'mbdg':     mbdg,
              'fbdg':     fbdg,
              'qtz':      qtz,      # SiO2 polymorphs
              'coe':      coe,
              'stv':      stv}

