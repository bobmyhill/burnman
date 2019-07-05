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
if not os.path.exists('burnman') and os.path.exists('../../burnman'):
    sys.path.insert(1, os.path.abspath('../..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.solutionbases import transform_solution_to_new_basis

#########################
# ENDMEMBER DEFINITIONS #
#########################

# Iron endmembers
bcc_iron = burnman.minerals.SE_2015.bcc_iron()
fcc_iron = burnman.minerals.SE_2015.fcc_iron()
hcp_iron = burnman.minerals.SE_2015.hcp_iron()

# O2
O2 = burnman.minerals.HP_2011_fluids.O2()

# Oxides
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

# Orthopyroxene endmembers
oen = burnman.minerals.HHPH_2013.en()
ofs = burnman.minerals.HHPH_2013.fs()
mgts = burnman.minerals.HP_2011_ds62.mgts()
odi = burnman.minerals.HP_2011_ds62.di()
odi.params['H_0'] += -0.1e3
odi.params['S_0'] += -0.211
odi.params['V_0'] += 0.005e-5

# High pressure clinopyroxene endmembers
hen = burnman.minerals.HHPH_2013.hen()
hfs = burnman.minerals.HHPH_2013.hfs()

# Garnet endmembers
py = burnman.minerals.HHPH_2013.py()
alm = burnman.minerals.HHPH_2013.alm()
gr = burnman.minerals.HHPH_2013.gr()
dmaj = burnman.minerals.HHPH_2013.maj()
andr = burnman.minerals.HP_2011_ds62.andr()
nagt = burnman.minerals.HHPH_2013.nagt()

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


mins = [per, wus, fo, fa, mwd, fwd, mrw, frw, py, alm, gr, andr, dmaj]

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

cpx = Solution(name = 'clinopyroxene',
               solution_type = 'symmetric',
               endmembers = [[di,   '[Ca][Mg][Si]2O6'],
                             [hed,  '[Ca][Fe][Si]2O6'],
                             [cen,  '[Mg][Mg][Si]2O6'],
                             [cats, '[Ca][Al][Si1/2Al1/2]2O6'],
                             [jd,   '[Na][Al][Si]2O6'],
                             [aeg,  '[Na][Fef][Si]2O6']],
              energy_interaction=[[0.e3, 25.e3, 26.e3, 24.e3, 24.e3],
                                  [25.e3, 0.e3, 0.e3, 0.],
                                  [61.e3, 0.e3, 0.],
                                  [10.e3, 10.e3],
                                  [0.e3]],
              volume_interaction=[[0.e-7, 0.e-7, 0.e-7, 0.e-7, 0.e-7],
                                  [0.e-7, 0.e-7, 0.e-7, 0.e-7],
                                  [0.e-7, 0.e-7, 0.e-7],
                                  [0.e-7, 0.e-7],
                                  [0.e-7]])
               
opx = Solution(name = 'orthopyroxene',
               solution_type ='symmetric',
               endmembers=[[oen,  '[Mg][Mg]SiSiO6'],
                           [ofs,  '[Fe][Fe]SiSiO6'],
                           [mgts, '[Mg][Al]AlSiO6'], # Al avoidance, see Figure 3
                           [odi,  '[Ca][Mg]SiSiO6']],
               energy_interaction=[[2.e3, 0.e3, 0.e3],
                                   [0.e3, 0.e3],
                                   [0.e3]],
               volume_interaction=[[0.e-7, 0.e-7, 0.e-7],
                                   [0.e-7, 0.e-7],
                                   [0.e-7]])

hpx = Solution(name = 'high pressure clinopyroxene',
               solution_type ='symmetric',
               endmembers=[[hen, '[Mg]2Si2O6'], [hfs, '[Fe]2Si2O6']],
               energy_interaction=[[2.e3]],
               volume_interaction=[[0.e-7]])

gt = Solution(name = 'disordered garnet',
              solution_type = 'symmetric',
              endmembers = [[py, '[Mg]3[Al]2Si3O12'],
                            [alm, '[Fe]3[Al]2Si3O12'],
                            [gr, '[Ca]3[Al]2Si3O12'],
                            [andr, '[Ca]3[Fe]2Si3O12'],
                            [dmaj, '[Mg]3[Mg1/2Si1/2]2Si3O12'],
                            [nagt, '[Na1/3Mg2/3]3[Al1/2Si1/2]2Si3O12']],
              energy_interaction=[[0.e3, 30.e3, 60.e3, 0., 0.], # py-.....
                                  [0.e3, 0.e3, 0., 0.], # alm-....
                                  [5.e3, 0., 0.], # gr-...
                                  [0., 0.], # andr-..
                                  [0.]], # dmaj-namaj
              volume_interaction=[[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0.],
                                  [0.]])

# Child solutions *must* be in dictionary to be reset properly
child_solutions = {'py_alm_gt': transform_solution_to_new_basis(gt,
                                                                np.array([[1., 0., 0., 0., 0., 0.],
                                                                          [0., 1., 0., 0., 0., 0.]])),
                   'py_gr_gt': transform_solution_to_new_basis(gt,
                                                               np.array([[1., 0., 0., 0., 0., 0.],
                                                                         [0., 0., 1., 0., 0., 0.]])),
                   'alm_sk_gt': transform_solution_to_new_basis(gt,
                                                                np.array([[0., 1.,  0., 0., 0., 0.],
                                                                          [0., 1., -1., 1., 0., 0.]])),
                   'sk_gt': transform_solution_to_new_basis(gt, np.array([[0., 1., -1., 1., 0., 0.]])),
                   'py_dmaj_gt': transform_solution_to_new_basis(gt,
                                                                np.array([[1., 0., 0., 0., 0., 0.],
                                                                          [0., 0., 0., 0., 1., 0.]])),
                   'ring': transform_solution_to_new_basis(spinel,
                                                           np.array([[0., 0., 0., 1., 0.],
                                                                     [0., 0., 0., 0., 1.]])),
                   'herc_mt_frw': transform_solution_to_new_basis(spinel,
                                                                  np.array([[0., 1., 0., 0., 0.],
                                                                            [0., 0., 1., 0., 0.],
                                                                            [0., 0., 0., 0., 1.]])),
                   'mt_frw': transform_solution_to_new_basis(spinel,
                                                             np.array([[0., 0., 1., 0., 0.],
                                                                       [0., 0., 0., 0., 1.]])),
                   'oen_ofs': transform_solution_to_new_basis(opx,
                                                              np.array([[1., 0., 0., 0.],
                                                                        [0., 1., 0., 0.]])),
                   'oen_mgts': transform_solution_to_new_basis(opx,
                                                               np.array([[1., 0., 0., 0.],
                                                                         [0., 0., 1., 0.]])),
                   'oen_mgts_odi': transform_solution_to_new_basis(opx,
                                                                   np.array([[1., 0., 0., 0.],
                                                                             [0., 0., 1., 0.],
                                                                             [0., 0., 0., 1.]])),
                   'oen_odi': transform_solution_to_new_basis(opx,
                                                              np.array([[1., 0., 0., 0.],
                                                                        [0., 0., 0., 1.]])),
                   'di_cen': transform_solution_to_new_basis(cpx,
                                                             np.array([[1., 0., 0., 0., 0., 0.],
                                                                       [0., 0., 1., 0., 0., 0.]])),
                   'di_hed': transform_solution_to_new_basis(cpx,
                                                             np.array([[1., 0., 0., 0., 0., 0.],
                                                                       [0., 1., 0., 0., 0., 0.]])),
                   'di_cen_cats': transform_solution_to_new_basis(cpx,
                                                                  np.array([[1., 0., 0., 0., 0., 0.],
                                                                            [0., 0., 1., 0., 0., 0.],
                                                                            [0., 0., 0., 1., 0., 0.]]))}


solutions = {'mw': fper,
             'ol': ol,
             'wad': wad,
             'sp': spinel,
             'gt': gt,
             'cpx': cpx,
             'opx': opx,
             'hpx': hpx}

endmembers = {'bcc_iron': bcc_iron, # iron polymorphs
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
              'cats':     cats,
              'jd':       jd,
              'oen':      oen,      # orthopyroxene
              'ofs':      ofs,
              'mgts':     mgts,
              'odi':      odi,
              'hen':      hen,      # high pressure (C2/c) clinopyroxene
              'hfs':      hfs,
              'qtz':      qtz,      # SiO2 polymorphs
              'coe':      coe,
              'stv':      stv}
