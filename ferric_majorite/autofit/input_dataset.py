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

# Some figures
mrw_volume_diagram = mpimg.imread('figures/Katsura_2004_rw_volumes.png')
mwd_Cp_diagram = mpimg.imread('frost_2003_figures/Cp_wadsleyite_Jacobs_2007.png')
mrw_Cp_diagram = mpimg.imread('frost_2003_figures/Cp_ringwoodite_Jacobs_2007.png')

fa_Cp_diagram = mpimg.imread('frost_2003_figures/fa_Cp_Benisek_2012.png')
frw_Cp_diagram = mpimg.imread('frost_2003_figures/frw_Cp_Yong_2007.png')


# Load the minerals and solid solutions we wish to use
per = burnman.minerals.HHPH_2013.per()
wus = burnman.minerals.HHPH_2013.fper()
fo = burnman.minerals.HHPH_2013.fo()
fa = burnman.minerals.HHPH_2013.fa()
mwd = burnman.minerals.HHPH_2013.mwd()
fwd = burnman.minerals.HHPH_2013.fwd()
mrw = burnman.minerals.HHPH_2013.mrw()
frw = burnman.minerals.HHPH_2013.frw()
py = burnman.minerals.HHPH_2013.py()
alm = burnman.minerals.HHPH_2013.alm()

mins = [per, wus, fo, fa, mwd, fwd, mrw, frw, py, alm]

wus.params['H_0'] = -2.65453e+05
wus.params['S_0'] = 59.82
wus.params['V_0'] = 12.239e-06
wus.params['a_0'] = 3.22e-05
wus.params['K_0'] = 162.e9
wus.params['Kprime_0'] = 4.9
wus.params['Cp'] = np.array([42.638, 0.00897102, -260780.8, 196.6])

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
wus.params['S_0_orig'] = [58., 6.] # exp paper reported in Jacobs et al., 2017

fo.params['S_0_orig'] = [94.0, 0.1] # Dachs et al., 2007
fa.params['S_0_orig'] = [151.4, 0.1] # Dachs et al., 2007

mwd.params['S_0_orig'] = [86.4, 0.4] # exp paper reported in Jacobs et al., 2017
fwd.params['S_0_orig'] = [140.2, 10.] # same as Yong et al., 2007 for rw, but with bigger error

mrw.params['S_0_orig'] = [82.7, 0.5] # exp paper reported in Jacobs et al., 2017
frw.params['S_0_orig'] = [140.2, 1.] # Yong et al., 2007; formal error is 0.4


py.params['S_0_orig'] = [265.94, 1.] # Dachs and Geiger, 2006; nominally 0.23, but sample dependent. HP2011_ds62 has 269.5, SLB has 244.55 (yeah, this what happens when you use a Debye model)
alm.params['S_0_orig'] = [342.6, 2.] # Anovitz et al., 1993

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

fa.set_state(1.e5, 1673.15)
fa_gibbs = fa.gibbs

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
Cp = list(Cp)
Cp.extend([187.3, 190.9, 189, 180., 200., 205., 211.7, 219.6, 230.5])
sigma = list(sigma)
sigma.extend([1., 1., 1., 1., 1., 1., 1, 1])
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = Cp, p0=fa.params['Cp'])
fa.params['Cp'] = popt

# NOTE: making the heat capacity of frw lower than fa results in a convex-down-pressure boundary in poor agreement with the experimental data.
# NOTE: making the heat capacity of frw the same as fa results in a nearly-linear curve
# NOTE: making the heat capacity of frw the same as fa+10./(np.sqrt(2500.))*np.sqrt(T) results in a curve that is too convex-up-pressure

# NOTE: fwd should have a slightly lower heat capacity than frw
tweak = lambda T: -6./(np.sqrt(2500.))*np.sqrt(T)
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = np.array(Cp) + tweak(np.array(T)), p0=fa.params['Cp'])
fwd.params['Cp'] = popt

tweak = lambda T: -2./(np.sqrt(2500.))*np.sqrt(T)
popt, pcov = curve_fit(func_Cp, xdata = T, ydata = np.array(Cp) + tweak(np.array(T)), p0=fa.params['Cp'])
frw.params['Cp'] = popt

fa.set_state(1.e5, 1673.15)
fa.params['H_0'] += fa_gibbs - fa.gibbs


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

plt.show()


for (P, m1, m2) in [[14.25e9, fo, mwd],
                    [19.7e9, mwd, mrw],
                    [6.2e9, fa, frw],
                    [6.23e9, fa, fwd]]:
    m1.set_state(P, 1673.15)
    m2.set_state(P, 1673.15)
    m2.params['H_0'] += m1.gibbs - m2.gibbs
    

fper = Solution(name = 'ferropericlase',
                solution_type ='symmetric',
                endmembers=[[per, '[Mg]O'], [wus, '[Fe]O']],
                energy_interaction=[[11.1e3]],
                volume_interaction=[[1.1e-7]]) 
ol = Solution(name = 'olivine',
              solution_type ='symmetric',
              endmembers=[[fo, '[Mg]2SiO4'], [fa, '[Fe]2SiO4']],
              energy_interaction=[[5.2e3]],
              volume_interaction=[[0.e-7]]) # O'Neill et al., 2003
wad = Solution(name = 'wadsleyite',
               solution_type ='symmetric',
               endmembers=[[mwd, '[Mg]2SiO4'], [fwd, '[Fe]2SiO4']],
               energy_interaction=[[16.7e3]],
               volume_interaction=[[0.e-7]])
rw = Solution(name = 'ringwoodite',
              solution_type ='symmetric',
              endmembers=[[mrw, '[Mg]2SiO4'], [frw, '[Fe]2SiO4']],
              energy_interaction=[[7.6e3]],
              volume_interaction=[[0.e-7]]) 
gt = Solution(name = 'garnet',
              solution_type ='symmetric',
              endmembers=[[py, '[Mg]3Al2Si3O12'], [alm, '[Fe]3Al2Si3O12']],
              energy_interaction=[[-2.e3]],
              volume_interaction=[[0.e-7]])


solutions = {'mw': fper,
             'ol': ol,
             'wad': wad,
             'ring': rw,
             'gt': gt}

endmembers = {'per': per,
              'wus': wus,
              'fo': fo,
              'fa': fa,
              'mwd': mwd,
              'fwd': fwd,
              'mrw': mrw,
              'frw': frw,
              'py': py,
              'alm': alm}
