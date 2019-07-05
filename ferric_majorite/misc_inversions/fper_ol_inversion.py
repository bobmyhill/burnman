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

# Plot endmember phase diagrams
def eqm_pressures(m1, m2, temperatures):
    composition = m1.formula
    assemblage = burnman.Composite([m1, m2])
    assemblage.set_state(1.e5, temperatures[0])
    equality_constraints = [('T', temperatures), ('phase_proportion', (m1, 0.0))]
    sols, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                    initial_state_from_assemblage=True,
                                    store_iterates=False)
    if type(sols) is list:
        return np.array([sol.x[0] for sol in sols])
    else:
        return sols.x[0]

run_inversion = raw_input("Do you want to invert the data now [y/n]?:\n(n plots the results of a previous inversion) ")


# A few images:
ol_polymorph_img = mpimg.imread('frost_2003_figures/ol_polymorphs.png')
ol_polymorph_img_1200C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_1200C.png')
ol_polymorph_img_1000C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_1000C.png')
ol_polymorph_img_800C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_800C.png')



ol_fper_img = mpimg.imread('frost_2003_figures/ol_fper_RTlnKD.png')
wad_fper_img = mpimg.imread('frost_2003_figures/wad_fper_RTlnKD.png')
rw_fper_img = mpimg.imread('frost_2003_figures/ring_fper_gt_KD.png')
rw_fper_part_img = mpimg.imread('frost_2003_figures/ring_fper_partitioning.png')


# Some more
mrw_volume_diagram = mpimg.imread('figures/Katsura_2004_rw_volumes.png')
mwd_Cp_diagram = mpimg.imread('frost_2003_figures/Cp_wadsleyite_Jacobs_2007.png')
mrw_Cp_diagram = mpimg.imread('frost_2003_figures/Cp_ringwoodite_Jacobs_2007.png')

fa_Cp_diagram = mpimg.imread('frost_2003_figures/fa_Cp_Benisek_2012.png')
frw_Cp_diagram = mpimg.imread('frost_2003_figures/frw_Cp_Yong_2007.png')


fo_phase_diagram = mpimg.imread('frost_2003_figures/Mg2SiO4_phase_diagram_Jacobs_2017.png')
fa_phase_diagram = mpimg.imread('frost_2003_figures/Fe2SiO4_phase_diagram_Yagi_1987.png')
fa_phase_diagram2 = mpimg.imread('frost_2003_figures/Fe2SiO4_phase_diagram_Jacobs_2001.png')


# LOAD DATA
# Endmember reaction conditions
with open('data/endmember_transitions.dat', 'r') as f:
    ds = [line.split() for line in f if line.split() != [] and line[0] != '#']
    
endmember_transitions = [(float(d[0])*1.e9, float(d[1])*1.e9, float(d[2]), d[3], d[4]) for d in ds]

    
# Frost partitioning data
with open('data/Frost_2003_chemical_analyses.dat', 'r') as f:
    ds = [line.split() for line in f if line.split() != [] and line[0] != '#']

all_runs = [d[0] for d in ds]
set_runs = list(set([d[0] for d in ds]))
all_conditions = [(float(ds[all_runs.index(run)][2])*1.e9,
               float(ds[all_runs.index(run)][3])) for run in set_runs]
all_chambers = [list(set([d[1] for d in ds if d[0] == run])) for run in set_runs]

all_compositions = []
for i, run in enumerate(set_runs):
    all_compositions.append([])
    run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
    for j, chamber in enumerate(all_chambers[i]):
        chamber_indices = [run_idx for run_idx in run_indices
                           if (ds[run_idx][1] == chamber and
                               ds[run_idx][4] != 'cen' and
                               ds[run_idx][4] != 'anB' and
                               ds[run_idx][4] != 'mag')]
        if len(chamber_indices) > 1:
            all_compositions[-1].append({ds[idx][4]: map(float, ds[idx][5:13])
                                         for idx in chamber_indices})
            #print(run, chamber, [ds[idx][4] for idx in chamber_indices])

# Take only the data at > 2.2 GPa (i.e. not the PC experiment)
compositions, conditions, chambers, runs = zip(*[[all_compositions[i],
                                                  all_conditions[i],
                                                  all_chambers[i],
                                                  set_runs[i]]
                                                 for i, c in enumerate(all_compositions)
                                                 if all_conditions[i][0] > 2.2e9])

def endmember_affinity(P, T, m1, m2):
    m1.set_state(P, T)
    m2.set_state(P, T)
    return m1.gibbs - m2.gibbs

# compositions is a nested list of run, chamber, and minerals in that chamber. The minerals in each chamber are contained in a dictionary. Each dictionary has mineral attributes ('mw', 'ol', 'wad', 'ring', 'AnB'), which are simple lists of floats of atomic compositions in the following order:
# [Fe, Feerr, Al, Alerr, Si, Sierr, Mg, Mgerr]

        
# Now, let's load the minerals and solid solutions we wish to use
per = burnman.minerals.HHPH_2013.per()
wus = burnman.minerals.HHPH_2013.fper()
fo = burnman.minerals.HHPH_2013.fo()
fa = burnman.minerals.HHPH_2013.fa()
mwd = burnman.minerals.HHPH_2013.mwd()
fwd = burnman.minerals.HHPH_2013.fwd()
mrw = burnman.minerals.HHPH_2013.mrw()
frw = burnman.minerals.HHPH_2013.frw()
mins = [per, wus, fo, fa, mwd, fwd, mrw, frw]

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

per.params['S_0_orig'] = [26.9, 0.1] # exp paper reported in Jacobs et al., 2017
wus.params['S_0_orig'] = [58., 6.] # exp paper reported in Jacobs et al., 2017

fo.params['S_0_orig'] = [94.0, 0.1] # Dachs et al., 2007
fa.params['S_0_orig'] = [151.4, 0.1] # Dachs et al., 2007

mwd.params['S_0_orig'] = [86.4, 0.4] # exp paper reported in Jacobs et al., 2017
fwd.params['S_0_orig'] = [140.2, 10.] # same as Yong et al., 2007 for rw, but with bigger error

mrw.params['S_0_orig'] = [82.7, 0.5] # exp paper reported in Jacobs et al., 2017
frw.params['S_0_orig'] = [140.2, 1.] # Yong et al., 2007; formal error is 0.4


for m in mins:
    m.params['S_0'] = m.params['S_0_orig'][0]


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
               energy_interaction=[[15.e3]],
               volume_interaction=[[0.e-7]])
rw = Solution(name = 'ringwoodite',
              solution_type ='symmetric',
              endmembers=[[mrw, '[Mg]2SiO4'], [frw, '[Fe]2SiO4']],
              energy_interaction=[[8.2e3]],
              volume_interaction=[[0.e-7]]) 

solutions = {'mw': fper,
             'ol': ol,
             'wad': wad,
             'ring': rw}

endmembers = {'per': per,
              'wus': wus,
              'fo': fo,
              'fa': fa,
              'mwd': mwd,
              'fwd': fwd,
              'mrw': mrw,
              'frw': frw}
              


# For each mineral pair, use the pressure and temperature to calculate the independent reaction affinities

# We want to minimize these by varying:
# - Endmember gibbs
# - Endmember thermal expansivities (essentially volumes at 1673 K)
# - Solution model (3)
# - Pressure error for each experiment (15)

# The pressure errors for each experiment have a Gaussian uncertainty of 0.5 GPa centered on the nominal pressure. 

def rms_misfit_affinities(args):
    # set parameters 
    Perr = set_params(args)
    
    misfit = 0.
    n=0.
    for i, run in enumerate(compositions):
        P, T = conditions[i]
        P += Perr[i]
        
        for j, chamber in enumerate(compositions[i]):
            mus = []
            mus_plus = []
            mus_minus = []
            minerals = []
            for (k, c) in compositions[i][j].iteritems():
                minerals.append(k)
                FeoverFeMg = c[0]/(c[0] + c[6])
                solutions[k].set_composition([1. - FeoverFeMg, FeoverFeMg])
                solutions[k].set_state(P, T)
                if k == 'mw':
                    mus.append(solutions[k].partial_gibbs)
                else:
                    mus.append(solutions[k].partial_gibbs/2.)
                    
                FeoverFeMg_plus = (c[0]+c[1]/2.)/(c[0]+c[1]/2. + c[6]-c[7]/2.)
                solutions[k].set_composition([1. - FeoverFeMg_plus, FeoverFeMg_plus])
                solutions[k].set_state(P, T)
                if k == 'mw':
                    mus_plus.append(solutions[k].partial_gibbs)
                else:
                    mus_plus.append(solutions[k].partial_gibbs/2.)
                    
                FeoverFeMg_minus = (c[0]-c[1]/2.)/(c[0]-c[1]/2. + c[6]+c[7]/2.)
                solutions[k].set_composition([1. - FeoverFeMg_minus, FeoverFeMg_minus])
                solutions[k].set_state(P, T)
                if k == 'mw':
                    mus_minus.append(solutions[k].partial_gibbs)
                else:
                    mus_minus.append(solutions[k].partial_gibbs/2.)

            
            ol_polymorph_indices = [idx for idx in range(len(minerals))
                                    if minerals[idx] != 'mw']
            mw_idx = minerals.index('mw')

            # First, deal with mw equilibria
            for l in ol_polymorph_indices:
                dGdFsigmaF = np.array(mus_plus) - np.array(mus_minus)
                dGdFsigmaF = dGdFsigmaF[:,0] - dGdFsigmaF[:,1]
                n+=1
                dG = (mus[mw_idx][0] - mus[mw_idx][1] - mus[l][0] + mus[l][1])
                sigma_dG = np.sqrt(dGdFsigmaF[mw_idx]*dGdFsigmaF[mw_idx] + dGdFsigmaF[l]*dGdFsigmaF[l])
                misfit += np.power(dG/sigma_dG, 2.) # Mg endmembers

            if len(ol_polymorph_indices) == 2:
                i1, i2 = ol_polymorph_indices
                
                dGdFsigmaF = np.array(mus_plus) - np.array(mus_minus)
                dGs = mus[i1] - mus[i2]
                sigma_dGs = np.sqrt(dGdFsigmaF[i1]*dGdFsigmaF[i1] +
                                    dGdFsigmaF[i2]*dGdFsigmaF[i2])
                misfit += np.power(dGs[0]/sigma_dGs[0], 2.) # Mg endmembers
                misfit += np.power(dGs[1]/sigma_dGs[1], 2.) # Fe endmembers
            if len(ol_polymorph_indices) == 3:
                i1, i2, i3 = ol_polymorph_indices
                dGdFsigmaF = np.array(mus_plus) - np.array(mus_minus)
                dGs = mus[i1] - mus[i2]
                sigma_dGs = np.sqrt(dGdFsigmaF[i1]*dGdFsigmaF[i1] +
                                    dGdFsigmaF[i2]*dGdFsigmaF[i2])
                misfit += np.power(dGs[0]/sigma_dGs[0], 2.) # Mg endmembers 
                misfit += np.power(dGs[1]/sigma_dGs[1], 2.) # Fe endmembers
                
                dGs = mus[i1] - mus[i3]
                sigma_dGs = np.sqrt(dGdFsigmaF[i1]*dGdFsigmaF[i1] +
                                    dGdFsigmaF[i3]*dGdFsigmaF[i3])
                misfit += np.power(dGs[0]/sigma_dGs[0], 2.) # Mg endmembers
                misfit += np.power(dGs[1]/sigma_dGs[1], 2.) # Fe endmembers
                
                dGs = mus[i2] - mus[i3]
                sigma_dGs = np.sqrt(dGdFsigmaF[i2]*dGdFsigmaF[i2] +
                                    dGdFsigmaF[i3]*dGdFsigmaF[i3])
                misfit += np.power(dGs[0]/sigma_dGs[0], 2.) # Mg endmembers
                misfit += np.power(dGs[1]/sigma_dGs[1], 2.) # Fe endmembers


    # apply strong priors to stable endmember transition pressures
    max_P_err = 0.
    for (Pnom, P_sigma, Tnom, m1, m2) in endmember_transitions:
        dP = fsolve(endmember_affinity, Pnom, args=(Tnom, endmembers[m1], endmembers[m2]))[0] - Pnom
        #print('dP({0}, {1}): {2}'.format(m1, m2, dP/1.e9))
        misfit += np.power(dP/P_sigma, 2.)
        max_weighted_P_err = np.max([np.abs(dP/P_sigma), max_P_err])
    print('Maximum dP/sigma(P): {0}'.format(max_weighted_P_err))
    
    # apply prior to fwd instability ove a wide temperature range
    T = 473.15
    dP = (fsolve(endmember_affinity, 6.2e9, args=(T, fa, fwd))[0] -
          fsolve(endmember_affinity, 6.2e9, args=(T, fa, frw))[0])
    if dP < 0.1e9:
        misfit += np.power(np.abs(dP-0.1e9)/0.1e9, 2.)
    #print('dP({0}, {1}): {2} at {3} K'.format(fwd.name, frw.name, dP/1.e9, T))
    T = 1873.15
    dP = (fsolve(endmember_affinity, 6.2e9, args=(T, fa, fwd))[0] -
          fsolve(endmember_affinity, 6.2e9, args=(T, fa, frw))[0])
    if dP < 0.1e9:
        misfit += np.power(np.abs(dP-0.1e9)/0.1e9, 2.)
    #print('dP({0}, {1}): {2} at {3} K'.format(fwd.name, frw.name, dP/1.e9, T))
    
    
    # include original alphas as priors
    for (ms, sigma) in [([per, fo, fa, mrw], 2.e-7),
                        ([wus, mwd, frw], 5.e-7),
                        ([fwd], 20.e-7)]:
        for m in ms:
            d_alpha = m.params['a_0_orig'] - m.params['a_0']
            misfit += np.power(d_alpha/sigma, 2.)
            
    # include original entropies as priors
    for m in mins:
        misfit += np.power((m.params['S_0_orig'][0] - m.params['S_0'])/m.params['S_0_orig'][1], 2.)

    # add prior for olivine interaction parameter
    misfit += np.power((ol.solution_model.We[0][1] - 5.2e3)/(1.e3), 2.)

    # Nominal pressure error of experiments
    misfit += np.sum(Perr*Perr)/(0.5e9*0.5e9)
    n += len(Perr)
    rms_misfit = np.sqrt(misfit)/float(n)
    print(rms_misfit)
    print(*args, sep = ", ")
    return rms_misfit

def get_params(Perr):
    args = [fper.endmembers[0][0].params['H_0']*1.e-3,
            fper.endmembers[1][0].params['H_0']*1.e-3,
            wad.endmembers[0][0].params['H_0']*1.e-3,
            wad.endmembers[1][0].params['H_0']*1.e-3,
            rw.endmembers[0][0].params['H_0']*1.e-3,
            rw.endmembers[1][0].params['H_0']*1.e-3,
            fper.endmembers[0][0].params['S_0'],
            fper.endmembers[1][0].params['S_0'],
            ol.endmembers[0][0].params['S_0'],
            ol.endmembers[1][0].params['S_0'],
            wad.endmembers[0][0].params['S_0'],
            wad.endmembers[1][0].params['S_0'],
            rw.endmembers[0][0].params['S_0'],
            rw.endmembers[1][0].params['S_0'],
            fper.endmembers[0][0].params['a_0']*1.e5,
            fper.endmembers[1][0].params['a_0']*1.e5, 
            ol.endmembers[0][0].params['a_0']*1.e5,
            ol.endmembers[1][0].params['a_0']*1.e5, 
            wad.endmembers[0][0].params['a_0']*1.e5,
            wad.endmembers[1][0].params['a_0']*1.e5,
            rw.endmembers[0][0].params['a_0']*1.e5,
            rw.endmembers[1][0].params['a_0']*1.e5,
            fper.solution_model.We[0][1]*1.e-3,
            ol.solution_model.We[0][1]*1.e-3,
            wad.solution_model.We[0][1]*1.e-3,
            rw.solution_model.We[0][1]*1.e-3]

    args.extend(Perr)
    return args

def set_params(args):
    fper.endmembers[0][0].params['H_0'] = args[0]*1.e3
    fper.endmembers[1][0].params['H_0'] = args[1]*1.e3
    wad.endmembers[0][0].params['H_0'] = args[2]*1.e3
    wad.endmembers[1][0].params['H_0'] = args[3]*1.e3
    rw.endmembers[0][0].params['H_0'] = args[4]*1.e3
    rw.endmembers[1][0].params['H_0'] = args[5]*1.e3
    
    fper.endmembers[0][0].params['S_0'] = args[6]
    fper.endmembers[1][0].params['S_0'] = args[7]
    ol.endmembers[0][0].params['S_0'] = args[8]
    ol.endmembers[1][0].params['S_0'] = args[9]
    wad.endmembers[0][0].params['S_0'] = args[10]
    wad.endmembers[1][0].params['S_0'] = args[11]
    rw.endmembers[0][0].params['S_0'] = args[12]
    rw.endmembers[1][0].params['S_0'] = args[13]
    
    fper.endmembers[0][0].params['a_0'] = args[14]*1.e-5
    fper.endmembers[1][0].params['a_0'] = args[15]*1.e-5
    ol.endmembers[0][0].params['a_0'] = args[16]*1.e-5
    ol.endmembers[1][0].params['a_0'] = args[17]*1.e-5
    wad.endmembers[0][0].params['a_0'] = args[18]*1.e-5
    wad.endmembers[1][0].params['a_0'] = args[19]*1.e-5
    rw.endmembers[0][0].params['a_0'] = args[20]*1.e-5
    rw.endmembers[1][0].params['a_0'] = args[21]*1.e-5
    
    
    fper.solution_model.We[0][1] = args[22]*1.e3
    ol.solution_model.We[0][1] = args[23]*1.e3
    wad.solution_model.We[0][1] = args[24]*1.e3
    rw.solution_model.We[0][1] = args[25]*1.e3
    
    Perr = np.array(args[26:])*1.e9 # arguments in GPa
    return Perr

fper.endmembers[0][0].params['a_0_orig'] = fper.endmembers[0][0].params['a_0']
fper.endmembers[1][0].params['a_0_orig'] = fper.endmembers[1][0].params['a_0']
ol.endmembers[0][0].params['a_0_orig'] = ol.endmembers[0][0].params['a_0']
ol.endmembers[1][0].params['a_0_orig'] = ol.endmembers[1][0].params['a_0']
wad.endmembers[0][0].params['a_0_orig'] = wad.endmembers[0][0].params['a_0']
wad.endmembers[1][0].params['a_0_orig'] = wad.endmembers[1][0].params['a_0']
rw.endmembers[0][0].params['a_0_orig'] = rw.endmembers[0][0].params['a_0']
rw.endmembers[1][0].params['a_0_orig'] = rw.endmembers[1][0].params['a_0']

args = get_params([0.]*len(compositions))

# mwd, frw: 0, 4
args = np.array([-6.04659013e+02, -2.62364260e+02, -2.14497024e+03, -1.46392867e+03,
                 -2.13447658e+03, -1.46210882e+03,  2.68990125e+01,  6.21490796e+01,
                 9.41294929e+01,  1.51419065e+02,  8.50428195e+01,  1.42774157e+02,
                 8.15933125e+01,  1.38208471e+02,  3.07918132e+00,  3.39458748e+00,
                 2.84213200e+00,  2.77623376e+00,  2.13474857e+00,  1.99628333e+00,
                 2.22408127e+00,  1.94133547e+00,  1.11333420e+01,  5.00437164e+00,
                 1.57423442e+01,  7.57846521e+00, -8.27432755e-02, -1.17414129e-01,
                 7.46439836e-02, -4.59182495e-01, -3.49214631e-02,  5.47369309e-01,
                 -5.92176590e-02, -3.19531339e-01, -1.09428616e+00,  4.54302255e-01,
                 2.78984832e-01, -6.18452574e-02,  2.07796321e-01, -3.01902796e-01,
                 9.23075711e-01,  1.57796037e-01, -1.52361107e-02,  8.03159438e-01,
                 -5.63615619e-01, -2.41119543e-01, -1.18825217e+00, -3.69594607e-01,
                 -1.09254906e-01])

# -2, 2
args = [-604.647266862, -262.376007143, -2144.95376795, -1464.01677766, -2134.50100115, -1462.45368056, 26.8827811498, 62.0757801241, 94.1029147167, 151.315576339, 85.020050769, 142.916251188, 81.5759757215, 138.689079496, 3.08481852749, 3.39261851987, 2.84674756023, 2.78017942756, 2.13901107531, 1.88429601859, 2.23400101287, 1.92649235633, 11.1809223691, 5.01012836854, 15.7682972952, 7.58840926312, -0.0755892458475, -0.116407545658, 0.119111212601, -0.464280106369, -0.0312889519228, 0.547956834137, -0.0594776346765, -0.293532918493, -1.06488615805, 0.458814588423, 0.261099725295, -0.0575384489927, 0.209104152415, -0.298408086129, 0.971880455557, 0.162368404455, -0.00664251485364, 0.798451629717, -0.553358268055, -0.253536253022, -1.12769700359, -0.367107772082, -0.112662013682]

# -4, 0
args = [-604.493422398, -262.529852822, -2144.9927018, -1464.2929659, -2134.56565173, -1462.90077172, 26.9147408469, 61.9328730884, 94.103566932, 151.385707994, 85.0188746001, 143.383139214, 81.5373429227, 139.374737812, 3.07554445302, 3.3920453455, 2.83343418781, 2.77722968253, 2.13297021438, 1.81690472934, 2.22591418552, 1.92930491706, 11.1786326401, 5.05694782305, 15.9068149163, 7.55642904454, -0.065361281466, -0.119359756732, 0.272504687232, -0.476241692776, -0.0188794981785, 0.551256076128, -0.0597029535157, -0.249504442624, -1.00779670572, 0.473934565401, 0.197036213563, -0.0504916269646, 0.212209687562, -0.280883192136, 1.02825926289, 0.179566055035, 0.0157209653469, 0.811019505664, -0.533514694072, -0.269069377221, -0.958649478777, -0.384788412933, -0.106961697389]

# -6, -2
args=np.array([-6.02747777e+02, -2.64275678e+02, -2.14499275e+03, -1.46610625e+03,
               -2.13450637e+03, -1.46547080e+03,  2.68996056e+01,  5.98323367e+01,
               9.41310288e+01,  1.51414041e+02,  8.50283980e+01,  1.44104565e+02,
               8.15643547e+01,  1.38707780e+02,  3.07887583e+00,  3.39632247e+00,
               2.84187197e+00,  2.77768230e+00,  2.13662249e+00,  2.00907629e+00,
               2.22453691e+00,  1.92937022e+00,  1.11480556e+01,  5.12628257e+00,
               1.57526115e+01,  7.62046929e+00, -7.40250600e-02, -1.25124869e-01,
               1.41585944e-02, -4.45970654e-01, -3.14972926e-02,  5.49313020e-01,
               -5.86154820e-02, -2.98316475e-01, -1.06471059e+00,  4.63321961e-01,
               2.22041502e-01, -5.41369376e-02,  2.16523549e-01, -2.92372606e-01,
               9.62297419e-01,  1.66647706e-01, -7.30509033e-03,  8.14478439e-01,
               -5.54772564e-01, -2.32384445e-01, -1.14896987e+00, -4.05818126e-01,
               -1.04205365e-01])

# THIS LINE RUNS THE MINIMIZATION!!!
if run_inversion == 'y' or run_inversion == 'Y' or run_inversion == 'yes' or run_inversion == 'Yes':
    print(minimize(rms_misfit_affinities, args, method='BFGS')) # , options={'eps': 1.e-02}))

Perr = set_params(args)

# Plot mrw EoS
pressures = np.linspace(1.e5, 25.e9, 101) 
plt.imshow(mrw_volume_diagram, extent=[0., 25., 35.5,40.5], aspect='auto')
for T in [300., 700., 1100., 1500., 1900.]:
    temperatures = pressures*0. + T
    plt.plot(pressures/1.e9, mrw.evaluate(['V'], pressures, temperatures)[0]*1.e6)
plt.show()


fig = plt.figure(figsize=(30, 15))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
ax[0].imshow(fo_phase_diagram, extent=[1000., 2600., 5, 30], aspect='auto')

temperatures = np.linspace(1000., 2600., 21)
ax[0].plot(temperatures, eqm_pressures(fo, mwd, temperatures)/1.e9, linewidth=4.)
ax[0].plot(temperatures, eqm_pressures(mwd, mrw, temperatures)/1.e9, linewidth=4.)

"""
ax[1].imshow(fa_phase_diagram, extent=[3., 7., 550.+273.15, 1350.+273.15], aspect='auto')
temperatures = np.linspace(550.+273.15, 1350.+273.15, 21)
ax[1].plot(eqm_pressures(fa, frw, temperatures)/1.e9, temperatures, linewidth=4.)
ax[1].plot(eqm_pressures(fa, fwd, temperatures)/1.e9, temperatures, linestyle=':', linewidth=4.)
"""

ax[1].imshow(fa_phase_diagram2, extent=[700., 1900., 0., 10.], aspect='auto')
temperatures = np.linspace(700., 1900., 21)
ax[1].plot(temperatures, eqm_pressures(fa, frw, temperatures)/1.e9, linewidth=4.)
ax[1].plot(temperatures, eqm_pressures(fa, fwd, temperatures)/1.e9, linestyle=':', linewidth=4.)


ax[0].set_xlabel('T (K)')
ax[0].set_ylabel('P (GPa)')
ax[1].set_xlabel('P (GPa)')
ax[1].set_ylabel('T (K)')
plt.show()


# PLOT VOLUMES COMPARED WITH OTHER DATASETS
class Murnaghan_EOS(object):
    def __init__(self, V0, K, Kprime):
        self.V0 = V0
        self.K = K
        self.Kprime = Kprime
        self.V = lambda P: self.V0*np.power(1. + P*(self.Kprime/self.K),
                                            -1./self.Kprime) 


M_fo  = Murnaghan_EOS(4.6053e-5, 95.7e9, 4.6)
M_mwd = Murnaghan_EOS(4.2206e-5, 146.2544e9, 4.21)
M_mrw = Murnaghan_EOS(4.1484e-5, 145.3028e9, 4.4)
M_per = Murnaghan_EOS(1.1932e-5, 125.9e9, 4.1)
M_py  = Murnaghan_EOS(11.8058e-5, 129.0e9, 4.)

M_fa = Murnaghan_EOS(4.8494e-5, 99.8484e9, 4.)
M_fwd = Murnaghan_EOS(4.4779e-5, 139.9958e9, 4.)
M_frw = Murnaghan_EOS(4.3813e-5, 160.781e9, 5.)
M_fper = Murnaghan_EOS(1.2911e-5, 152.6e9, 4.)
M_alm = Murnaghan_EOS(12.1153e-5, 120.7515e9, 5.5)

from burnman.minerals import HHPH_2013, SLB_2011
dss = [[M_per, M_fper, M_fo, M_fa, M_mwd, M_fwd, M_mrw, M_frw, 'Frost (2003)'],
       [HHPH_2013.per(), HHPH_2013.fper(),
        HHPH_2013.fo(), HHPH_2013.fa(),
        HHPH_2013.mwd(), HHPH_2013.fwd(),
        HHPH_2013.mrw(), HHPH_2013.frw(), 'HHPH'],
       [SLB_2011.periclase(), SLB_2011.wuestite(),
        SLB_2011.forsterite(), SLB_2011.fayalite(),
        SLB_2011.mg_wadsleyite(), SLB_2011.fe_wadsleyite(),
        SLB_2011.mg_ringwoodite(), SLB_2011.fe_ringwoodite(), 'SLB'],
       [per,  wus,  fo,  fa,  mwd,  fwd,  mrw,  frw, 'this study']]


fig = plt.figure(figsize=(24,12))
ax = [fig.add_subplot(2, 4, i) for i in range(1, 9)]
pressures = np.linspace(1.e5, 24.e9, 101)
T=1673.15
temperatures = pressures*0. + T
for i, ds in enumerate(dss):
    if i == 0:
        Vs = [m.V(pressures) for m in ds[0:-1]]
    else:
        Vs = [m.evaluate(['V'], pressures, temperatures)[0] for m in ds[0:-1]]
    if i==3:
        linewidth=3.
    else:
        linewidth=1.
        
    for j, V in enumerate(Vs):
            ax[j].plot(pressures/1.e9, Vs[j]*1.e6, label=ds[-1], linewidth=linewidth)
        

for i in range(0, 8):
    ax[i].legend(loc='best')
plt.show()


# FPER-OL POLYMORPH PARTITIONING
def affinity_ol_fper(v, x_ol, G, T, W_ol, W_fper):
    """
    G is deltaG = G_per + G_fa/2. - G_fper - G_fo/2.
    """
    x_fper = v[0]
    if np.abs(np.abs(x_ol - 0.5) - 0.5) < 1.e-10:
        v[0] = x_ol
        return 0.
    else:
        KD = ((x_ol*(1. - x_fper))/
              (x_fper*(1. - x_ol)))
        if KD < 0.:
            KD = 1.e-12
        return G - W_ol*(2.*x_ol - 1.) - W_fper*(1 - 2.*x_fper) + burnman.constants.gas_constant*T*np.log(KD)


# NOW PLOT THE FPER-OL POLYMORPH EQUILIBRIA


fig = plt.figure(figsize=(30,10))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

# OLIVINE
ax[0].imshow(ol_fper_img, extent=[0.0, 0.8, -45000., -5000.], aspect='auto')


viridis = cm.get_cmap('viridis', 101)
Pmin = 0.e9
Pmax = 15.e9

T = 1673.15
for P in [1.e5, 5.e9, 10.e9, 15.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (per.gibbs - wus.gibbs - fo.gibbs/2. + fa.gibbs/2.)
    W_ol = (ol.solution_model.We[0][1] + ol.solution_model.Wv[0][1] * P) /2. # 1 cation
    W_fper = fper.solution_model.We[0][1] + fper.solution_model.Wv[0][1] * P
    
    x_ols = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_ol], args=(x_ol, G, T, W_ol, W_fper))[0]
                        for x_ol in x_ols])
    KDs = ((x_ols*(1. - x_fpers))/
           (x_fpers*(1. - x_ols)))
    ax[0].plot(x_ols, burnman.constants.gas_constant*T*np.log(KDs), color = viridis((P-Pmin)/(Pmax-Pmin)), linewidth=3., label='{0} GPa'.format(P/1.e9))



pressures = []
x_ols = []
x_fpers = []
RTlnKDs = []
for i, run in enumerate(compositions):
    P, T = conditions[i]
    for j, chamber in enumerate(compositions[i]):
        if 'ol' in compositions[i][j]:
            pressures.append(conditions[i][0])
            x_ols.append(compositions[i][j]['ol'][0]/(compositions[i][j]['ol'][0] +
                                                      compositions[i][j]['ol'][6]))
            x_fpers.append(compositions[i][j]['mw'][0]/(compositions[i][j]['mw'][0] +
                                                          compositions[i][j]['mw'][6]))
            RTlnKDs.append(burnman.constants.gas_constant*T*np.log((x_ols[-1]*(1. - x_fpers[-1]))/
                                                                   (x_fpers[-1]*(1. - x_ols[-1]))))
            
ax[0].scatter(x_ols, RTlnKDs, c=pressures, s=80., label='data', cmap=viridis, vmin=Pmin, vmax=Pmax)



ax[0].set_xlim(0., 0.8)
ax[0].legend(loc='best')


# WADSLEYITE
ax[1].imshow(wad_fper_img, extent=[0.0, 0.4, -25000., -5000.], aspect='auto')

viridis = cm.get_cmap('viridis', 101)
Pmin = 10.e9
Pmax = 18.e9


T = 1673.15
for P in [10.e9, 12.e9, 14.e9, 16.e9, 18.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (per.gibbs - wus.gibbs - mwd.gibbs/2. + fwd.gibbs/2.)
    W_wad = (wad.solution_model.We[0][1] + wad.solution_model.Wv[0][1] * P)/2. # 1 cation
    W_fper = fper.solution_model.We[0][1] + fper.solution_model.Wv[0][1] * P
    
    x_wads = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_wad], args=(x_wad, G, T, W_wad, W_fper))[0]
                        for x_wad in x_wads])
    KDs = ((x_wads*(1. - x_fpers))/
           (x_fpers*(1. - x_wads)))
    ax[1].plot(x_wads, burnman.constants.gas_constant*T*np.log(KDs), color = viridis((P-Pmin)/(Pmax-Pmin)), linewidth=3., label='{0} GPa'.format(P/1.e9))

pressures = []
x_wads = []
x_fpers = []
RTlnKDs = []
for i, run in enumerate(compositions):
    P, T = conditions[i]
    for j, chamber in enumerate(compositions[i]):
        if 'wad' in compositions[i][j]:
            pressures.append(conditions[i][0])
            x_wads.append(compositions[i][j]['wad'][0]/(compositions[i][j]['wad'][0] +
                                                      compositions[i][j]['wad'][6]))
            x_fpers.append(compositions[i][j]['mw'][0]/(compositions[i][j]['mw'][0] +
                                                          compositions[i][j]['mw'][6]))
            RTlnKDs.append(burnman.constants.gas_constant*T*np.log((x_wads[-1]*(1. - x_fpers[-1]))/
                                                                   (x_fpers[-1]*(1. - x_wads[-1]))))
            
ax[1].scatter(x_wads, RTlnKDs, c=pressures, s=80., label='data', cmap=viridis, vmin=Pmin, vmax=Pmax)

ax[1].set_xlim(0., 0.4)
ax[1].legend(loc='best')


# RINGWOODITE
#ax[2].imshow(rw_fper_part_img, extent=[0.0, 1., 0., 1.], aspect='auto')

viridis = cm.get_cmap('viridis', 101)
Pmin = 10.e9
Pmax = 24.e9


T = 1673.15
for P in [10.e9, 12.5e9, 15.e9, 17.5e9, 20.e9]:
    for m in mins:
        m.set_state(P, T)
    G = (per.gibbs - wus.gibbs - mrw.gibbs/2. + frw.gibbs/2.) 
    W_rw = (rw.solution_model.We[0][1]  + rw.solution_model.Wv[0][1] * P)/2. # 1 cation
    W_fper = fper.solution_model.We[0][1] + fper.solution_model.Wv[0][1] * P
    
    x_rws = np.linspace(0.00001, 0.99999, 101)
    x_fpers = np.array([fsolve(affinity_ol_fper, [x_rw], args=(x_rw, G, T, W_rw, W_fper))[0]
                        for x_rw in x_rws])

    ax[2].plot(x_rws, x_fpers, color=viridis((P-Pmin)/(Pmax-Pmin)), linewidth=3., label=P/1.e9)


pressures = []
x_rws = []
x_fpers = []
for i, run in enumerate(compositions):
    for j, chamber in enumerate(compositions[i]):
        if 'ring' in compositions[i][j]:
            pressures.append(conditions[i][0])
            x_rws.append(compositions[i][j]['ring'][0]/(compositions[i][j]['ring'][0] +
                                                      compositions[i][j]['ring'][6]))
            x_fpers.append(compositions[i][j]['mw'][0]/(compositions[i][j]['mw'][0] +
                                                        compositions[i][j]['mw'][6]))
            
c = ax[2].scatter(x_rws, x_fpers, c=pressures, s=80., label='data', cmap=viridis, vmin=Pmin, vmax=Pmax)

ax[2].set_xlim(0., 1.)
ax[2].set_ylim(0., 1.)
ax[2].legend(loc='best')
plt.show()

# BINARY PHASE DIAGRAM

#plt.imshow(ol_polymorph_img, extent=[0., 1., 6., 20.], aspect='auto')
#plt.imshow(ol_polymorph_img_1000C, extent=[-0.01, 1.005, 4., 21.], aspect='auto')

for (T0, color) in [(1273.15, 'blue'),
                    (1673.15, 'orange'),
                    (2073.15, 'purple')]:

    x_m1 = 0.3

    composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1.-x_m1), 'Si': 1., 'O': 4.}
    rw.guess = np.array([1. - x_m1, x_m1])
    wad.guess = np.array([1. - x_m1, x_m1])
    ol.guess = np.array([1. - x_m1, x_m1])
    assemblage = burnman.Composite([ol, wad, rw])
    equality_constraints = [('T', T0), ('phase_proportion', (ol, 0.0))]
    sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
    P_inv = assemblage.pressure
    x_ol_inv = assemblage.phases[0].molar_fractions[1]
    x_wad_inv = assemblage.phases[1].molar_fractions[1]
    x_rw_inv = assemblage.phases[2].molar_fractions[1]

    for (m1, m2) in [(wad, ol), (wad, rw), (ol, rw)]:
        composition = {'Fe': 0., 'Mg': 2., 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([m1.endmembers[0][0], m2.endmembers[0][0]])
        equality_constraints = [('T', T0), ('phase_proportion', (m1.endmembers[0][0], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
        P1 = assemblage.pressure
        composition = {'Fe': 2., 'Mg': 0., 'Si': 1., 'O': 4.}
        assemblage = burnman.Composite([m1.endmembers[1][0], m2.endmembers[1][0]])
        equality_constraints = [('T', T0), ('phase_proportion', (m1.endmembers[1][0], 0.0))]
        sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False)
        P0 = assemblage.pressure
        print(P0/1.e9, P1/1.e9)
        if m1 is wad:
            x_m1s = np.linspace(0.001, x_wad_inv, 21)
        else:
            x_m1s = np.linspace(x_ol_inv, 0.999, 21)
        
        pressures = np.empty_like(x_m1s)
        x_m2s = np.empty_like(x_m1s)
        for i, x_m1 in enumerate(x_m1s):
            composition = {'Fe': 2.*x_m1, 'Mg': 2.*(1. - x_m1), 'Si': 1., 'O': 4.}
            assemblage = burnman.Composite([m1, m2])
            assemblage.set_state(P1, T0)
            m1.guess = np.array([1. - x_m1, x_m1])
            m2.guess = np.array([1. - x_m1, x_m1])
            equality_constraints = [('T', T0), ('phase_proportion', (m2, 0.0))]
            sol, prm = burnman.equilibrate(composition, assemblage, equality_constraints,
                                           initial_state_from_assemblage=True,
                                           store_iterates=False)
            x_m2s[i] = m2.molar_fractions[1]
            pressures[i] = assemblage.pressure

        plt.plot(x_m1s, pressures/1.e9, linewidth=3., color=color)
        plt.plot(x_m2s, pressures/1.e9, linewidth=3., color=color, label='{0} K'.format(T0))
    plt.plot([x_ol_inv, x_rw_inv], [P_inv/1.e9, P_inv/1.e9], linewidth=3., color=color)
        
p_x_m = {'ol': [], 'wad': [], 'ring': []}
for i, run in enumerate(compositions):
    P, T = conditions[i]
    for j, chamber in enumerate(compositions[i]):
        n_ol_polymorphs = 0
        for m in ['ol', 'wad', 'ring']:
            if m in compositions[i][j]:
                n_ol_polymorphs += 1
                
        if n_ol_polymorphs > 1:
            for m in ['ol', 'wad', 'ring']:
                if m in compositions[i][j]:
                    p_x_m[m].append([P, compositions[i][j][m][0]/(compositions[i][j][m][0] +
                                                                  compositions[i][j][m][6])])

for m in ['ol', 'wad', 'ring']:
    pressures, xs = np.array(zip(*p_x_m[m]))
    plt.scatter(xs, pressures/1.e9, s=80., label='data')

plt.legend()
plt.show()
