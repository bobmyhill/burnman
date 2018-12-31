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

    
# Frost fper-ol-wad-rw partitioning data
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
F2003_mw_compositions, F2003_mw_conditions, F2003_mw_chambers, F2003_mw_runs = zip(*[[all_compositions[i],
                                                                                      all_conditions[i],
                                                                                      all_chambers[i],
                                                                                      set_runs[i]]
                                                                                     for i, c in enumerate(all_compositions)
                                                                                     if all_conditions[i][0] > 2.2e9])


# Garnet-olivine partitioning data
ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_ol_gt_KD.dat')
ol_gt_data[:,0] *= 1.e9 # P (GPa) to P (Pa)


# Frost fper-gt-ol-wad-rw partitioning data
with open('data/Frost_2003_FMASO_garnet_analyses.dat', 'r') as f:
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
                           if (ds[run_idx][1] == chamber)]
        if len(chamber_indices) > 1:
            # Fe, Al, Si, Mg
            all_compositions[-1].append({ds[idx][4]: map(float, ds[idx][5:13])
                                         for idx in chamber_indices})

# Take only the data at > 2.2 GPa (i.e. not the PC experiment)
F03_gt_compositions, F03_gt_conditions, F03_gt_chambers, F03_gt_runs = zip(*[[all_compositions[i],
                                                                              all_conditions[i],
                                                                              all_chambers[i],
                                                                              set_runs[i]]
                                                                             for i, c in enumerate(all_compositions)])



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
    for i, run in enumerate(F2003_mw_compositions):
        P, T = F2003_mw_conditions[i]
        P += Perr[i]
        
        for j, chamber in enumerate(F2003_mw_compositions[i]):
            mus = []
            mus_plus = []
            mus_minus = []
            minerals = []

            for (k, c) in F2003_mw_compositions[i][j].iteritems():
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


    # Run through olivine-garnet data
    for P, T, XMgOl, lnKD, lnKDerr in ol_gt_data:
        # KD is (XGtFe*XOlMg)/(XGtMg*XOlFe)
        KD = np.exp(lnKD)
        XMgGt = 1./( 1. + ((1. - XMgOl)/XMgOl)*KD)
        gt.set_composition([XMgGt, 1. - XMgGt])
        ol.set_composition([XMgOl, 1. - XMgOl])
        gt.set_state(P, T)
        ol.set_state(P, T)
        mus = list(gt.partial_gibbs/3.)
        mus.extend(list(ol.partial_gibbs/2.))

        dXMgGtdlnKD = -(1. - XMgOl)*KD/(XMgOl * np.power( (1. - XMgOl)*KD/XMgOl + 1., 2. ))
        XMgGterr = dXMgGtdlnKD*lnKDerr
        XMgGt_plus = XMgGt + XMgGterr/2.
        gt.set_composition([XMgGt_plus, 1. - XMgGt_plus])
        mus_plus = list(gt.partial_gibbs/3.)
        mus_plus.extend(list(ol.partial_gibbs/2.))

        XMgGt_minus = XMgGt - XMgGterr/2.
        gt.set_composition([XMgGt_minus, 1. - XMgGt_minus])
        mus_minus = list(gt.partial_gibbs/3.)
        mus_minus.extend(list(ol.partial_gibbs/2.))


        dGdFsigmaF = np.array(mus_plus) - np.array(mus_minus)
        dGdFsigmaF = np.array([dGdFsigmaF[0] - dGdFsigmaF[1],
                               dGdFsigmaF[2] - dGdFsigmaF[3]]) # Mg-Fe for all phases

        dG = mus[0] - mus[1] - mus[2] + mus[3]
        sigma_dG = np.sqrt(dGdFsigmaF[0]*dGdFsigmaF[0] + dGdFsigmaF[1]*dGdFsigmaF[1])
        misfit += np.power(dG/sigma_dG, 2.)

                
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
            gt.endmembers[1][0].params['H_0']*1.e-3,
            fper.endmembers[0][0].params['S_0'],
            fper.endmembers[1][0].params['S_0'],
            ol.endmembers[0][0].params['S_0'],
            ol.endmembers[1][0].params['S_0'],
            wad.endmembers[0][0].params['S_0'],
            wad.endmembers[1][0].params['S_0'],
            rw.endmembers[0][0].params['S_0'],
            rw.endmembers[1][0].params['S_0'],
            gt.endmembers[1][0].params['S_0'],
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
            rw.solution_model.We[0][1]*1.e-3,
            gt.solution_model.We[0][1]*1.e-3]

    args.extend(Perr)
    return args

def set_params(args):
    fper.endmembers[0][0].params['H_0'] = args[0]*1.e3
    fper.endmembers[1][0].params['H_0'] = args[1]*1.e3
    wad.endmembers[0][0].params['H_0'] = args[2]*1.e3
    wad.endmembers[1][0].params['H_0'] = args[3]*1.e3
    rw.endmembers[0][0].params['H_0'] = args[4]*1.e3
    rw.endmembers[1][0].params['H_0'] = args[5]*1.e3
    gt.endmembers[1][0].params['H_0'] = args[6]*1.e3
    
    fper.endmembers[0][0].params['S_0'] = args[7]
    fper.endmembers[1][0].params['S_0'] = args[8]
    ol.endmembers[0][0].params['S_0'] = args[9]
    ol.endmembers[1][0].params['S_0'] = args[10]
    wad.endmembers[0][0].params['S_0'] = args[11]
    wad.endmembers[1][0].params['S_0'] = args[12]
    rw.endmembers[0][0].params['S_0'] = args[13]
    rw.endmembers[1][0].params['S_0'] = args[14]
    gt.endmembers[1][0].params['S_0'] = args[15]
    
    fper.endmembers[0][0].params['a_0'] = args[16]*1.e-5
    fper.endmembers[1][0].params['a_0'] = args[17]*1.e-5
    ol.endmembers[0][0].params['a_0'] = args[18]*1.e-5
    ol.endmembers[1][0].params['a_0'] = args[19]*1.e-5
    wad.endmembers[0][0].params['a_0'] = args[20]*1.e-5
    wad.endmembers[1][0].params['a_0'] = args[21]*1.e-5
    rw.endmembers[0][0].params['a_0'] = args[22]*1.e-5
    rw.endmembers[1][0].params['a_0'] = args[23]*1.e-5
    
    
    fper.solution_model.We[0][1] = args[24]*1.e3
    ol.solution_model.We[0][1] = args[25]*1.e3
    wad.solution_model.We[0][1] = args[26]*1.e3
    rw.solution_model.We[0][1] = args[27]*1.e3
    gt.solution_model.We[0][1] = args[28]*1.e3
    
    Perr = np.array(args[29:])*1.e9 # arguments in GPa
    return Perr

fper.endmembers[0][0].params['a_0_orig'] = fper.endmembers[0][0].params['a_0']
fper.endmembers[1][0].params['a_0_orig'] = fper.endmembers[1][0].params['a_0']
ol.endmembers[0][0].params['a_0_orig'] = ol.endmembers[0][0].params['a_0']
ol.endmembers[1][0].params['a_0_orig'] = ol.endmembers[1][0].params['a_0']
wad.endmembers[0][0].params['a_0_orig'] = wad.endmembers[0][0].params['a_0']
wad.endmembers[1][0].params['a_0_orig'] = wad.endmembers[1][0].params['a_0']
rw.endmembers[0][0].params['a_0_orig'] = rw.endmembers[0][0].params['a_0']
rw.endmembers[1][0].params['a_0_orig'] = rw.endmembers[1][0].params['a_0']
gt.endmembers[0][0].params['a_0_orig'] = gt.endmembers[0][0].params['a_0']
gt.endmembers[1][0].params['a_0_orig'] = gt.endmembers[1][0].params['a_0']

args = get_params([0.]*len(F2003_mw_compositions))

args=[-601.746980562, -265.276034006, -2144.71808977, -1469.48233461, -2134.29281496, -1468.63466009, -5258.72190383, 26.9051660668, 58.5650007626, 94.1341411507, 151.436804912, 85.2595397066, 141.59078394, 81.6944694747, 136.858605106, 341.247642929, 3.0735457519, 3.38361213012, 2.84077558721, 2.78356225901, 2.13735721521, 1.86516990446, 2.21729429679, 1.94870855992, 11.1933607058, 5.36559846496, 16.7285246068, 7.63313673751, -2.19494923262, -0.0686185435259, -0.128658784141, -0.0448149995613, -0.43597561594, 0.0253782583923, 0.656901571957, -0.0650126447055, -0.278004147651, -1.03846776617, 0.447206414024, 0.202738370713, -0.0538262759527, 0.147687088716, -0.323765584945, 1.28000180067, 0.150132784573, -0.00441653723548, 0.804843464431, -0.585043751064, -0.298859309975, -1.12861762895, -0.415092369702, -0.134296076711]

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


# NOW PLOT THE FPER-OL-GT POLYMORPH EQUILIBRIA / PARTITIONING
viridis = cm.get_cmap('viridis', 101)
Tmin = 1273.1
Tmax = 1673.2
fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
for i, Pplot in enumerate(set(ol_gt_data.T[0])):
    for T in set(ol_gt_data.T[1]):
        for m in mins:
            m.set_state(Pplot, T)
        G = (py.gibbs/3. - alm.gibbs/3. - fo.gibbs/2. + fa.gibbs/2.)
        W_ol = (ol.solution_model.We[0][1] + ol.solution_model.Wv[0][1] * Pplot) /2. # 1 cation
        W_gt = (gt.solution_model.We[0][1] + gt.solution_model.Wv[0][1] * Pplot) /3.

    
        x_ols = np.linspace(0.00001, 0.99999, 101)
        x_gts = np.array([fsolve(affinity_ol_fper, [x_ol], args=(x_ol, G, T, W_ol, W_gt))[0]
                          for x_ol in x_ols])
        KDs = ((x_gts*(1. - x_ols))/
               (x_ols*(1. - x_gts)))
        ax[i].plot(1. - x_ols, np.log(KDs), color = viridis((T-Tmin)/(Tmax-Tmin)), linewidth=3., label='{0} K'.format(T))

    
    Ps, Ts, XMgOl, lnKD, lnKDerr = ol_gt_data.T
    mask = [idx for idx, P in enumerate(Ps) if np.abs(P - Pplot) < 10000.]
    ax[i].errorbar(XMgOl[mask], lnKD[mask], yerr=lnKDerr[mask], linestyle='None')
    ax[i].scatter(XMgOl[mask], lnKD[mask], c=Ts[mask], s=80., label='data', cmap=viridis, vmin=Tmin, vmax=Tmax)

    
                                    

fig = plt.figure(figsize=(30,10))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

# OLIVINE
ax[0].imshow(ol_fper_img, extent=[0.0, 0.8, -45000., -5000.], aspect='auto')


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
for i, run in enumerate(F2003_mw_compositions):
    P, T = F2003_mw_conditions[i]
    for j, chamber in enumerate(F2003_mw_compositions[i]):
        if 'ol' in F2003_mw_compositions[i][j]:
            pressures.append(F2003_mw_conditions[i][0])
            x_ols.append(F2003_mw_compositions[i][j]['ol'][0]/
                         (F2003_mw_compositions[i][j]['ol'][0] +
                          F2003_mw_compositions[i][j]['ol'][6]))
            x_fpers.append(F2003_mw_compositions[i][j]['mw'][0]/
                           (F2003_mw_compositions[i][j]['mw'][0] +
                            F2003_mw_compositions[i][j]['mw'][6]))
            RTlnKDs.append(burnman.constants.gas_constant *
                           T*np.log((x_ols[-1]*(1. - x_fpers[-1]))/
                                    (x_fpers[-1]*(1. - x_ols[-1]))))
            
ax[0].scatter(x_ols, RTlnKDs, c=pressures, s=80., label='data',
              cmap=viridis, vmin=Pmin, vmax=Pmax)



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
for i, run in enumerate(F2003_mw_compositions):
    P, T = F2003_mw_conditions[i]
    for j, chamber in enumerate(F2003_mw_compositions[i]):
        if 'wad' in F2003_mw_compositions[i][j]:
            pressures.append(F2003_mw_conditions[i][0])
            x_wads.append(F2003_mw_compositions[i][j]['wad'][0]/
                          (F2003_mw_compositions[i][j]['wad'][0] +
                           F2003_mw_compositions[i][j]['wad'][6]))
            x_fpers.append(F2003_mw_compositions[i][j]['mw'][0]/
                           (F2003_mw_compositions[i][j]['mw'][0] +
                            F2003_mw_compositions[i][j]['mw'][6]))
            RTlnKDs.append(burnman.constants.gas_constant*T*np.log((x_wads[-1]*(1. - x_fpers[-1]))/
                                                                   (x_fpers[-1]*(1. - x_wads[-1]))))
            
ax[1].scatter(x_wads, RTlnKDs, c=pressures, s=80., label='data',
              cmap=viridis, vmin=Pmin, vmax=Pmax)

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
for i, run in enumerate(F2003_mw_compositions):
    for j, chamber in enumerate(F2003_mw_compositions[i]):
        if 'ring' in F2003_mw_compositions[i][j]:
            pressures.append(F2003_mw_conditions[i][0])
            x_rws.append(F2003_mw_compositions[i][j]['ring'][0]/
                         (F2003_mw_compositions[i][j]['ring'][0] +
                          F2003_mw_compositions[i][j]['ring'][6]))
            x_fpers.append(F2003_mw_compositions[i][j]['mw'][0]/
                           (F2003_mw_compositions[i][j]['mw'][0] +
                            F2003_mw_compositions[i][j]['mw'][6]))
            
c = ax[2].scatter(x_rws, x_fpers, c=pressures, s=80., label='data',
                  cmap=viridis, vmin=Pmin, vmax=Pmax)

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
for i, run in enumerate(F2003_mw_compositions):
    P, T = F2003_mw_conditions[i]
    for j, chamber in enumerate(F2003_mw_compositions[i]):
        n_ol_polymorphs = 0
        for m in ['ol', 'wad', 'ring']:
            if m in F2003_mw_compositions[i][j]:
                n_ol_polymorphs += 1
                
        if n_ol_polymorphs > 1:
            for m in ['ol', 'wad', 'ring']:
                if m in F2003_mw_compositions[i][j]:
                    p_x_m[m].append([P,
                                     F2003_mw_compositions[i][j][m][0]/
                                     (F2003_mw_compositions[i][j][m][0] +
                                      F2003_mw_compositions[i][j][m][6])])

for m in ['ol', 'wad', 'ring']:
    pressures, xs = np.array(zip(*p_x_m[m]))
    plt.scatter(xs, pressures/1.e9, s=80., label='data')

plt.legend()
plt.show()
