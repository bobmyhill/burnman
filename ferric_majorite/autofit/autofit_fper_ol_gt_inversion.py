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
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit



from input_dataset import * 



if len(sys.argv) == 2:
    if sys.argv[1] == '--fit':
        run_inversion = True
        print('Running inversion')
    else:
        run_inversion = False
        print('Not running inversion. Use --fit as command line argument to invert parameters')
else:
    run_inversion = False
    print('Not running inversion. Use --fit as command line argument to invert parameters')


def minimize_func(params, assemblages):
    # Set parameters 
    set_params(params)

    chisqr = []
    # Run through all assemblages for affinity misfit
    for i, assemblage in enumerate(assemblages):
        #print(i, assemblage.experiment_id, [phase.name for phase in assemblage.phases])
        # Assign compositions and uncertainties to solid solutions
        for j, phase in enumerate(assemblage.phases):
            if isinstance(phase, burnman.SolidSolution):
                molar_fractions, phase.molar_fraction_covariances = assemblage.stored_compositions[j]
                phase.set_composition(molar_fractions)
                
        # Assign a state to the assemblage
        P, T = np.array(assemblage.nominal_state)
        try:
             P += dict_experiment_uncertainties[assemblage.experiment_id]['P']
             T += dict_experiment_uncertainties[assemblage.experiment_id]['T']
        except:
            pass
        
        assemblage.set_state(P, T)
        
        
        # Calculate the misfit and store it
        assemblage.chisqr = assemblage_affinity_misfit(assemblage)
        #print('assemblage chisqr', assemblage.experiment_id, [phase.name for phase in assemblage.phases], assemblage.chisqr)
        chisqr.append(assemblage.chisqr)

    # Endmember priors
    for p in endmember_priors:
        c = np.power(((dict_endmember_args[p[0]][p[1]] - p[2])/p[3]), 2.)
        #print('endmember_prior', p[0], p[1], dict_endmember_args[p[0]][p[1]], p[2], p[3], c)
        chisqr.append(c)
        
    # Solution priors
    for p in solution_priors:
        c = np.power(((dict_solution_args[p[0]]['{0}{1}{2}'.format(p[1], p[2], p[3])] -
                       p[4])/p[5]), 2.)
        #print('solution_prior', c)
        chisqr.append(c)

    # Experiment uncertainties
    for u in experiment_uncertainties:
        c = np.power(u[2]/u[3], 2.)
        #print('pressure uncertainty', c)
        chisqr.append(c)

    rms_misfit = np.sqrt(np.sum(chisqr)/float(len(chisqr)))
    print(rms_misfit)
    string = '['
    for i, p in enumerate(params):
        string += '{0:.4f}'.format(p)
        if i < len(params) - 1:
            string += ', '
    string+=']'
    print(string)
    return rms_misfit


endmember_args = [['per', 'H_0', per.params['H_0'], 1.e3],
                  ['wus', 'H_0', wus.params['H_0'], 1.e3],
                  ['mwd', 'H_0', mwd.params['H_0'], 1.e3],
                  ['fwd', 'H_0', fwd.params['H_0'], 1.e3],
                  ['mrw', 'H_0', mrw.params['H_0'], 1.e3],
                  ['frw', 'H_0', frw.params['H_0'], 1.e3],
                  ['alm', 'H_0', alm.params['H_0'], 1.e3],
                  ['per', 'S_0', per.params['S_0'], 1.],
                  ['wus', 'S_0', wus.params['S_0'], 1.],
                  ['fo',  'S_0', fo.params['S_0'],  1.],
                  ['fa',  'S_0', fa.params['S_0'],  1.],
                  ['mwd', 'S_0', mwd.params['S_0'], 1.],
                  ['fwd', 'S_0', fwd.params['S_0'], 1.],
                  ['mrw', 'S_0', mrw.params['S_0'], 1.],
                  ['frw', 'S_0', frw.params['S_0'], 1.],
                  ['alm', 'S_0', alm.params['S_0'], 1.],
                  ['fwd', 'V_0', fwd.params['V_0'], 1.e-5],
                  ['per', 'a_0', per.params['a_0'], 1.e-5],
                  ['wus', 'a_0', wus.params['a_0'], 1.e-5],
                  ['fo',  'a_0', fo.params['a_0'],  1.e-5],
                  ['fa',  'a_0', fa.params['a_0'],  1.e-5],
                  ['mwd', 'a_0', mwd.params['a_0'], 1.e-5],
                  ['fwd', 'a_0', fwd.params['a_0'], 1.e-5],
                  ['mrw', 'a_0', mrw.params['a_0'], 1.e-5],
                  ['frw', 'a_0', frw.params['a_0'], 1.e-5]]

solution_args = [['mw', 'E', 0, 0, fper.energy_interaction[0][0], 1.e3],
                 ['ol', 'E', 0, 0, ol.energy_interaction[0][0], 1.e3],
                 ['wad', 'E', 0, 0, wad.energy_interaction[0][0], 1.e3],
                 ['ring', 'E', 0, 0, rw.energy_interaction[0][0], 1.e3],
                 ['gt', 'E', 0, 0, gt.energy_interaction[0][0], 1.e3]]

endmember_priors = [['per', 'a_0', per.params['a_0_orig'], 2.e-7],
                    ['fo', 'a_0', fo.params['a_0_orig'], 2.e-7],
                    ['fa', 'a_0', fa.params['a_0_orig'], 2.e-7],
                    ['mrw', 'a_0', mrw.params['a_0_orig'], 2.e-7],
                    ['wus', 'a_0', wus.params['a_0_orig'], 5.e-7],
                    ['mwd', 'a_0', mwd.params['a_0_orig'], 5.e-7],
                    ['frw', 'a_0', frw.params['a_0_orig'], 5.e-7],
                    ['fwd', 'a_0', fwd.params['a_0_orig'], 20.e-7],
                    ['fwd', 'V_0', 4.31e-5, 2.15e-7], # 0.5% uncertainty, somewhat arbitrary
                    ['per', 'S_0', per.params['S_0_orig'][0], per.params['S_0_orig'][1]],
                    ['wus', 'S_0', wus.params['S_0_orig'][0], wus.params['S_0_orig'][1]],
                    ['fo',  'S_0', fo.params['S_0_orig'][0],  fo.params['S_0_orig'][1]],
                    ['fa',  'S_0', fa.params['S_0_orig'][0],  fa.params['S_0_orig'][1]],
                    ['mwd', 'S_0', mwd.params['S_0_orig'][0], mwd.params['S_0_orig'][1]],
                    ['fwd', 'S_0', fwd.params['S_0_orig'][0], fwd.params['S_0_orig'][1]],
                    ['mrw', 'S_0', mrw.params['S_0_orig'][0], mrw.params['S_0_orig'][1]],
                    ['frw', 'S_0', frw.params['S_0_orig'][0], frw.params['S_0_orig'][1]],
                    ['alm', 'S_0', alm.params['S_0_orig'][0], alm.params['S_0_orig'][1]]]

solution_priors = [['ol', 'E', 0, 0, 5.2e3, 1.e3]]

experiment_uncertainties = [['49Fe', 'P', 0., 0.5e9],
                            ['50Fe', 'P', 0., 0.5e9],
                            ['61Fe', 'P', 0., 0.5e9],
                            ['62Fe', 'P', 0., 0.5e9],
                            ['63Fe', 'P', 0., 0.5e9],
                            ['64Fe', 'P', 0., 0.5e9],
                            ['66Fe', 'P', 0., 0.5e9],
                            ['67Fe', 'P', 0., 0.5e9],
                            ['68Fe', 'P', 0., 0.5e9],
                            ['V189', 'P', 0., 0.5e9],
                            ['V191', 'P', 0., 0.5e9],
                            ['V192', 'P', 0., 0.5e9],
                            ['V200', 'P', 0., 0.5e9],
                            ['V208', 'P', 0., 0.5e9],
                            ['V209', 'P', 0., 0.5e9],
                            ['V212', 'P', 0., 0.5e9],
                            ['V217', 'P', 0., 0.5e9],
                            ['V220', 'P', 0., 0.5e9],
                            ['V223', 'P', 0., 0.5e9],
                            ['V227', 'P', 0., 0.5e9],
                            ['V229', 'P', 0., 0.5e9],
                            ['V252', 'P', 0., 0.5e9],
                            ['V254', 'P', 0., 0.5e9]]  


# Make dictionaries
dict_endmember_args = {a[0]: {} for a in endmember_args}
for a in endmember_args:
    dict_endmember_args[a[0]][a[1]] = a[2]

dict_solution_args = {a[0]: {} for a in solution_args}
for a in solution_args:
    dict_solution_args[a[0]]['{0}{1}{2}'.format(a[1], a[2], a[3])] = a[4]

dict_experiment_uncertainties = {u[0] : {'P': 0., 'T': 0.} for u in experiment_uncertainties}
for u in experiment_uncertainties:
    dict_experiment_uncertainties[u[0]][u[1]] = u[2]


def get_params():
    """
    This function gets the parameters from the parameter lists
    """
    
    # Endmember parameters
    args = [a[2]/a[3] for a in endmember_args]

    # Solution parameters
    args.extend([a[4]/a[5] for a in solution_args])

    # Experimental uncertainties
    args.extend([u[2]/u[3] for u in experiment_uncertainties])
    return args

def set_params(args):
    """
    This function sets the parameters *both* in the parameter lists, 
    the parameter dictionaries and also in the minerals / solutions.
    """
    
    i=0

    # Endmember parameters
    for j, a in enumerate(endmember_args):
        dict_endmember_args[a[0]][a[1]] = args[i]*a[3]
        endmember_args[j][2] = args[i]*a[3]
        endmembers[a[0]].params[a[1]] = args[i]*a[3]
        i+=1

    # Solution parameters
    for j, a in enumerate(solution_args):
        dict_solution_args[a[0]][a[1]] = args[i]*a[5]
        solution_args[j][4] = args[i]*a[5]
        if a[1] == 'E':
            solutions[a[0]].energy_interaction[int(a[2])][int(a[3])] = args[i]*a[5]
        else:
            raise Exception('Not implemented')
        i+=1

    # Reinitialize solutions
    for name in solutions:
        burnman.SolidSolution.__init__(solutions[name])

    # Experimental uncertainties
    for j, u in enumerate(experiment_uncertainties):
        dict_experiment_uncertainties[u[0]][u[1]] = args[i]*u[3]
        experiment_uncertainties[j][2] = args[i]*u[3]
        i+=1

    return None

set_params([-601.3453, -265.6778, -2145.2891, -1465.4495, -2135.2160, -1469.7595, -5258.7219, 26.8835, 57.8398, 94.1394, 151.4104, 85.0606, 143.4888, 80.8637, 135.6502, 341.5676, 4.3099, 3.0837, 3.3717, 2.8343, 2.7844, 2.1969, 1.7645, 2.2154, 1.9452, 11.3566, 5.9386, 18.5443, 7.9449, -2.1949, -0.3940, -0.0641, -2.9597, -0.0518, 1.0549, -3.8583, -0.0432, -0.2590, -0.6305, 0.1343, 0.1782, 1.5145, 1.8494, 2.0586, -0.6711, -1.0292, -1.0906, -0.2548, 0.2573, 0.7087, 0.4244, -0.1634, 1.2794])
set_params([-600.8729, -266.1502, -2145.5672, -1464.9199, -2135.0357, -1469.8020, -5258.7219, 26.9476, 57.6979, 94.1621, 151.4353, 85.0096, 143.3162, 81.5983, 137.0470, 341.6043, 4.3102, 3.0842, 3.3528, 2.8185, 2.7849, 2.1472, 1.4385, 2.2359, 1.9630, 11.3218, 4.1223, 18.3509, 8.4515, -2.1949, -0.6021, -0.0862, -2.8626, -0.0920, 1.0177, -3.7252, -0.2226, -0.2805, -0.7647, 0.1421, 0.1777, 1.4966, 1.3428, 1.6705, -1.0299, -1.0966, -0.9390, -0.6202, -0.1556, 0.3566, 0.0831, -0.1665, 0.9527])
#set_params([-601.746980562, -265.276034006, -2144.71808977, -1469.48233461, -2134.29281496, -1468.63466009, -5258.72190383, 26.9051660668, 58.5650007626, 94.1341411507, 151.436804912, 85.2595397066, 141.59078394, 81.6944694747, 136.858605106, 341.247642929, 3.0735457519, 3.38361213012, 2.84077558721, 2.78356225901, 2.13735721521, 1.86516990446, 2.21729429679, 1.94870855992, 11.1933607058, 5.36559846496, 16.7285246068, 7.63313673751, -2.19494923262, -0.0686185435259, -0.128658784141, -0.0448149995613, -0.43597561594, 0.0253782583923, 0.656901571957, -0.0650126447055, -0.278004147651, -1.03846776617, 0.447206414024, 0.202738370713, -0.0538262759527, 0.147687088716, -0.323765584945, 1.28000180067, 0.150132784573, -0.00441653723548, 0.804843464431, -0.585043751064, -0.298859309975, -1.12861762895, -0.415092369702, -0.134296076711])

from Frost_2003_fper_ol_wad_rw import Frost_2003_assemblages
from endmember_reactions import endmember_reaction_assemblages
from destabilise_endmember_reactions import destabilised_endmember_reaction_assemblages
assemblages = Frost_2003_assemblages
assemblages.extend(endmember_reaction_assemblages)
assemblages.extend(destabilised_endmember_reaction_assemblages)
minimize_func(get_params(), assemblages)


# THIS LINE RUNS THE MINIMIZATION!!!
if run_inversion:
    print(minimize(minimize_func, get_params(), args=(assemblages), method='BFGS')) # , options={'eps': 1.e-02}))

# Print the current parameters
print(get_params())


###################
# A few images:
###################

ol_polymorph_img = mpimg.imread('frost_2003_figures/ol_polymorphs.png')
ol_polymorph_img_1200C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_1200C.png')
ol_polymorph_img_1000C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_1000C.png')
ol_polymorph_img_800C = mpimg.imread('frost_2003_figures/Akimoto_1987_fo_fa_phase_diagram_800C.png')

ol_fper_img = mpimg.imread('frost_2003_figures/ol_fper_RTlnKD.png')
wad_fper_img = mpimg.imread('frost_2003_figures/wad_fper_RTlnKD.png')
rw_fper_img = mpimg.imread('frost_2003_figures/ring_fper_gt_KD.png')
rw_fper_part_img = mpimg.imread('frost_2003_figures/ring_fper_partitioning.png')


fo_phase_diagram = mpimg.imread('frost_2003_figures/Mg2SiO4_phase_diagram_Jacobs_2017.png')
fa_phase_diagram = mpimg.imread('frost_2003_figures/Fe2SiO4_phase_diagram_Yagi_1987.png')
fa_phase_diagram2 = mpimg.imread('frost_2003_figures/Fe2SiO4_phase_diagram_Jacobs_2001.png')

###################################################
# FUNCTION

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

###################################################
# PLOTS

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
ax[1].plot(temperatures, eqm_pressures(fa, frw, temperatures)/1.e9, linewidth=4., label='fa-frw')
ax[1].plot(temperatures, eqm_pressures(fa, fwd, temperatures)/1.e9, linestyle=':', linewidth=4., label='fa-fwd (should be metastable)')


ax[0].set_xlabel('T (K)')
ax[0].set_ylabel('P (GPa)')
ax[1].set_xlabel('P (GPa)')
ax[1].set_ylabel('T (K)')
ax[1].legend()
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
        try:
            ax[j].plot(pressures/1.e9, Vs[j]*1.e6, label=ds[j].name+' '+ds[-1],
                       linewidth=linewidth)
        except:
            ax[j].plot(pressures/1.e9, Vs[j]*1.e6, label=ds[-1],
                       linewidth=linewidth)

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
ol_gt_data = np.loadtxt('data/ONeill_Wood_1979_ol_gt_KD.dat')
ol_gt_data[:,0] *= 1.e9 # P (GPa) to P (Pa)

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

plt.show()
                                    

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


"""
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
"""


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

"""
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
"""
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

"""
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
"""
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


P_Xmg_phase = {'ol': [], 'wad': [], 'ring': []}
for assemblage in Frost_2003_assemblages:
    if len(assemblage.phases) > 2:
        for i, phase in enumerate(assemblage.phases):
            for m in ['ol', 'wad', 'ring']:
                if phase == solutions[m]:
                    P_Xmg_phase[m].append([assemblage.nominal_state[0],
                                           assemblage.stored_compositions[i][0][1]])

for m in ['ol', 'wad', 'ring']:
    pressures, xs = np.array(zip(*P_Xmg_phase[m]))
    plt.scatter(xs, pressures/1.e9, s=80., label='data')

plt.legend()
plt.show()
