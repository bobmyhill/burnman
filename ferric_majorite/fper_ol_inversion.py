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


run_inversion = raw_input("Do you want to invert the data now [y/n]?:\n(n plots the results of a previous inversion) ")


# A few images:
ol_polymorph_img = mpimg.imread('frost_2003_figures/ol_polymorphs.png')
ol_fper_img = mpimg.imread('frost_2003_figures/ol_fper_RTlnKD.png')
wad_fper_img = mpimg.imread('frost_2003_figures/wad_fper_RTlnKD.png')
rw_fper_img = mpimg.imread('frost_2003_figures/ring_fper_gt_KD.png')
rw_fper_part_img = mpimg.imread('frost_2003_figures/ring_fper_partitioning.png')


# First, let's load the Frost database:
with open('data/Frost_2003_chemical_analyses.dat', 'r') as f:
    ds = [line.split() for line in f if line.split() != [] and line[0] != '#']

all_runs = [d[0] for d in ds]
runs = list(set([d[0] for d in ds]))
conditions = [(float(ds[all_runs.index(run)][2])*1.e9,
               float(ds[all_runs.index(run)][3])) for run in runs]
chambers = [list(set([d[1] for d in ds if d[0] == run])) for run in runs]

compositions = []
for i, run in enumerate(runs):
    compositions.append([])
    run_indices = [idx for idx, x in enumerate(ds) if x[0] == run]
    for j, chamber in enumerate(chambers[i]):
        chamber_indices = [run_idx for run_idx in run_indices
                           if (ds[run_idx][1] == chamber and
                               ds[run_idx][4] != 'cen' and
                               ds[run_idx][4] != 'anB' and
                               ds[run_idx][4] != 'mag')]
        if len(chamber_indices) > 1:
            compositions[-1].append({ds[idx][4]: map(float, ds[idx][5:13])
                                     for idx in chamber_indices})
            #print(run, chamber, [ds[idx][4] for idx in chamber_indices])

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
wus.params['S_0'] = 58.
wus.params['V_0'] = 12.239e-06
wus.params['a_0'] = 3.22e-05
wus.params['K_0'] = 152.e9
wus.params['Kprime_0'] = 4.9
wus.params['Cp'] = np.array([42.638, 0.00897102, -260780.8, 196.6])

fwd.params['V_0'] = 4.31e-5 # original for HP is 4.321e-5
fwd.params['K_0'] = 160.e9 # 169.e9 is SLB
fwd.params['a_0'] = 2.48e-5 # 2.31e-5 is SLB

# Katsura et al., 2004
mrw.params['V_0'] = 3.949e-5
mrw.params['a_0'] = 2.2e-5
mrw.params['K_0'] = 182.e9

# Armentrout and Kavner, 2011
frw.params['V_0'] = 42.03e-6
frw.params['K_0'] = 202.e9
frw.params['Kprime_0'] = 4.
frw.params['a_0'] = 1.95e-5

for (P, m1, m2) in [[14.25e9, fo, mwd],
                    [20.e9, mwd, mrw],
                    [6.3e9, fa, frw],
                    [6.8e9, fa, fwd]]:
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
              energy_interaction=[[5.2e3]]) # O'Neill et al., 2003
wad = Solution(name = 'wadsleyite',
               solution_type ='symmetric',
               endmembers=[[mwd, '[Mg]2SiO4'], [fwd, '[Fe]2SiO4']],
               energy_interaction=[[15.e3]])
rw = Solution(name = 'ringwoodite',
              solution_type ='symmetric',
              endmembers=[[mrw, '[Mg]2SiO4'], [frw, '[Fe]2SiO4']],
              energy_interaction=[[8.2e3]]) 

solutions = {'mw': fper,
             'ol': ol,
             'wad': wad,
             'ring': rw}

# For each mineral pair, use the pressure and temperature to calculate the independent reaction affinities

# We want to minimize these by varying:
# - Endmember gibbs
# - Endmember thermal expansivities (essentially volumes at 1673 K)
# - Solution model (3)
# - Pressure error for each experiment (15)

# The pressure errors for each experiment have a Gaussian uncertainty of 0.5 GPa centered on the nominal pressure. 

def rms_misfit_affinities(args):

    ol.endmembers[0][0].params['H_0'] = args[0]*1.e3
    ol.endmembers[1][0].params['H_0'] = args[1]*1.e3
    wad.endmembers[0][0].params['H_0'] = args[2]*1.e3
    wad.endmembers[1][0].params['H_0'] = args[3]*1.e3
    rw.endmembers[0][0].params['H_0'] = args[4]*1.e3
    rw.endmembers[1][0].params['H_0'] = args[5]*1.e3
    
    fper.endmembers[0][0].params['a_0'] = args[6]*1.e-5
    fper.endmembers[1][0].params['a_0'] = args[7]*1.e-5
    ol.endmembers[0][0].params['a_0'] = args[8]*1.e-5
    ol.endmembers[1][0].params['a_0'] = args[9]*1.e-5
    wad.endmembers[0][0].params['a_0'] = args[10]*1.e-5
    wad.endmembers[1][0].params['a_0'] = args[11]*1.e-5
    rw.endmembers[0][0].params['a_0'] = args[12]*1.e-5
    rw.endmembers[1][0].params['a_0'] = args[13]*1.e-5

    
    fper.solution_model.We[0][1] = args[14]*1.e3
    wad.solution_model.We[0][1] = args[15]*1.e3
    rw.solution_model.We[0][1] = args[16]*1.e3
    
    Perr = np.array(args[17:])*1.e9 # arguments in GPa
    
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


    # apply strong priors to stable endmember transition pressures
    for (Porig, m1, m2) in [(14.25e9, ol.endmembers[0][0], wad.endmembers[0][0]),
                        (20.e9, wad.endmembers[0][0], rw.endmembers[0][0]),
                        (6.2e9, ol.endmembers[1][0], rw.endmembers[1][0])]:
        dP = fsolve(endmember_affinity, Porig, args=(T, m1, m2))[0] - Porig
        print('dP({0}, {1}): {2}'.format(m1.name, m2.name, dP/1.e9))
        misfit += (dP*dP)/0.1e9/0.1e9
    
    # we also need fwd to be unstable relative to frw
    P_fa_fwd = fsolve(endmember_affinity, 6.3e9, args=(T,
                                                      ol.endmembers[1][0],
                                                      wad.endmembers[1][0]))
    P_fa_frw = fsolve(endmember_affinity, 6.2e9, args=(T,
                                                      ol.endmembers[1][0],
                                                      rw.endmembers[1][0]))
    dP = P_fa_fwd - P_fa_frw
    if dP < 0.1e9:
        m = P_fa_fwd - P_fa_frw - 0.1e9
        misfit += (m*m)/0.1e9/0.1e9

    
    # include original alphas as priors
    well_constrained_indices = [0, 1, 2, 3] # per, wus, fo, fa
    less_constrained_indices = [4, 6, 7] # mwd, mrw, frw
    poorly_constrained_indices = [5] # fwd
    das = [fper.endmembers[0][0].params['a_0_orig'] - fper.endmembers[0][0].params['a_0'],
           fper.endmembers[1][0].params['a_0_orig'] - fper.endmembers[1][0].params['a_0'],
           ol.endmembers[0][0].params['a_0_orig'] - ol.endmembers[0][0].params['a_0'],
           ol.endmembers[1][0].params['a_0_orig'] - ol.endmembers[1][0].params['a_0'],
           wad.endmembers[0][0].params['a_0_orig'] - wad.endmembers[0][0].params['a_0'],
           wad.endmembers[1][0].params['a_0_orig'] - wad.endmembers[1][0].params['a_0'],
           rw.endmembers[0][0].params['a_0_orig'] - rw.endmembers[0][0].params['a_0'],
           rw.endmembers[1][0].params['a_0_orig'] - rw.endmembers[1][0].params['a_0']]
    for idx in well_constrained_indices:
        misfit += das[idx]*das[idx]/(3.e-7*3.e-7)
    for idx in less_constrained_indices:
        misfit += das[idx]*das[idx]/(6.e-7*6.e-7)
    for idx in poorly_constrained_indices:
        misfit += das[idx]*das[idx]/(10.e-7*10.e-7)
 
    
    misfit += np.sum(Perr*Perr)/(0.5e9*0.5e9)
    n += len(Perr)
    rms_misfit = np.sqrt(misfit)/float(n)
    print(rms_misfit)
    print(*args, sep = ", ")
    return rms_misfit

args = [ol.endmembers[0][0].params['H_0']*1.e-3,
        ol.endmembers[1][0].params['H_0']*1.e-3,
        wad.endmembers[0][0].params['H_0']*1.e-3,
        wad.endmembers[1][0].params['H_0']*1.e-3,
        rw.endmembers[0][0].params['H_0']*1.e-3,
        rw.endmembers[1][0].params['H_0']*1.e-3,
        fper.endmembers[0][0].params['a_0']*1.e5,
        fper.endmembers[1][0].params['a_0']*1.e5, 
        ol.endmembers[0][0].params['a_0']*1.e5,
        ol.endmembers[1][0].params['a_0']*1.e5, 
        wad.endmembers[0][0].params['a_0']*1.e5,
        wad.endmembers[1][0].params['a_0']*1.e5,
        rw.endmembers[0][0].params['a_0']*1.e5,
        rw.endmembers[1][0].params['a_0']*1.e5,
        fper.solution_model.We[0][1]*1.e-3,
        wad.solution_model.We[0][1]*1.e-3,
        rw.solution_model.We[0][1]*1.e-3]

args.extend([0.]*len(compositions))

fper.endmembers[0][0].params['a_0_orig'] = fper.endmembers[0][0].params['a_0']
fper.endmembers[1][0].params['a_0_orig'] = fper.endmembers[1][0].params['a_0']
ol.endmembers[0][0].params['a_0_orig'] = ol.endmembers[0][0].params['a_0']
ol.endmembers[1][0].params['a_0_orig'] = ol.endmembers[1][0].params['a_0']
wad.endmembers[0][0].params['a_0_orig'] = wad.endmembers[0][0].params['a_0']
wad.endmembers[1][0].params['a_0_orig'] = wad.endmembers[1][0].params['a_0']
rw.endmembers[0][0].params['a_0_orig'] = rw.endmembers[0][0].params['a_0']
rw.endmembers[1][0].params['a_0_orig'] = rw.endmembers[1][0].params['a_0']

# HERE ARE THE ARGUMENTS TAKEN FROM ONE ITERATION (NOT NECESSARILY CONVERGED)
args = [-2171.68926628, -1476.93427354, -2135.1866126, -1464.36985721, -2131.92327782, -1472.07798858, 3.00096351404, 3.3288933686, 2.97726179874, 2.59466331239, 2.30255271613, 2.18230856407, 2.52249277112, 2.1871625162, 11.4197490885, 15.4771147383, 7.97425321059, 0.00359895247643, -0.000781087881406, 0.0227282501241, -0.101107082049, 0.0204740691209, 0.00130320839318, -0.0200635829394, -0.217865531078, -0.664896983702, 0.435740067012, 0.323924067, -0.00726587433416, 0.489646105897, -0.274645680342, -0.82851903331, 0.19499058072, -0.00261065958816, 0.718866878964, -0.558988472635, -0.0692191746842, 0.522619719052, -0.828759778583, -0.0525072998252, 0.116585922889]

# 0.13837235
args = [-2172.34610531, -1477.73211874, -2134.34892171, -1465.0234708, -2131.6104473, -1471.12029336, 3.00624767712, 3.31288109174, 3.01526262103, 2.65091385796, 2.19548325357, 2.26596314784, 2.4536616818, 2.03541247468, 11.5194717698, 14.952230769, 8.77405321164, -0.158925791626, -0.0296728447761, -0.708495437561, -0.462134548126, 0.0612368339494, 0.17847474152, -0.0608736960728, -0.634134208195, -1.28082935237, 0.478804835282, 0.73627584009, -0.116033266802, -0.0196608577644, -0.2453002901, -1.22689170887, 0.202350973926, 0.0451222981939, 0.786357401416, -0.493058434364, -0.000993673193318, 2.07219196459, -2.03593900896, -0.128061722901, -0.216298709251]
 
# 0.13819087
args=[-2172.21223539, -1477.72452097, -2134.2341807, -1465.29608059, -2131.62715247, -1471.08200087, 3.01245452303, 3.30715621229, 3.0109207735, 2.65425579193, 2.18386033433, 2.33041032258, 2.45190017049, 2.02432920654, 11.4911896823, 14.2782037565, 8.69990330194, -0.20341258985, -0.0513066750318, -0.600080851911, -0.457218321663, 0.137997878128, 0.219686938956, -0.0713644048061, -0.652668366406, -1.27941046463, 0.42905411959, 0.686702729068, -0.154854734077, 0.00406750447899, -0.288107895233, -0.966887361548, 0.15406382392, -0.0158241532333, 0.743780621605, -0.533529483197, -0.0401675584242, 2.28971469789, -1.96563096836, -0.150833204653, -0.214300555933]

# 0.13818157
args=[-2172.24229763, -1477.73094097, -2134.21348695, -1465.31766259, -2131.63032415, -1471.04378335, 3.01193296779, 3.30738399758, 3.01271689759, 2.65405572914, 2.18066223274, 2.33720344564, 2.45049133556, 2.01972735396, 11.4891376195, 14.1993935189, 8.69849871036, -0.186502273085, -0.0440581798871, -0.64773423322, -0.470447001968, 0.157226141826, 0.2054996226, -0.0682073255836, -0.650536954345, -1.29282668751, 0.427370007216, 0.698874720275, -0.139509968718, -0.0123107992564, -0.288102521504, -1.03411671737, 0.153692111568, -0.0178630849702, 0.744145456324, -0.533673662192, -0.048844253772, 2.25905517232, -1.99209646683, -0.14634334767, -0.229038577315]

# 0.13818149
args=[-2172.24174745, -1477.73130752, -2134.21346864, -1465.31939364, -2131.62992484, -1471.04264588, 3.01200165238, 3.30732943344, 3.0127521696, 2.65419477917, 2.18044095698, 2.33747380641, 2.45021206894, 2.01927829666, 11.4897316424, 14.1940311961, 8.69867294921, -0.186422444988, -0.0438628576799, -0.646750372017, -0.470004056874, 0.158148675231, 0.205052961369, -0.0681584220526, -0.650116080449, -1.29253514405, 0.425710717915, 0.698762866569, -0.139297710744, -0.0127964487537, -0.289254807464, -1.03472470265, 0.152425119556, -0.0192100585134, 0.743293988333, -0.53512354493, -0.0487160442682, 2.25691971525, -1.99175907231, -0.146009953735, -0.229528314654]

# Try with a smaller interaction parameter for wadsleyite (better for KD with fper) ... this is in progress
#args= [-2172.34922804, -1477.74726094, -2134.34337761, -1464.99322165, -2131.61972781, -1471.12853973, 3.0461037191, 3.27607594536, 2.97600626585, 2.56960649568, 2.17684187064, 2.54686774039, 2.42431153731, 1.95566185472, 11.5082491237, 9.02339820944, 8.76547268054, -0.159039215714, -0.0297233984955, -0.708189288327, -0.462176307003, 0.0635574717662, 0.178261547124, -0.0609242009597, -0.634273914862, -1.28078173749, 0.46849007803, 0.736092054282, -0.116105363338, -0.015042697209, -0.235494122185, -1.22739703651, 0.192196743595, 0.0193257330532, 0.785738461228, -0.495400734321, -0.00159701161369, 2.0722115239, -2.03565008925, -0.128196669735, -0.210663818324]
# THIS LINE RUNS THE MINIMIZATION!!!
if run_inversion == 'y' or run_inversion == 'Y' or run_inversion == 'yes' or run_inversion == 'Yes':
    print(minimize(rms_misfit_affinities, args, method='BFGS')) # , options={'eps': 1.e-02}))



ol.endmembers[0][0].params['H_0'] = args[0]*1.e3
ol.endmembers[1][0].params['H_0'] = args[1]*1.e3
wad.endmembers[0][0].params['H_0'] = args[2]*1.e3
wad.endmembers[1][0].params['H_0'] = args[3]*1.e3
rw.endmembers[0][0].params['H_0'] = args[4]*1.e3
rw.endmembers[1][0].params['H_0'] = args[5]*1.e3

fper.endmembers[0][0].params['a_0'] = args[6]*1.e-5
fper.endmembers[1][0].params['a_0'] = args[7]*1.e-5
ol.endmembers[0][0].params['a_0'] = args[8]*1.e-5
ol.endmembers[1][0].params['a_0'] = args[9]*1.e-5
wad.endmembers[0][0].params['a_0'] = args[10]*1.e-5
wad.endmembers[1][0].params['a_0'] = args[11]*1.e-5
rw.endmembers[0][0].params['a_0'] = args[12]*1.e-5
rw.endmembers[1][0].params['a_0'] = args[13]*1.e-5


fper.solution_model.We[0][1] = args[14]*1.e3
wad.solution_model.We[0][1] = args[15]*1.e3
rw.solution_model.We[0][1] = args[16]*1.e3

Perr = np.array(args[17:])*1.e9 # arguments in GPa


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
    W_ol = ol.solution_model.We[0][1]/2. # 1 cation
    W_fper = fper.solution_model.We[0][1]
    
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
    W_wad = wad.solution_model.We[0][1]/2. # 1 cation
    W_fper = fper.solution_model.We[0][1]
    
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
    W_rw = rw.solution_model.We[0][1]/2. # 1 cation
    W_fper = fper.solution_model.We[0][1]
    
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
T0 = 1673.15
x_m1 = 0.30
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

plt.imshow(ol_polymorph_img, extent=[0., 1., 6., 20.], aspect='auto')

plt.plot([x_ol_inv, x_rw_inv], [P_inv/1.e9, P_inv/1.e9], linewidth=4., color='blue')

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

    plt.plot(x_m1s, pressures/1.e9, linewidth=4., color='blue')
    plt.plot(x_m2s, pressures/1.e9, linewidth=4., color='blue')
plt.show()
