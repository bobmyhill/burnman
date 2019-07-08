from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import minimize, fsolve, curve_fit
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.solutionbases import transform_solution_to_new_basis

#########################
# ENDMEMBER DEFINITIONS #
#########################

def create_minerals():
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

    # Bridgmanite
    mbdg = burnman.minerals.HHPH_2013.mpv()
    fbdg = burnman.minerals.HHPH_2013.fpv()
    abdg = burnman.minerals.HHPH_2013.apv()
    fefbdg = burnman.minerals.HP_2011_ds62.hem()

    fabdg = burnman.CombinedMineral([fefbdg, abdg], [0.5, 0.5], [-6.6e3, 0., 0.], name='fabdg')

    # SiO2 polymorphs
    qtz = burnman.minerals.HP_2011_ds62.q()
    coe = burnman.minerals.HP_2011_ds62.coe()
    stv = burnman.minerals.HP_2011_ds62.stv()




    ###############################
    # MODIFY ENDMEMBER PROPERTIES #
    ###############################

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


    mbdg.params['S_0_orig'] = [57.9, 0.3] # Akaogi et al. (2008), formal error is 0.3
    fbdg.params['S_0_orig'] = [fbdg.params['S_0'], 15.] # v. large uncertainties

    sp.params['S_0_orig'] = [80.9, 0.6] # Klemme and Ahrens, 2007; 10.1007/s00269-006-0128-4

    mins = [per, wus, fo, fa, mwd, fwd, mrw, frw, mbdg, fbdg]

    for m in mins:
        # Set entropies
        m.params['S_0'] = m.params['S_0_orig'][0]

        # Get thermal expansivities
        m.params['a_0_orig'] = m.params['a_0']




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


    endmembers = {'per':      per,      # MgO / FeO
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
                  'mbdg':     mbdg,
                  'fbdg':     fbdg,
                  'qtz':      qtz,      # SiO2 polymorphs
                  'coe':      coe,
                  'stv':      stv}

    solutions = {'mw': fper,
                 'ol': ol,
                 'wad': wad,
                 'sp': spinel,
                 'bdg': bdg}


    # Child solutions *must* be in dictionary to be reset properly
    child_solutions = {'mg_fe_bdg': transform_solution_to_new_basis(solutions['bdg'],
                                                                    np.array([[1., 0., 0., 0., 0.],
                                                                              [0., 1., 0., 0., 0.]]),
                                                                    solution_name='mg-fe bridgmanite'),
                       'ring': transform_solution_to_new_basis(solutions['sp'],
                                                               np.array([[0., 0., 0., 1., 0.],
                                                                         [0., 0., 0., 0., 1.]]),
                                                               solution_name='ringwoodite')}

    mineral_dataset = {'endmembers': endmembers,
                       'solutions': solutions,
                       'child_solutions': child_solutions}
    return mineral_dataset
