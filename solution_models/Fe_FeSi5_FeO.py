import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals

###############
print_latex = False
print_grid = False
melting_curves=True
mw_test = False
pv_test = False
###############


# Fe
from fcc_iron import fcc_iron
from hcp_iron import hcp_iron
from liq_iron_AA1994 import liq_iron

Fe_hcp = hcp_iron()
Fe_fcc = fcc_iron()
Fe_liq = liq_iron()

# FeO
from B1_wuestite import B1_wuestite
from liq_wuestite_AA1994 import liq_FeO

FeO_B1 = B1_wuestite()
FeO_liq = liq_FeO()

# Fe0.5Si0.5
from B20_FeSi import B20_FeSi
from B2_FeSi import B2_FeSi
from liq_Fe5Si5_AA1994 import liq_Fe5Si5

FeSi_B20 = B20_FeSi()
FeSi_B2 = B2_FeSi()
Fe5Si5_liq = liq_Fe5Si5()

from Fe_FeO import Fe_FeO_liquid_simple, Fe_FeO_liquid
Fe_FeO_liq = Fe_FeO_liquid()
Fe_FeO_liq_simple = Fe_FeO_liquid_simple()


class FeSiO_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe - Fe0.5Si0.5 - FeO solution'
        self.type='full_subregular'
        self.P_0=1.e5
        self.T_0=1650.
        self.endmembers = [[Fe_liq,     '[Fe]0.5[Fe]0.5'],
                           [Fe5Si5_liq, '[Fe]0.5[Si]0.5'],
                           [FeO_liq,    'Fe[O]0.5[O]0.5']]

        # Parameters for Fe-Fe0.5Si0.5
        E0=-40.e3
        V0=0.
        ksi0=7./3.
        zeta0=1.
        
        E_FeFeO = Fe_FeO_liq.energy_interaction[0][0][0]
        E_FeOFe = Fe_FeO_liq.energy_interaction[0][0][1]
        self.energy_interaction = [[[E0, E0], [E_FeFeO, E_FeOFe]],
                                   [[E_FeFeO, E_FeOFe]]]


        V_FeFeO = Fe_FeO_liq.volume_interaction[0][0][0]
        V_FeOFe = Fe_FeO_liq.volume_interaction[0][0][1]
        self.volume_interaction = [[[V0, V0], [V_FeFeO, V_FeOFe]],
                                   [[V_FeFeO, V_FeOFe]]]
 
        ksi_FeFeO = Fe_FeO_liq.kprime_interaction[0][0][0]
        ksi_FeOFe = Fe_FeO_liq.kprime_interaction[0][0][1]
        self.kprime_interaction = [[[ksi0, ksi0], [ksi_FeFeO, ksi_FeOFe]],
                                   [[ksi_FeFeO, ksi_FeOFe]]]

        zeta_FeFeO = Fe_FeO_liq.thermal_pressure_interaction[0][0][0]
        zeta_FeOFe = Fe_FeO_liq.thermal_pressure_interaction[0][0][1]
        self.thermal_pressure_interaction = [[[zeta0, zeta0], [zeta_FeFeO, zeta_FeOFe]],
                                             [[zeta_FeFeO, zeta_FeOFe]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

class FeSiO_liquid_Frost(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe - Fe0.5Si0.5 - FeO solution'
        self.type='subregular'
        self.endmembers = [[Fe_liq,     '[Fe]0.5[Fe]0.5'],
                           [Fe5Si5_liq, '[Fe]0.5[Si]0.5'],
                           [FeO_liq,    'Fe[O]0.5[O]0.5']]

        self.enthalpy_interaction = [[[-40.e3, -40.e3], [83307.,135943.]],
                                    [[83307.,135943.]]]
        
        self.entropy_interaction = [[[0., 0.], [8.978, 31.122]],
                                   [[8.978, 31.122]]]

        self.volume_interaction = [[[0., 0.], [-0.9e-6,-0.59e-6]],
                                   [[-0.9e-6,-0.59e-6]]]


        burnman.SolidSolution.__init__(self, molar_fractions)

class FeSiO_liquid_simple(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe - Fe0.5Si0.5 - FeO solution'
        self.type='simple_subregular'
        self.endmembers = [[Fe_liq,     '[Fe]0.5[Fe]0.5'],
                           [Fe5Si5_liq, '[Fe]0.5[Si]0.5'],
                           [FeO_liq,    'Fe[O]0.5[O]0.5']]

        E0 = Fe_FeO_liq_simple.energy_interaction[0][0][0]
        E1 = Fe_FeO_liq_simple.energy_interaction[0][0][1]
        self.energy_interaction = [[[-40.e3, -40.e3], [E0, E1]],
                                    [[E0, E1]]]
        
        S0 = Fe_FeO_liq_simple.entropy_interaction[0][0][0]
        S1 = Fe_FeO_liq_simple.entropy_interaction[0][0][1]
        self.entropy_interaction = [[[0., 0.], [S0, S1]],
                                   [[S0, S1]]]

        V0 = Fe_FeO_liq_simple.volume_interaction[0][0][0]
        V1 = Fe_FeO_liq_simple.volume_interaction[0][0][1]
        self.volume_interaction = [[[0., 0.], [V0, V1]],
                                   [[V0, V1]]]
        
        kxs = ksi_FeFeO = Fe_FeO_liq_simple.modulus_interaction[0][0][0]
        self.modulus_interaction = [[[kxs, kxs], [kxs, kxs]],
                                    [[kxs, kxs]]]


        burnman.SolidSolution.__init__(self, molar_fractions)

class ferropericlase(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ferropericlase'
        self.endmembers = [[burnman.minerals.SLB_2011.periclase(), '[Mg]O'],[FeO_B1, '[Fe]O']] # NOTE that SLB_2011_per has a different base energy...
        self.type='symmetric'
        self.enthalpy_interaction=[[13.e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)


en_HP = burnman.minerals.HP_2011_ds62.en()
en = burnman.minerals.SLB_2011.enstatite()
mpv = burnman.minerals.SLB_2011.mg_bridgmanite()

en_HP.set_state(1.e5, 300.)
en.set_state(1.e5, 300.)
mpv.set_state(1.e5, 300.)

SLBsubHP = en.gibbs - en_HP.gibbs
mpv.params['F_0'] = mpv.params['F_0'] - 0.5*(SLBsubHP)
print 0.5*(SLBsubHP)

fs_HP = burnman.minerals.HP_2011_ds62.fs()
fs = burnman.minerals.SLB_2011.ferrosilite()


fpv = burnman.minerals.SLB_2011.fe_bridgmanite()

fs_HP.set_state(1.e5, 300.)
fs.set_state(1.e5, 300.)
fpv.set_state(1.e5, 300.)

SLBsubHP = fs.gibbs - fs_HP.gibbs
fpv.params['F_0'] = fpv.params['F_0'] - 0.5*(SLBsubHP)
print 0.5*(SLBsubHP)


#liq = FeSiO_liquid_Frost()
#liq = FeSiO_liquid()
liq = FeSiO_liquid_simple()
fper = ferropericlase()



#mpv = burnman.minerals.HHPH_2013.mpv()
#fpv = burnman.minerals.HHPH_2013.fpv()


        
class mg_fe_bridgmanite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ferropericlase'
        self.endmembers = [[mpv, '[Mg]SiO3'],[fpv, '[Fe]SiO3']]
        self.type='symmetric'
        self.enthalpy_interaction=[[0.e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)
bdg = mg_fe_bridgmanite()

if print_latex==True:
    import json
    print '\subsection{Solid endmember parameters}'
    
    minerals=[Fe_hcp, Fe_fcc, FeO_B1, FeSi_B20, FeSi_B2]
    for mineral in minerals:
        print '\subsubsection{'+mineral.name+'}'
        print '\\begin{lstlisting}'
        print json.dumps(mineral.params, indent=2)
        print '\end{lstlisting}'
        print ''
        
    print ''
    print '\subsection{Liquid endmember parameters}'
        
    minerals=[Fe_liq, FeO_liq, Fe5Si5_liq]
    for mineral in minerals:
        print '\subsubsection{'+mineral.name+'}'
        print '\\begin{lstlisting}'
        print json.dumps(mineral.params, indent=2)
        print '\end{lstlisting}'
        print ''
    
    print ''
    print '\subsection{Solution model parameters}'
    print '\\begin{lstlisting}'
    print 'energy_interaction =', liq.energy_interaction
    print 'entropy_interaction =', liq.entropy_interaction
    print 'volume_interaction =', liq.volume_interaction
    print 'modulus_interaction =', liq.modulus_interaction
    print '\end{lstlisting}'


if print_grid==True:
    temperatures = np.linspace(1000., 4000., 11)
    Wsij = np.empty_like(temperatures)
    Wsji = np.empty_like(temperatures)
    for P in [1.e5, 100.e9, 200.e9]:
        for i, T in enumerate(temperatures):
            liq.set_composition([0.5, 0.4, 0.1]) # not important
            liq.set_state(P, T)
            Wsij[i] = liq.solution_model.Ws[0][2]
            Wsji[i] = liq.solution_model.Ws[2][0]
        plt.plot(temperatures, Wsij, linestyle='--', label=str(P/1.e9)+' GPa')
        plt.plot(temperatures, Wsji, linestyle='--', label=str(P/1.e9)+' GPa')
    plt.legend(loc='lower left')
    plt.show()
            
if melting_curves==True:
    # Fe
    Fe_bcc = minerals.HP_2011_ds62.iron()
    
    # Metastable Fe_fcc <-> Fe_hcp at 1 bar is ca. 500 K
    # Fe_bcc <-> Fe_hcp at 300 K is at ca. 12 GPa
    print burnman.tools.equilibrium_temperature([Fe_fcc, Fe_hcp], [1.0, -1.0], 1.e5)
    print burnman.tools.equilibrium_pressure([Fe_bcc, Fe_hcp], [1.0, -1.0], 298.15)/1.e9




    fig1 = mpimg.imread('data/Anzellini_2013_Fe_melting.png')  # Uncomment these two lines if you want to overlay the plot on a screengrab from SLB2011
    plt.imshow(fig1, extent=[0., 230., 1200., 5200.], aspect='auto')
    
    
    
    # Find triple points
    P_inv, T_inv = burnman.tools.invariant_point([Fe_fcc, Fe_hcp], [1.0, -1.0],\
                                                 [Fe_bcc, Fe_hcp], [1.0, -1.0],\
                                                 [10.e9, 800.])
    P_inv2, T_inv2 = burnman.tools.invariant_point([Fe_fcc, Fe_hcp], [1.0, -1.0],\
                                                   [Fe_liq, Fe_hcp], [1.0, -1.0],\
                                                   [100.e9, 3000.])

    
    temperatures = np.linspace(298.15, T_inv, 11)
    pressures = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = burnman.tools.equilibrium_pressure([Fe_bcc, Fe_hcp], [1.0, -1.0], T, 10.e9)
    plt.plot(pressures/1.e9, temperatures)
    
    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/bcc_hcp_equilibrium.dat')

    pressures = np.linspace(1.e5, P_inv, 11)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([Fe_bcc, Fe_fcc], [1.0, -1.0], P, 1000.)
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/bcc_fcc_equilibrium.dat')

    temperatures = np.linspace(T_inv, T_inv2, 11)
    pressures = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = burnman.tools.equilibrium_pressure([Fe_fcc, Fe_hcp], [1.0, -1.0], T, 100.e9)
    
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/fcc_hcp_equilibrium.dat')
    
    pressures = np.linspace(5.2e9, P_inv2, 11)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([Fe_fcc, Fe_liq], [1.0, -1.0], P, 1000.)
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/fcc_liq_equilibrium.dat')

    pressures = np.linspace(P_inv2, 350.e9, 11)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([Fe_hcp, Fe_liq], [1.0, -1.0], P, 3000.) 
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/hcp_liq_equilibrium.dat')

    plt.ylim(300., 7000.)
    plt.xlim(0., 350.)
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Temperature (K)")
    plt.show()

    # FeO
    pressures = np.linspace(1.e5, 250.e9, 101)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([FeO_B1, FeO_liq], [1.0, -1.0], P, 3000.) 
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/FeO_B1_liq_equilibrium.dat')
    plt.ylim(300., 7000.)
    plt.xlim(0., 350.)
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Temperature (K)")
    plt.show()

    # FeSi
    # Find triple point
    P_inv, T_inv = burnman.tools.invariant_point([FeSi_B20, FeSi_B2], [1.0, -1.0],\
                                                 [FeSi_B20, Fe5Si5_liq], [1.0, -2.0],\
                                                 [30.e9, 2500.])
    pressures = np.linspace(1.e5, P_inv, 31)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([FeSi_B20, Fe5Si5_liq],
                                                                [1.0, -2.0], P, 3000.) 
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/FeSi_B20_liq_equilibrium.dat')

    pressures = np.linspace(P_inv, 150.e9, 101)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([FeSi_B2, Fe5Si5_liq],
                                                                [1.0, -2.0], P, 3000.) 
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/FeSi_B2_liq_equilibrium.dat')


    temperatures = np.linspace(300., T_inv, 21)
    pressures = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        pressures[i] = burnman.tools.equilibrium_pressure([FeSi_B2, FeSi_B20], [1.0, -1.0], T, 30.e9)
    
    plt.plot(pressures/1.e9, temperatures)

    np.savetxt(header='Pressures (GPa) Temperatures (K)', X=zip(*[pressures/1.e9, temperatures]), fname='output_data/FeSi_B20_B2_equilibrium.dat')

    plt.show()
    
    
    
if mw_test==True:
    Ozawa_data = np.loadtxt(fname='data/Ozawa_et_al_2008_fper_iron.dat', unpack=True)
    expt, P, T, t, O_wtmelt, O_wtmelt_err, O_molmelt, O_molmelt_err, XFeO, XFeO_err = Ozawa_data
    P = P*1.e9
    Terr = T/10.
    XFeO_melt = O_molmelt / (100.-O_molmelt)
    
    '''
    plt.errorbar(O_molmelt, XFeO, xerr = O_molmelt_err, yerr = XFeO_err, linestyle='none')
    plt.show()
    '''
    
    def fper_melt_eqm(XFeO_melt, P, T, XFeO_per):
        fper.set_composition([1.-XFeO_per, XFeO_per])
        fper.set_state(P, T)
    
        liq.set_composition([1. - XFeO_melt[0], 0., XFeO_melt[0]])
        liq.set_state(P, T)
    
        return fper.partial_gibbs[1] - liq.partial_gibbs[2]
    
    '''
    for datum in Ozawa_data:
    P, T, Terr, O_melt, O_melt_err, XFeO, XFeO_err = datum
    XFeO_melt = O_melt / (100.-O_melt)
    print P/1.e9, T, XFeO, XFeO_melt, fsolve(fper_melt_eqm, [XFeO_melt/10.], args=(P, T, XFeO))[0]
    '''
    
    XFeO_mw = 0.2
    temperatures = np.linspace(3000., 4500., 4)
    pressures = np.linspace(15.e9, 140.e9, 15)
    for T in temperatures:
        lnKd = np.empty_like(pressures)
        guess=0.01
        for i, P in enumerate(pressures):
            XFeO_melt = fsolve(fper_melt_eqm, [guess], args=(P, T, XFeO_mw))[0]
            guess = XFeO_melt # for next step
            lnKd[i] = np.log(XFeO_melt/XFeO_mw)
            print P/1.e9, T, lnKd[i]
        plt.plot(pressures, lnKd, label=str(T)+'K')

        np.savetxt(fname='output_data/lnD_melt_mw_'+str(T)+'_K.dat',
                   X=zip(*[pressures/1.e9, lnKd]),
                   header='Pressure (GPa) lnD (XFeO_melt/XFeO_mw)')
    
    plt.title('FeO partitioning between periclase and metallic melt')
    plt.xlabel('Pressure (GPa)')
    plt.ylabel('ln KD')
    plt.legend(loc='upper right')
    plt.show()

if pv_test==True:

    Ozawa_data = np.loadtxt(fname='data/Ozawa_et_al_2009_pv_melt_compositions.dat', unpack=True)
    expt, c0, c1, P, T, t, O_wtmelt, O_wtmelt_err, Si_wtmelt, Si_wtmelt_err, O_molmelt, O_molmelt_err, Si_molmelt, Si_molmelt_err, XFeSiO3, XFeSiO3_err = Ozawa_data
    P = P*1.e9

    XFeO_melt = O_molmelt
    XFe5Si5_melt = 2.*Si_molmelt
    XFe_melt = 1. - O_molmelt - Si_molmelt

    total = XFeO_melt + XFe5Si5_melt + XFe_melt

    XFeO_melt = XFeO_melt/total
    XFe5Si5_melt = XFe5Si5_melt/total
    XFe_melt = XFe_melt/total
    


    def bdg_melt_eqm(XFeSiO3_bdg, P, T, XFeO_melt, XFe5Si5_melt):
        bdg.set_composition([1.-XFeSiO3_bdg, XFeSiO3_bdg])
        bdg.set_state(P, T)
    
        liq.set_composition([1. - XFeO_melt[0] - XFe5Si5_melt, XFe5Si5_melt, XFeO_melt[0]])
        liq.set_state(P, T)
        
        mu_FeSiO3 = burnman.chemicalpotentials.chemical_potentials([liq], [bdg.endmembers[1][0].params['formula']])[0]
        return bdg.partial_gibbs[1] - mu_FeSiO3
        
    

    def melt_bdg_eqm(XFeO_melt, P, T, XFeSiO3_bdg, XFe5Si5_melt):
        bdg.set_composition([1.-XFeSiO3_bdg, XFeSiO3_bdg])
        bdg.set_state(P, T)
    
        liq.set_composition([1. - XFeO_melt[0] - XFe5Si5_melt, XFe5Si5_melt, XFeO_melt[0]])
        liq.set_state(P, T)
        
        mu_FeSiO3 = burnman.chemicalpotentials.chemical_potentials([liq], [bdg.endmembers[1][0].params['formula']])[0]
        return bdg.partial_gibbs[1] - mu_FeSiO3
    

    pressures = [25.e9, 100.e9]
    temperatures = np.linspace(2773., 4273., 4)
    XFeSiO3_bdg = 0.1

    invXs = np.linspace(2., 500., 11)
    Xs = np.linspace(0.002, 0.5, 11)
    
    Xs_Fe5Si5_melt = np.unique(np.sort(np.concatenate((1./invXs, Xs))))

    
    
    for P in pressures:
        for T in temperatures:
            X_Sis_wt = np.empty_like(Xs_Fe5Si5_melt)
            X_Os_wt = np.empty_like(Xs_Fe5Si5_melt)
            guess = 0.01
            for i, XFe5Si5_melt in enumerate(Xs_Fe5Si5_melt):
                XFeO_melt = fsolve(melt_bdg_eqm, [guess], args=(P, T, XFeSiO3_bdg, XFe5Si5_melt))[0]

                guess = XFeO_melt
                # Convert molar melt compositions (components) into wt % (elemental)
                mol_O = XFeO_melt
                mol_Si = 0.5 * XFe5Si5_melt
                mol_Fe = (1. - XFeO_melt - XFe5Si5_melt) + XFeO_melt + 0.5 * XFe5Si5_melt
    
                wt_O = mol_O*15.9994
                wt_Si = mol_Si*28.0855
                wt_Fe = mol_Fe*55.845
    
                wt_total = (wt_O + wt_Si + wt_Fe)/100.
                
                X_Sis_wt[i] = wt_Si/wt_total
                X_Os_wt[i] = wt_O/wt_total
                
                print P/1.e9, T, X_Sis_wt[i], X_Os_wt[i]
        
            plt.plot(X_Os_wt, X_Sis_wt, label=str(P/1.e9)+' GPa, '+str(T)+'K')
            
            np.savetxt(fname='output_data/metal_Mg'+str(1.-XFeSiO3_bdg)+'Fe'+str(XFeSiO3_bdg)+'SiO3_equilibrium_'+str(P/1.e9)+'_GPa_'+str(T)+'_K.dat',
                       X=zip(*[X_Os_wt, X_Sis_wt]),
                       header='O (wt %) Si (wt %)')
        
    plt.title('Metallic melt in equilibrium with '+\
              'Mg'+str(1.-XFeSiO3_bdg)+\
              'Fe'+str(XFeSiO3_bdg)+\
              'SiO3')
    plt.xlabel('XO (wt %)')
    plt.ylabel('XSi (wt %)')
    plt.xlim(0.0, 10.0)
    plt.ylim(0.0, 10.0)
    plt.legend(loc='upper right')
    plt.show()
    
