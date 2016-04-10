import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals

###############
print_grid = False
mw_test = True
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

class FeSiO_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe - Fe0.5Si0.5 - FeO solution'
        self.type='full_subregular'
        self.P_0=1.e5
        self.T_0=1650.
        self.endmembers = [[Fe_liq,     '[Fe]0.5[Fe]0.5'],
                           [Fe5Si5_liq, '[Fe]0.5[Si]0.5'],
                           [FeO_liq,    'Fe[O]0.5[O]0.5']]

        self.energy_interaction = [[[0.e3, 0.e3], [105.e3,118.e3]],
                                   [[105.e3,118.e3]]]
        
        self.volume_interaction = [[[0., 0.], [-1.4e-6,-1.2e-6]],
                                   [[-1.4e-6,-1.2e-6]]]
        ksi0=7./3.
        ksi1=7./3.
        self.kprime_interaction = [[[ksi0, ksi0], [ksi1, ksi1]],
                                   [[ksi1, ksi1]]]
        zeta0=1.
        zeta1=1.101
        self.thermal_pressure_interaction = [[[zeta0, zeta0], [zeta1, zeta1]],
                                   [[zeta1, zeta1]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)

class FeSiO_liquid_Frost(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe - Fe0.5Si0.5 - FeO solution'
        self.type='subregular'
        self.endmembers = [[Fe_liq,     '[Fe]0.5[Fe]0.5'],
                           [Fe5Si5_liq, '[Fe]0.5[Si]0.5'],
                           [FeO_liq,    'Fe[O]0.5[O]0.5']]

        self.enthalpy_interaction = [[[0., 0.], [83307.,135943.]],
                                    [[83307.,135943.]]]
        
        self.entropy_interaction = [[[0., 0.], [8.978, 31.122]],
                                   [[8.978, 31.122]]]

        self.volume_interaction = [[[0., 0.], [-0.9e-6,-0.059e-6]],
                                   [[-0.9e-6,-0.059e-6]]]

        burnman.SolidSolution.__init__(self, molar_fractions)

class ferropericlase(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ferropericlase'
        self.endmembers = [[burnman.minerals.SLB_2011.periclase(), '[Mg]O'],[FeO_B1, '[Fe]O']] # NOTE that SLB_2011_per has a different base energy...
        self.type='symmetric'
        self.enthalpy_interaction=[[13.e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)

class mg_fe_bridgmanite(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='ferropericlase'
        self.endmembers = [[burnman.minerals.HHPH_2013.mpv(), '[Mg]SiO3'],[burnman.minerals.HHPH_2013.fpv(), '[Fe]SiO3']]
        self.type='symmetric'
        self.enthalpy_interaction=[[0.e3]]

        burnman.SolidSolution.__init__(self, molar_fractions)
bdg = mg_fe_bridgmanite()

#liq = FeSiO_liquid_Frost()
liq = FeSiO_liquid()
fper = ferropericlase()


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
    temperatures = np.linspace(2900., 3300., 3)
    pressures = np.linspace(20.e9, 140.e9, 11)
    for T in temperatures:
        lnKd = np.empty_like(pressures)
        guess=0.01
        for i, P in enumerate(pressures):
            XFeO_melt = fsolve(fper_melt_eqm, [guess], args=(P, T, XFeO_mw))[0]
            guess = XFeO_melt # for next step
            lnKd[i] = np.log(XFeO_melt/XFeO_mw)
            print P/1.e9, T, lnKd[i]
        plt.plot(pressures, lnKd, label=str(T)+'K')
    
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
    

    P = 25.e9
    XFeSiO3_bdg = 0.1
    temperatures = np.linspace(2773., 4273., 2)
    Xs_Fe5Si5_melt = np.linspace(0.002, 0.5, 11) 

    for T in temperatures:
        X_Sis_wt = np.empty_like(Xs_Fe5Si5_melt)
        X_Os_wt = np.empty_like(Xs_Fe5Si5_melt)
        guess = 0.01
        for i, XFe5Si5_melt in enumerate(Xs_Fe5Si5_melt):
            XFeO_melt = fsolve(melt_bdg_eqm, [guess], args=(P, T, XFeSiO3_bdg, XFe5Si5_melt))[0]
            print T, XFe5Si5_melt, XFeO_melt 
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
    
        plt.plot(X_Os_wt, X_Sis_wt, label=str(T)+'K')
    
    plt.title('Metallic melt in equilibrium with '+\
              'Mg'+str(1.-XFeSiO3_bdg)+\
              'Fe'+str(XFeSiO3_bdg)+\
              'SiO3 at '+str(P/1.e9)+' GPa')
    plt.xlabel('XO (wt %)')
    plt.ylabel('XSi (wt %)')
    plt.xlim(0.0, 10.0)
    plt.ylim(0.0, 10.0)
    plt.legend(loc='upper right')
    plt.show()
    
