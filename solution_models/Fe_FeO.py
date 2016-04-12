import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals

if __name__ == "__main__":
    read_data=True
    fit_data=True
    plot_interaction=True
    plot_eutectic=True
    plot_isobaric_diagram=True
    plot_solvus = True
else:
    read_data=False
    fit_data=False
    plot_interaction=False
    plot_eutectic=False
    plot_isobaric_diagram=False
    plot_solvus = False



from B1_wuestite import B1_wuestite
from liq_wuestite_AA1994 import liq_FeO
from fcc_iron import fcc_iron
from hcp_iron import hcp_iron
from liq_iron_AA1994 import liq_iron

Fe_hcp = hcp_iron()
Fe_fcc = fcc_iron()
Fe_liq = liq_iron()
FeO_B1 = B1_wuestite()
FeO_liq = liq_FeO()


class Fe_FeO_liquid(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe-FeO solution'
        self.type='full_subregular'
        self.P_0=1.e5
        self.T_0=1650.
        self.n_atoms=1.
        self.endmembers = [[liq_iron(), '[Fe]'], [liq_FeO(), 'Fe[O]']]
        self.energy_interaction = [[[83.6e3, 136.2e3]]]
        # [[[60.e3, 101.e3]]]        # [[[84.e3, 137.e3]]]
        # [[[105.e3, 118.e3]]]     # [[[45.e3, 53.e3]]] # 
        self.volume_interaction = [[[-1.33e-06, -0.93e-06]]]
        # [[[-1.61e-06, -1.27e-06]]] # [[[-1.39e-06, -0.96e-06]]]
        # [[[-1.403e-6, -1.2e-6]]] # [[[-1.85e-06,  -1.7e-06]]] #
        self.kprime_interaction = [[[0.1, 0.1]]]
        # [[[1.01, 1.01]]]           # [[[7./3., 7./3.]]]
        self.thermal_pressure_interaction = [[[1.064, 1.139]]]
        # [[[1.024, 1.079]]]         # [[[1.063, 1.139]]]
        # [[[1.101, 1.101]]]       # [[[1., 1.]]] #
        burnman.SolidSolution.__init__(self, molar_fractions)

class Fe_FeO_liquid_Frost(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe - Fe0.5Si0.5 - FeO solution'
        self.type='subregular'
        self.endmembers = [[Fe_liq,     '[Fe]'],
                           [FeO_liq,    'Fe[O]']]

        self.enthalpy_interaction = [[[83307.,135943.]]]
        
        self.entropy_interaction = [[[8.978, 31.122]]]

        self.volume_interaction = [[[-0.9e-6,-0.59e-6]]]

        burnman.SolidSolution.__init__(self, molar_fractions)


class Fe_FeO_liquid_simple(burnman.SolidSolution):
    def __init__(self, molar_fractions=None):
        self.name='Fe - Fe0.5Si0.5 - FeO solution'
        self.type='simple_subregular'
        self.endmembers = [[Fe_liq,     '[Fe]'],
                           [FeO_liq,    'Fe[O]']]

        self.energy_interaction = [[[83307.,135943.]]]
        
        self.entropy_interaction = [[[8.978, 31.122]]]

        self.volume_interaction = [[[-1.11e-6,-0.55e-6]]]

        kxs = 100.e9
        self.modulus_interaction = [[[kxs, kxs]]]
        
        burnman.SolidSolution.__init__(self, molar_fractions)
        
#liq = Fe_FeO_liquid()
#liq = Fe_FeO_liquid_Frost()
liq = Fe_FeO_liquid_simple()

    

if read_data==True:
    # READ IN DATA
    eutectic_PT = []
    
    f=open('data/Fe_FeO_eutectic_temperature.dat', 'r')
    datastream = f.read()  # We need to open the file
    f.close()
    datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
    for line in datalines:
        if line[0] != "%":
            eutectic_PT.append([float(line[0])*1.e9, float(line[1]), 
                                float(line[2]), float(line[3])])
    
    
    eutectic_PTc = []
    f=open('data/Fe_FeO_eutectic.dat', 'r') 
    datastream = f.read()  # We need to open the file
    f.close()
    datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
    for line in datalines:
        if line[0] != "%":
            # P, Perr, T, Terr, FeO (mol fraction), err
            eutectic_PTc.append([float(line[1])*1.e9, 2.0e9, 
                                float(line[2]), float(line[3]), 
                                float(line[4])/100., float(line[5])/100.])
    
    solvus_PTcc = []
    f=open('data/Fe_FeO_solvus.dat', 'r')
    datastream = f.read()  # We need to open the file
    f.close()
    datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
    for line in datalines:
        if line[0] != "%":
            # P, Perr, T, Terr, FeO (mol fraction), err, FeO (mol fraction), err
            solvus_PTcc.append([float(line[1])*1.e9, 2.0e9, 
                                float(line[2]), float(line[3]), 
                                float(line[4])/100., float(line[5])/100.,
                                float(line[8])/100., float(line[9])/100.])

    eutectic_PT = np.array(eutectic_PT).T
    eutectic_PTc = np.array(eutectic_PTc).T
    solvus_PTcc = np.array(solvus_PTcc).T
  
    def fit_excess_solvus(args, compositions, T):
        W_FeFeO, W_FeOFe = args
    
        Gxs_c = []
        for X_FeO in compositions:
            X_Fe = 1. - X_FeO
            
            RTlny_Fe = (W_FeFeO + 2. * (W_FeOFe - W_FeFeO) * X_Fe) * X_FeO * X_FeO
            RTlny_FeO = (W_FeOFe + 2. * (W_FeFeO - W_FeOFe) * X_FeO) * X_Fe * X_Fe
        
            RTlnX_Fe = burnman.constants.gas_constant*T*np.log(X_Fe)
            RTlnX_FeO = burnman.constants.gas_constant*T*np.log(X_FeO) 
            
            Gxs_c.append([RTlnX_Fe + RTlny_Fe, RTlnX_FeO + RTlny_FeO])
    
        return [Gxs_c[0][0] - Gxs_c[1][0],
                Gxs_c[0][1] - Gxs_c[1][1]]
    
    
    
    # FITTING DATA
    fitting_data = []
    HP_weighting = 2.
    eutectic_weighting = 5.
    
    # First, find the gibbs free energy difference corresponding to
    # Fe (solid-liquid) and FeO (solid-liquid)
    # These will be the partial excess gibbs free energies of the liquid phase
    # Seagle et al. (2008) reported HCP iron above ~50 GPa
    '''
    for datum in eutectic_PTc.T:
        P, Perr, T, Terr, X_FeO, X_FeO_err = datum
        
        X_Fe = 1. - X_FeO
        
        Fe_hcp.set_state(P, T)
        Fe_liq.set_state(P, T)
        FeO_B1.set_state(P, T)
        FeO_liq.set_state(P, T)
    
        Gxs_Fe = (Fe_hcp.gibbs - Fe_liq.gibbs)
        Gxs_FeO = (FeO_B1.gibbs - FeO_liq.gibbs)
        print Gxs_Fe, Gxs_FeO
        
        RTlnX_Fe = burnman.constants.gas_constant*T*np.log(X_Fe)
        RTlnX_FeO = burnman.constants.gas_constant*T*np.log(X_FeO) 
    
        RTlny_Fe = Gxs_Fe - RTlnX_Fe
        RTlny_FeO = Gxs_FeO - RTlnX_FeO
    
        W_FeFeO = (RTlny_Fe/(X_FeO*X_FeO) + 2.*X_Fe*(RTlny_Fe/(X_FeO*X_FeO) + RTlny_FeO/(X_Fe*X_Fe)))/(1. + 4.*X_Fe)
        W_FeOFe = (RTlny_FeO/(X_Fe*X_Fe) + 2.*X_FeO*(RTlny_FeO/(X_Fe*X_Fe) + RTlny_Fe/(X_FeO*X_FeO)))/(1. + 4.*X_FeO)
        
        fitting_data.append([P, T, W_FeFeO, W_FeOFe, eutectic_weighting])
    
    '''

    # and now fit the solvus:
    # at a given pressure and temperature, two compositions have the same partial gibbs excesses
    # FCC IRON
    for datum in solvus_PTcc.T:
        P, Perr, T, Terr, c0, c0err, c1, c1err = datum
        Ws = fsolve(fit_excess_solvus, [200.e3, 200.e3], args=([c0, c1], T))
    
        fitting_data.append([P, T, Ws[0], Ws[1], 1.])




    # Ambient pressure values are from Kowalski and Spencer (1995)
    # Taken from Frost et al, 2010
    def Frost_interaction(P, T):
        WE_FeFeO = 83307.
        WS_FeFeO = 8.978
        WV_FeFeO = -0.09e-5 # convert from /bar to /Pa
        WE_FeOFe = 135943.
        WS_FeOFe = 31.122
        WV_FeOFe = -0.059e-5 # convert from /bar to /Pa
    
        W_FeFeO = WE_FeFeO - T*WS_FeFeO + P*WV_FeFeO
        W_FeOFe = WE_FeOFe - T*WS_FeOFe + P*WV_FeOFe
        
        return W_FeFeO, W_FeOFe

    P = 1.e5
    T = 1800.
    W0, W1 = Frost_interaction(P, T)
    fitting_data.append([P, T, W0, W1, 10.])
    P = 1.e5
    T = 2200.
    W0, W1 = Frost_interaction(P, T)
    fitting_data.append([P, T, W0, W1, 10.])
    
    pressures, temperatures, W_FeFeO, W_FeOFe, weighting = zip(*fitting_data)
    pressures = np.array(pressures)
    plt.plot(pressures/1.e9, W_FeFeO, marker='o', linestyle='None', label='W_FeFeO')
    plt.plot(pressures/1.e9, W_FeOFe, marker='o', linestyle='None', label='W_FeOFe')
    plt.legend(loc='upper right')
    plt.xlim(-1, 100.)
    plt.ylim(-20.e3, 100.e3)
    plt.show()

    def calculate_interaction_parameters(xdata):
        liq.set_composition([0.5, 0.5]) # doesn't matter what we choose here...
        Ws = []
        for x in xdata:
            P = x[0]
            T = x[1]
            liq.set_state(P, T)
            Wg = liq.solution_model.We - T*liq.solution_model.Ws + P*liq.solution_model.Wv
            if x[-1] == 0: # W_FeFeO
                Gex = Wg[0][1]
            else: # W_FeOFe
                Gex = Wg[1][0]
            Ws.append(Gex)
        return Ws
    
if fit_data==True:

    def find_interaction_parameters(xdata, E0, E1, V0, V1):
        print E0, E1, V0, V1
        # initialise properties
        liq.energy_interaction = [[[E0, E1]]]
        liq.volume_interaction = [[[V0, V1]]]
        burnman.SolidSolution.__init__(liq)
        return calculate_interaction_parameters(xdata)


    xdata=[]
    ydata=[]
    sigmas=[]
    for i, P in enumerate(pressures):
        xdata.append([pressures[i], temperatures[i], 0])
        ydata.append(W_FeFeO[i])
        xdata.append([pressures[i], temperatures[i], 1])
        ydata.append(W_FeOFe[i])
        sigmas.append(1./weighting[i])
        sigmas.append(1./weighting[i])
    
    guesses = [105.e3, 118.e3, -1.403e-06, -1.2e-06]
    #guesses = [109.e3, 109.e3, -1.38e-6, -1.38e-6, 7./3., 1.101] # includes fitting of eutectic
    #guesses = [108.e3, 116.e3, -1.18e-6, -1.18e-6, 7./3., 1.101] 
    #guesses = [0., 0., 0., 0., 7./3., 1.]
    
    #guesses = [111.e3, 110.e3, -7.525e-07, -1.53e-06, 0.362, 1.104] # includes fitting of eutectic
    #guesses = [111.e3, 109.e3, -8.759e-07, -1.88e-06, 7./3., 1.101]
    #guesses = [108.e3, 116.e3, -1.18e-06, 1.101]
    
    popt, pcov = curve_fit(find_interaction_parameters, xdata, ydata, guesses, sigmas)
    print 'Solution:', popt, pcov

    Ws = calculate_interaction_parameters(xdata)


if plot_interaction==True:
    print 'Plotting interaction energies'
    plt.plot(pressures/1.e9, W_FeFeO, marker='x', linestyle='None', label='W_FeFeO')
    plt.plot(pressures/1.e9, W_FeOFe, marker='o', linestyle='None', label='W_FeOFe')

    np.savetxt(fname='output_data/experimental_interactions.dat',
               X=zip(*[pressures/1.e9, temperatures, W_FeFeO, W_FeOFe]),
               header='Pressure (GPa) Temperature (K) W_FeFeO (J/mol) W_FeOFe (J/mol)')
    
    #plt.plot(np.array(zip(*xdata)[0][::2])/1.e9, Ws[::2], marker='o', linestyle='None', label='W_FeFeO, model')
    #plt.plot(np.array(zip(*xdata)[0][1::2])/1.e9, Ws[1::2], marker='o', linestyle='None', label='W_FeOFe, model')
    
    
    # Now plot a couple of isotherms
    for T in [2273., 3273.]:
        pressures = np.linspace(1.e5, 250.e9, 10.)
        temperatures = pressures*0. + T
        interactions = []
        for f in [0, 1]:
            print 'Plotting', T, 'K ('+str(f)+')'
            flag = (pressures*0. + f)
            flag = flag.astype(int)
            Ws = calculate_interaction_parameters(zip(*[pressures, temperatures, flag]))
            plt.plot(pressures/1.e9, Ws, label=str(T)+' K')
            interactions.append(Ws)
        np.savetxt(fname='output_data/interaction_terms_'+str(T)+'_K.dat',
                   X=zip(*[pressures/1.e9, interactions[0], interactions[1]]),
                   header='Pressure (GPa) W_FeFeO (J/mol) W_FeOFe (J/mol)')
            
    W_FeFeO, W_FeOFe = Frost_interaction(pressures, T)
    plt.plot(pressures/1.e9, W_FeFeO, linestyle='--', label=str(T)+' K; Frost')
    plt.plot(pressures/1.e9, W_FeOFe, linestyle='--', label=str(T)+' K; Frost')
 
    plt.legend(loc='upper right')
    plt.xlim(-1., 250.)
    plt.ylim(-20000., 81000.)
    plt.show()


def eutectic_liquid(cT, P, liq, Fe_phase, FeO_phase):
    c, T = cT
        
    liq.set_composition([1.-c, c])
    liq.set_state(P, T)
    Fe_phase.set_state(P, T)
    FeO_phase.set_state(P, T)
    
    equations = [ Fe_phase.gibbs - liq.partial_gibbs[0],
                  FeO_phase.gibbs - liq.partial_gibbs[1]]
    return equations

# Plot eutectic temperatures and compositions
if plot_eutectic==True:
    print 'Plotting eutectic temperatures and compositions'
    pressures = np.linspace(30.e9, 250.e9, 16)
    eutectic_compositions = np.empty_like(pressures)
    eutectic_temperatures = np.empty_like(pressures)
    
    c, T = [0.5, 4000.]
    for i, P in enumerate(pressures):
        c, T = fsolve(eutectic_liquid, [c, T], args=(P, liq, Fe_hcp, FeO_B1))
        print c, T
        eutectic_compositions[i] = c
        eutectic_temperatures[i] = T


    np.savetxt(fname='output_data/eutectic_TX.dat',
               X=zip(*[pressures/1.e9, eutectic_temperatures, eutectic_compositions]),
               header='Pressure (GPa) Temperature (K) X FeO (mol fraction)')
        
    plt.plot(pressures/1.e9, eutectic_compositions)
    plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[4], marker='o', linestyle='None', label='Model')
    plt.legend(loc='lower right')
    plt.show()
    
    plt.plot(pressures/1.e9, eutectic_temperatures)
    plt.plot(eutectic_PT[0]/1.e9, eutectic_PT[2], marker='o', linestyle='None', label='Model')
    plt.plot(eutectic_PTc[0]/1.e9, eutectic_PTc[2], marker='o', linestyle='None', label='Model')
    plt.legend(loc='lower right')
    plt.show()

# Phase diagram at 50 GPa
if plot_isobaric_diagram==True:
    print 'Plotting phase diagram at 50 GPa'
    
    P = 50.e9
    Fe_phase = Fe_hcp
    
    def mineral_fugacity(c, mineral, liquid, P, T):
        mineral.set_state(P, T)
        liq.set_composition([1. - c[0], c[0]])
        liq.set_state(P, T)
        return [burnman.chemicalpotentials.fugacity(mineral, [liquid]) - 1.]
    
    def molFeO2wtO(molFeO):
        molFe = 1.
        molO = molFeO
        massFe = molFe*55.845
        massO = molO*15.9994
        return massO/(massFe+massO)*100.
    def molFeO2wtFeO(molFeO):
        molFe = 1. - molFeO
        massFe = molFe*55.845
        massFeO = molFeO*(15.9994+55.845)
        return massFeO/(massFe+massFeO)*100.
    
    
    c, T_eutectic = fsolve(eutectic_liquid, [0.1, 2500.], args=(P, liq, Fe_phase, FeO_B1))
    T_Fe_melt = burnman.tools.equilibrium_temperature([Fe_phase, Fe_liq], [1., -1.], P)
    T_FeO_melt = burnman.tools.equilibrium_temperature([FeO_B1, FeO_liq], [1., -1.], P)
    
    print molFeO2wtO(c), T_eutectic, T_Fe_melt, T_FeO_melt
    
    temperatures = np.linspace(T_eutectic, T_Fe_melt, 20)
    Fe_liquidus_compositions = np.empty_like(temperatures)
    c=0.01
    for i, T in enumerate(temperatures):
        print i, T
        c = fsolve(mineral_fugacity, [c], args=(Fe_phase, liq, P, T))[0]
        Fe_liquidus_compositions[i] = c
    plt.plot(molFeO2wtO(Fe_liquidus_compositions), temperatures)

    np.savetxt(fname='output_data/Fe_liquidus_'+str(P/1.e9)+'_GPa.dat',
               X=zip(*[Fe_liquidus_compositions, molFeO2wtO(Fe_liquidus_compositions), temperatures]),
               header='X_FeO (mol fraction) X_FeO (wt % O) Temperature (K)')
        
    temperatures = np.linspace(T_eutectic, T_FeO_melt, 20)
    c=0.99
    FeO_liquidus_compositions = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        print i, T
        c = fsolve(mineral_fugacity, [c], args=(FeO_B1, liq, P, T))[0]
        FeO_liquidus_compositions[i] = c
    plt.plot(molFeO2wtO(FeO_liquidus_compositions), temperatures)
    plt.xlim(0., 23.)
    plt.xlabel('wt % O')

    np.savetxt(fname='output_data/FeO_liquidus_'+str(P/1.e9)+'_GPa.dat',
               X=zip(*[FeO_liquidus_compositions, molFeO2wtO(FeO_liquidus_compositions), temperatures]),
               header='X_FeO (mol fraction) X_FeO (wt % O) Temperature (K)')
        
    # Seagle data
    lo_eutectic = np.loadtxt(fname='data/Seagle_2008_low_eutectic_bounds.dat', comments='%')
    hi_eutectic = np.loadtxt(fname='data/Seagle_2008_high_eutectic_bounds.dat', comments='%')
    lo_liquidus = np.loadtxt(fname='data/Seagle_2008_low_liquidus_bounds.dat', comments='%')
    hi_liquidus = np.loadtxt(fname='data/Seagle_2008_high_liquidus_bounds.dat', comments='%')
    
    for a in [lo_eutectic, hi_eutectic, lo_liquidus, hi_liquidus]:
        a = a[a[:,0]>=45.0, :]
        a = a[a[:,0]<=55.0, :]
        plt.plot(a[:,3], a[:,2], marker='s', markersize=10, linestyle='None')
    
    plt.show()
    
    
# Plot solvus
if plot_solvus == True:
    print 'Plotting solvus'
    
    def eqm_two_liquid(cc, P, T, model):
        c1, c2 = cc
    
        model.set_composition([1.-c1, c1])
        model.set_state(P, T)
    
        partial_excesses_1 = model.excess_partial_gibbs
        model.set_composition([1.-c2, c2])
        model.set_state(P, T)
        partial_excesses_2 = model.excess_partial_gibbs
        equations = [ partial_excesses_1[0] - partial_excesses_2[0],
                    partial_excesses_1[1] - partial_excesses_2[1]]
        return equations
    
    temperatures = [2173., 2373., 2573.]
    pressures = np.linspace(1.e5, 30.e9, 31)
    
    for T in temperatures:
        print T
        c1=0.01
        c2=0.99
        
        pressures_1_2 = []
        compositions_1 = []
        compositions_2 = []
        
        for i, P in enumerate(pressures):
            print P
            if np.abs(c1 - c2) > 1.e-4:
                c1, c2 = fsolve(eqm_two_liquid, [c1, c2],
                                args=(P, T, liq), factor = 0.1, xtol=1.e-6)

            if np.abs(c1 - c2) > 1.e-4:
                pressures_1_2.append(P)
                compositions_1.append(c1)
                compositions_2.append(c2)

        plot_pressures = np.concatenate((pressures_1_2, pressures_1_2[::-1]))
        plot_compositions = np.concatenate((compositions_1, compositions_2[::-1]))
                
        plt.plot(plot_compositions, plot_pressures/1.e9, label=str(T)+' K')
        np.savetxt(fname='output_data/Fe_FeO_solvus_'+str(T)+'_K.dat', X=zip(*[plot_compositions, plot_pressures/1.e9]), header='X FeO (mol fraction), P (GPa)')
        
        
    plt.plot(solvus_PTcc[4], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')
    plt.plot(solvus_PTcc[6], solvus_PTcc[0]/1.e9, marker='o', linestyle='None')
    plt.legend(loc='upper right')
    plt.show()
    
