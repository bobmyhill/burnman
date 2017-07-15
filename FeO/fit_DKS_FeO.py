import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

from scipy.optimize import fsolve, curve_fit
import burnman
from burnman import minerals
from B1_wuestite import B1_wuestite

class FeO_liquid(burnman.Mineral):
    def __init__(self):
        #self.params = {'xi': 0.4, 'Tel_0': 2000.0, 'equation_of_state': 'dks_l', 'zeta_0': -0.002808709680209786, 'T_0': 8000.0, 'O_f': 4, 'el_V_0': 1.7129414584934439e-05, 'a': [[-107699.37215834828, -120250.47622745327, -374463.79617534496], [661080.43391359923, 1541260.6692052803, -275064.3030853378], [5472873.3242191793, -4244906.9188206643, -9111862.5366244931], [-4014897.4177358989, 7677054.5902194316, 0.0], [85250226.142369315, 0.0, 0.0]], 'V_0': 1.7129414584934439e-05, 'name': 'FeO_liquid', 'm': 0.63, 'O_theta': 2, 'eta': -1.0, 'formula': {'Fe': 1.0, 'O': 1.0}}
        self.params = {'xi': 0.4, 'Tel_0': 2000.0, 'equation_of_state': 'dks_l', 'zeta_0': 0.0024257065617444393, 'T_0': 8000.0, 'O_f': 4, 'el_V_0': 1.81044909456887e-05, 'a': [[-121067.64971185905, -146030.3431653101, -333418.70122137183], [327392.3159095949, 734700.3123169115, -2152192.059275142], [6090297.376045127, -1303771.1057723137, 2985350.1952913064], [-9726174.025790948, -587092.0412079562, 0.0], [90383884.86072995, 0.0, 0.0]], 'V_0': 1.81044909456887e-05, 'name': 'FeO_liquid', 'm': 0.63, 'O_theta': 2, 'eta': -1.0, 'formula': {'Fe': 1.0, 'O': 1.0}}
        burnman.Mineral.__init__(self)


if __name__ == "__main__":
    per_liq = minerals.DKS_2013_liquids.MgO_liquid()
    wus_liq = FeO_liquid()
    fa_liq = minerals.RS_2014_liquids.Fe2SiO4_liquid()
    stv_liq = minerals.DKS_2013_liquids_tweaked.SiO2_liquid()
    
    wus = B1_wuestite()
    #wus = minerals.SLB_2011.wuestite()
    per = minerals.SLB_2011.periclase()
    
    '''
    print per_liq.params
    per_liq.set_state(1.e5,  per_liq.params['T_0'])
    print per_liq.params['V_0'], per_liq.V
    wus_liq.set_state(1.e5,  wus_liq.params['T_0'])
    print wus_liq.params['V_0'], wus_liq.V
    '''
    
    # Parameters to vary:
    # a00, a01, a02, a10, a11, a12, a20, a21, a22, a30, a31 (11 parameters)
    # V_0
    # zeta_0
    # Tel_0
    
    # i.e. 14 parameters in total
    
    def fit_PVT_data(mineral, PT, V, V_sigma=None):
        def fit_data(PT, *params):
            print params

            prms = mineral.params
            
            mineral.params['V_0'] = params[0]
            mineral.params['el_V_0'] = params[0]
            mineral.params['zeta_0'] = params[1]
            
            
            mineral.params['a'] = [[prms['a'][0][0], prms['a'][0][1], prms['a'][0][2]],
                                   [params[2], params[3], params[4]],
                                   [params[5], params[6], params[7]],
                                   [params[8], params[9], 0.],
                                   [params[10], 0., 0.]]
            burnman.Mineral.__init__(mineral)
            '''
             # Now iterate until V_0 = the calculated V_0
             diffV = 1.
             while np.abs(diffV) > 1.e-9:
                 mineral.set_state(0., mineral.params['T_0'])
                 print mineral.V, mineral.params['V_0']
                 diffV = mineral.V - mineral.params['V_0']
                 mineral.params['V_0'] = mineral.V
            '''
         
            volumes = []
            for P, T in PT:
                mineral.set_state(P, T)
                volumes.append(mineral.V)
            return volumes

        p = mineral.params
        guesses = [p['V_0'], p['zeta_0'],
                   p['a'][1][0], p['a'][1][1], p['a'][1][2],
                   p['a'][2][0], p['a'][2][1], p['a'][2][2],
                   p['a'][3][0], p['a'][3][1],
                   p['a'][4][0]]
         
         
        popt, pcov = curve_fit(fit_data, PT, V, guesses, V_sigma, method='trf')

        return popt, pcov


    # Data input (taken from Holmstrom and Stixrude, 2016, with an additional three data points from the experiments of Hara et al., 1988 (possibly tweaked?).
    T, P, V = np.loadtxt(fname='../FeO/figures/Mg75Fe25O_TPV_Holmstrom_Stixrude_2016.dat', unpack='True')
    PT = zip(*[P*1.e9, T])

    V_MgO = []
    V_FeO = []
    for i, state in enumerate(PT):
        per_liq.set_state(*state)
        solution_volume = V[i]*burnman.constants.Avogadro*2./1.e30
        print solution_volume, per_liq.V
        
        # 4. * Mg75Fe25O = 3. * MgO + FeO
        # FeO = 4. * Mg75Fe25O - 3. * MgO
        
        V_MgO.append(per_liq.V)
        V_FeO.append(4. * solution_volume - 3. * per_liq.V)
        

    P = 1.e5
    T = 1650.
    PT.append((P, T))
    per_liq.set_state(P, T)
    V_MgO.append(per_liq.V)
    V_FeO.append(0.071844/4800.) # 5000. #4632. # 4632 is extrapolated from Hara et al., 1988a, b, (BaO and CaO solutions) similar to Ji et al., 1997 (4700)
    
    V_fudge = 6.e-7
    
    P = 300.e9
    T = 1650.
    PT.append((P, T))
    fa_liq.set_state(P, T)
    stv_liq.set_state(P, T)
    per_liq.set_state(P, T)
    V_MgO.append(per_liq.V)
    V_FeO.append((fa_liq.V - stv_liq.V)/2. + V_fudge)
    popt, pcov = fit_PVT_data(wus_liq, PT, V_FeO)

    def thermal_data(PT, *params):
        # properties at 1650 K
        # Gibbs = wuestite gibbs
        # Entropy = see Coughlin
        # Heat capacity = see Coughlin
        wus_liq.params['a'][0][0] = params[0]
        wus_liq.params['a'][0][1] = params[1]
        wus_liq.params['a'][0][2] = params[2]
        burnman.Mineral.__init__(wus_liq)
        
        wus_liq.set_state(1.e5, 1650.)
        return [wus_liq.gibbs, wus_liq.S, wus_liq.heat_capacity_p]

    S_fudge = 16.
    C_p_inc = 10.
    wus.set_state(1.e5, 1650.)
    GSCp = [wus.gibbs, wus.S + 19.*2./1.947 + S_fudge, wus.heat_capacity_p + C_p_inc]
    
    popt, pcov = curve_fit(thermal_data, [(), (), ()], GSCp, [0., 0., 0.])
    
    print wus_liq.params
    
    wus.set_state(1.e5, 1650.)
    wus_liq.set_state(1.e5, 1650.)

    print (wus.V-wus_liq.V)/(wus.S - wus_liq.S)*1.e9
    
    
    # plot entropy
    T_C, Hdiff, Sdiff = np.loadtxt(unpack='True', fname="data/Coughlin_KB_1951_FeO_HS_solid_liquid.dat")
    wus.set_state(1.e5, 298.15)
    S_C = wus.S + 4.184*Sdiff*(2./1.947)
    
    plt.plot(T_C, S_C, marker='o', label='Coughlin et al.')
    
    
    temperatures = np.linspace(1500., 2900., 101)
    entropies = np.empty_like(temperatures)
    entropies2 = np.empty_like(temperatures)
    entropies3 = np.empty_like(temperatures)
    entropies4 = np.empty_like(temperatures)
    for P in [1.e9, 10.e9, 20.e9, 100.e9]:
        for i, T in enumerate(temperatures):
            try:
                wus_liq.set_state(P, T)
                wus.set_state(P, T)
                per_liq.set_state(P, T)
                per.set_state(P, T)
                entropies[i] = wus_liq.S
                entropies2[i] = wus.S
                entropies3[i] = per_liq.S
                entropies4[i] = per.S
            except:
                print T
                
                entropies[i] = 0.
                entropies2[i] = 0.
                entropies3[i] = 0.
                entropies4[i] = 0.
        plt.plot(temperatures, entropies, label='wus_liq, P='+str(P/1.e9))
        plt.plot(temperatures, entropies2, label='wus, P='+str(P/1.e9))
        #plt.plot(temperatures, entropies3, linestyle='--', label='per liq')
        #plt.plot(temperatures, entropies4, linestyle='--', label='per')

    plt.legend(loc='upper right')
    plt.show()


    pressures = np.linspace(1.e5, 360.e9, 101)
    temperatures = np.empty_like(pressures)
    for i, P in enumerate(pressures):
        try:
            temperatures[i] = burnman.tools.equilibrium_temperature([wus, wus_liq], [1.0, -1.0], P, 3000.)
            print P/1.e9, temperatures[i]
        except:
            print 'Equilibrium search failed at '+str(P/1.e9)+' GPa'
            
    fig1 = mpimg.imread('figures/FeO_melting_curve.png')
    plt.imshow(fig1, extent=[0., 120., 1000., 6000.], aspect='auto')
    plt.plot(pressures/1.e9, temperatures, linewidth=4.)
    #plt.xlim(0., 20.)
    #plt.ylim(1500., 4000.)
    plt.show()
    #exit()
    


    wus_liq.set_state(0., wus_liq.params['T_0'])
    print wus_liq.V, wus_liq.params['V_0']





    pressures = np.linspace(10.e9, 200.e9, 40)
    volumes = np.empty_like(pressures)
    volumes2 = np.empty_like(pressures)
    for T in [2000., 4000., 6000., 8000.]:
        for i, p in enumerate(pressures):
            wus_liq.set_state(p, T)
            wus.set_state(p, T)
            volumes[i] = wus_liq.V
            volumes2[i] = wus.V
            print p, T, wus_liq.V
        plt.plot(pressures/1.e9, volumes, label='wus liquid')
        plt.plot(pressures/1.e9, volumes2, linestyle='--', label='wus')

    plt.plot(np.array(zip(*PT)[0])/1.e9, V_MgO, marker='.', linestyle='None', label='MgO')
    plt.plot(np.array(zip(*PT)[0])/1.e9, V_FeO, marker='o', linestyle='None', label='FeO')

    plt.legend(loc='lower right')
    plt.show()


    pressures = np.linspace(10.e9, 200.e9, 40)
    volumes = np.empty_like(pressures)
    volumes2 = np.empty_like(pressures)
    for T in [2000., 4000.]:
        for i, p in enumerate(pressures):
            wus.set_state(p, T)
            per.set_state(p, T)
            volumes[i] = wus.V
            volumes2[i] = per.V
        plt.plot(pressures/1.e9, volumes)
        plt.plot(pressures/1.e9, volumes2, linestyle='--')
    plt.legend(loc='upper right')
    plt.show()
            




    pressures = np.linspace(1.e9, 300.e9)
    V_diff = np.empty_like(pressures)
    
    for T in [2000., 3000., 4000., 5000.]:
        for i, P in enumerate(pressures):
            for liq in [wus_liq, fa_liq, stv_liq]:
                liq.set_state(P, T)
                
                V_diff[i] = 2.*wus_liq.V + 1.*stv_liq.V - fa_liq.V

        plt.plot(pressures, V_diff, label=str(T)+' K')

    plt.legend(loc='upper right')
    plt.show()
