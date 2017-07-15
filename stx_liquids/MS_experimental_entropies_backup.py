import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import HP_2011_ds62
from burnman.minerals import SLB_2011
from burnman.minerals import DKS_2008_fo
from burnman.minerals import DKS_2013_solids
from burnman.minerals import DKS_2013_liquids
from burnman import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit


test_FeO = False
test_MgO = False
test_Mg2SiO4 = False
test_SiO2 = True
test_mix = False


MgO_liq = DKS_2013_liquids.MgO_liquid()
Mg2SiO4_liq = DKS_2013_liquids.Mg2SiO4_liquid()
#Mg2SiO4_liq = DKS_2008_fo.fo_liquid()
SiO2_liq = DKS_2013_liquids.SiO2_liquid()
mins = [MgO_liq, Mg2SiO4_liq, SiO2_liq]


per_2 = SLB_2011.periclase()
per_2 = DKS_2013_solids.periclase()

per = HP_2011_ds62.per()
fo = HP_2011_ds62.fo()
fo = SLB_2011.fo()
#fo.params['grueneisen_0'] = 1.2 # was 0.99282
#fo.params['q_0'] = 1.05 # was 2.1
#print fo.params

'''
fo.set_state(1.e5, 2163.)
S = fo.S
fo = DKS_2008_fo.forsterite()
fo.set_state(1.e5, 2163.)
fo.property_modifiers = [['linear', {'delta_E': 0., 'delta_S': S - fo.S, 'delta_V': 0.}]]
burnman.Mineral.__init__(fo)
'''

qtz = HP_2011_ds62.q()
crst = HP_2011_ds62.crst()
coe = HP_2011_ds62.coe()
stv = HP_2011_ds62.stv()
stv.params['H_0'] = stv.params['H_0'] - 4.e3

'''
stv.set_state(1.e5, 1000.)
G0 = stv.gibbs
stv = SLB_2011.stishovite()
stv.set_state(1.e5, 1000.)
stv.params['F_0'] = stv.params['F_0'] - stv.gibbs + G0
'''

# VALUES TO FIT DeKOKER MODEL TO EXPERIMENTAL CONSTRAINTS

print 'for an increase in entropy with temperature,'
print 'the temperature dependence of delta_S must be'
print 'smaller than 1.33*np.log(T) - 0.00433*T'


a = [ 1., -0.005]
delta_S = lambda T: [-3., -29., a[0]*np.log(T/1999.) + a[1]*(T-1999.) + 9.85]
delta_S = lambda T: [-3., -12., a[0]*np.log(T/1999.) + a[1]*(T-1999.) + 9.85]
delta_Cp = lambda T: [0., 0., a[0] + a[1]*T]



SiO2_data = np.loadtxt(fname='data/JANAF_SiO2.dat', unpack=True)
MgO_data = np.loadtxt(fname='data/JANAF_MgO_liq.dat', unpack=True)
FeO_data = np.loadtxt(fname='data/JANAF_FeO.dat', unpack=True)
Mg2SiO4_data = np.loadtxt(fname='data/JANAF_Mg2SiO4.dat', unpack=True)

# WUESTITE MELTING
if test_FeO == True:
    plt.plot(FeO_data[0], FeO_data[2])
    plt.show()

# PERICLASE MELTING
if test_MgO == True:
    plt.plot(MgO_data[0], MgO_data[2])
    
    
    temperatures = np.linspace(500, 3000, 101)
    S_liq = np.empty_like(temperatures)
    S_per = np.empty_like(temperatures)
    S_per_2 = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        MgO_liq.set_state(1.e5, T)
        per.set_state(1.e5, T)
        per_2.set_state(1.e5, T)
        S_liq[i] = MgO_liq.S + delta_S(T)[0]
        S_per[i] = per.S
        S_per_2[i] = per_2.S
        
        print 'Cp difference at', T, 'K:', MgO_liq.heat_capacity_p - per.heat_capacity_p 
        
        
    plt.plot(temperatures, S_liq)
    plt.plot(temperatures, S_per)
    plt.plot(temperatures, S_per_2)
    plt.show()
    

    # Make a correction to fit the melting curve of MgO
    Pm = 1.e5
    Tm = 2852. + 273.15
    MgO_liq.set_state(Pm, Tm)
    per_2.set_state(Pm, Tm)
    delta_S_liq = delta_S(Tm)[0]
    delta_E_liq = per_2.gibbs - MgO_liq.gibbs + delta_S_liq*Tm
    delta_V_liq = 0.0
    MgO_liq.property_modifiers = [['linear', {'delta_E': delta_E_liq, 'delta_S': delta_S_liq, 'delta_V': delta_V_liq}]]
    burnman.Mineral.__init__(MgO_liq)
    
    fig1 = mpimg.imread('figures/Alfe_MgO_melting.png')
    plt.imshow(fig1, extent=[0., 150., 2000., 9000.], aspect='auto')
    
    pressures = np.linspace(1.e5, 150.e9, 101)
    temperatures = np.empty_like(pressures)
    S_diff = np.empty_like(pressures)
    Tguess = Tm
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([per_2, MgO_liq], [1., -1.], P, Tguess)
        Tguess = temperatures[i]
        S_diff[i] = MgO_liq.S - per_2.S
        print P, temperatures[i]



    plt.plot(pressures/1.e9, temperatures)
    plt.show()
    
    plt.plot(pressures, S_diff)
    plt.show()


# FORSTERITE MELTING
if test_Mg2SiO4 == True:
        
    temperatures = np.linspace(500., 3000., 31)
    
    fig1 = mpimg.imread('figures/entropies_fo_0_700J_Richet_1993.png')
    plt.imshow(fig1, extent=[0., 2500., 0., 700.], aspect='auto')
    
    
    plt.plot(Mg2SiO4_data[0], Mg2SiO4_data[2])
    
    S_sol = np.empty_like(temperatures)
    V_sol = np.empty_like(temperatures)
    S_liq = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        fo.set_state(1.e5, T)
        Mg2SiO4_liq.set_state(1.e5, T)
        
        S_sol[i] = fo.S
        S_liq[i] = Mg2SiO4_liq.S + delta_S(T)[1]

        V_sol[i] = fo.V
        
        print 'Cp difference (1 cation basis) at', T, 'K:', (Mg2SiO4_liq.heat_capacity_p - fo.heat_capacity_p)/3.
        
    plt.plot(temperatures, S_sol, label='solid')
    plt.plot(temperatures, S_liq, label='liquid')
    plt.ylim(200., 1000.)
    plt.xlim(500, 5000)
    plt.legend(loc='lower right')
    plt.show()

    fig1 = mpimg.imread('figures/fo_relative_volume.png')
    plt.imshow(fig1, extent=[300., 2300., fo.params['V_0'], 1.08*fo.params['V_0']], aspect='auto')    
    plt.plot(temperatures, V_sol)
    plt.show()
    
    # Make a correction to fit the melting curve of Mg2SiO4
    Pm = 1.e5
    Tm = 1890. + 273.15
    Mg2SiO4_liq.set_state(Pm, Tm)
    fo.set_state(Pm, Tm)
    
    delta_S_liq = delta_S(Tm)[1]
    delta_E_liq = fo.gibbs - Mg2SiO4_liq.gibbs + delta_S_liq*Tm
    delta_V_liq = 0. # -1.e-6
    Mg2SiO4_liq.property_modifiers = [['linear', {'delta_E': delta_E_liq, 'delta_S': delta_S_liq, 'delta_V': delta_V_liq}]]
    burnman.Mineral.__init__(Mg2SiO4_liq)

    Mg2SiO4_liq.set_state(Pm, Tm)
    fo.set_state(Pm, Tm)
    print (Mg2SiO4_liq.V - fo.V)*1.e6, 'Richet et al., 1993 suggest 6.41 cm^3/mol'
    print  Mg2SiO4_liq.S - fo.S, 'Richet et al., 1993 suggest 65.3 J/K/mol'

    print (Mg2SiO4_liq.V - fo.V)/(Mg2SiO4_liq.S - fo.S)*1.e8, 'PW1993 suggest 7.8 K/kbar'


    
    fig1 = mpimg.imread('figures/Presnall_Walter_1993_fo_melting_crop.png')
    plt.imshow(fig1, extent=[0., 21., 1600. + 273.15, 2600. + 273.15], aspect='auto')
    
    pressures = np.linspace(1.e5, 21.e9, 101)
    temperatures = np.empty_like(pressures)
    S_diff = np.empty_like(pressures)
    Tguess = Tm
    for i, P in enumerate(pressures):
        temperatures[i] = burnman.tools.equilibrium_temperature([fo, Mg2SiO4_liq], [1., -1.], P, Tguess)
        Tguess = temperatures[i]
        S_diff[i] = Mg2SiO4_liq.S - fo.S
        print P, temperatures[i], Mg2SiO4_liq.S - fo.S, Mg2SiO4_liq.V - fo.V, Mg2SiO4_liq.gibbs - fo.gibbs
    
    plt.plot(pressures/1.e9, temperatures)
    plt.show()
    
    plt.plot(pressures, S_diff)
    plt.show()

    
# QUARTZ MELTING
if test_SiO2 == True:
    SiO2_liq = DKS_2013_liquids.SiO2_liquid()

    for P in [1.e5]: #[1.e5, 5.e9, 10.e9, 50.e9]:
        if P < 20.e9:
            temperatures = np.linspace(500, 3600. + 5000./10.e9*P, 101)
        else:
            temperatures = np.linspace(500, 3600. + 5000., 101)
        S_liq = np.empty_like(temperatures)
        Cp_liq_additive = np.empty_like(temperatures)
        Cp_liq = np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            print T
            SiO2_liq.set_state(P, T)
            MgO_liq.set_state(P, T)
            Mg2SiO4_liq.set_state(P, T)
            Cp_liq_additive[i] = Mg2SiO4_liq.heat_capacity_p - 2.*MgO_liq.heat_capacity_p
            S_liq[i] = SiO2_liq.S
            Cp_liq[i] = SiO2_liq.heat_capacity_p
            
            
        Tm = 1999. # Richet et al., 1982
        Hm = 8900. # J/mol
        Sm = Hm/Tm
        crst.set_state(1.e5, Tm)
        S_liq_additive = cumtrapz(Cp_liq_additive/temperatures, temperatures, initial=0.)
        tweak = crst.S + Sm - np.interp([Tm],  temperatures, S_liq_additive)
        S_liq_additive += tweak
        
        def linear(x, a, b):
            return a + b*x 
        
        mask=((temperatures > 2200. + 400./10.e9*P) & (temperatures < 3000. + 1000./10.e9*P))
            
        popt, pcov = curve_fit(linear, temperatures[mask], np.array(Cp_liq_additive - Cp_liq)[mask])

        print tweak, popt, pcov
        
        plt.plot(temperatures, linear(temperatures, *popt), label='linear correction')
        plt.plot(temperatures, Cp_liq_additive - Cp_liq, label='Cp excess')
        plt.plot(temperatures, S_liq_additive - S_liq, label='S excess')
        plt.legend(loc='lower left')
    plt.show()

    
    plt.plot(SiO2_data[0], SiO2_data[2])
    
    temperatures = np.linspace(500, 3680, 101)
    #Cp_liq_2 = np.empty_like(temperatures)
    Cp_liq = np.empty_like(temperatures)
    Cp_MgO = np.empty_like(temperatures)
    Cp_Mg2SiO4 = np.empty_like(temperatures)
    Cp_liq_additive = np.empty_like(temperatures)
    Cp_crst = np.empty_like(temperatures)
    Cp_qtz = np.empty_like(temperatures)
    S_liq = np.empty_like(temperatures)
    S_crst = np.empty_like(temperatures)
    rho_crst = np.empty_like(temperatures)
    S_qtz = np.empty_like(temperatures)
    rho_qtz = np.empty_like(temperatures)
    rho_liq = np.empty_like(temperatures)
    
    for i, T in enumerate(temperatures):
        SiO2_liq.set_state(1.e5, T)
        MgO_liq.set_state(1.e5, T)
        Mg2SiO4_liq.set_state(1.e5, T)
        
        
        crst.set_state(1.e5, T)
        qtz.set_state(1.e5, T)
        rho_liq[i] = SiO2_liq.rho
        S_liq[i] = SiO2_liq.S + delta_S(T)[2]
        S_crst[i] = crst.S
        rho_crst[i] = crst.rho
        S_qtz[i] = qtz.S
        rho_qtz[i] = qtz.rho
        Cp_liq_additive[i] = Mg2SiO4_liq.heat_capacity_p - 2.*MgO_liq.heat_capacity_p
        Cp_liq[i] = SiO2_liq.heat_capacity_p + delta_Cp(T)[2]
        Cp_MgO[i] = MgO_liq.heat_capacity_p
        Cp_Mg2SiO4[i] = Mg2SiO4_liq.heat_capacity_p
        Cp_crst[i] = crst.heat_capacity_p
        Cp_qtz[i] = qtz.heat_capacity_p
        
        print 'Cp difference at', T, 'K:', Cp_liq[i] - Cp_qtz[i], Cp_liq_additive[i] - Cp_qtz[i]
        
        #SiO2_liq.set_state(1.e5, T+1.)
        #Cp_liq_2[i] = -T*(S_liq[i] - SiO2_liq.S) # checked, correct

    Pm = 1.e5
    Tm = 1999. # Richet et al., 1982
    Hm = 8900. # J/mol
    Sm = Hm/Tm
    crst.set_state(Pm, Tm)
    plt.plot([Tm], [crst.S + Sm], marker='o', linestyle='None')
    S_liq_additive = cumtrapz(Cp_liq_additive/temperatures, temperatures, initial=0.)
    tweak = crst.S + Sm - np.interp([Tm],  temperatures, S_liq_additive)
    S_liq_additive += tweak

    plt.plot(temperatures, S_liq_additive)
    plt.plot(temperatures, S_liq)
    plt.plot(temperatures, S_crst)
    plt.plot(temperatures, S_qtz)
    #plt.ylim(130., 250)
    plt.show()

    def linear(x, a, b):
        return a + b*x 

    mask=((temperatures > 2000.) & (temperatures < 4000.))

    popt, pcov = curve_fit(linear, temperatures[mask], np.array(S_liq_additive - S_liq)[mask])
    plt.plot(temperatures, linear(temperatures, *popt), label='linear correction')
    
    plt.plot(temperatures, S_liq_additive - S_liq, label='S excess')
    plt.plot(temperatures, Cp_liq_additive - Cp_liq, label='Cp excess')
    plt.legend(loc='lower left')
    plt.show()

    
    # Cp from Richet et al., 2000
    T0 = 1700.
    T1 = 1999.
    dS = 13.4 # J/K/mol between T0 and T1
    Cp = 13.4/np.log(T1/T0)
    plt.plot((T0 + T1)/2., Cp, marker='o', linestyle='None')
    SiO2_interp1 = np.interp(temperatures+1.,  SiO2_data[0], SiO2_data[2])
    SiO2_interp0 = np.interp(temperatures-1.,  SiO2_data[0], SiO2_data[2])
    Cp_SiO2_interp = temperatures*(SiO2_interp1 - SiO2_interp0)/2.
    plt.plot(temperatures, Cp_SiO2_interp)
    plt.plot(temperatures, Cp_liq_additive)
    plt.plot(temperatures, Cp_liq)
    plt.plot(temperatures, Cp_crst)
    plt.plot(temperatures, Cp_qtz)
    plt.plot(temperatures, Cp_MgO)
    plt.plot(temperatures, Cp_Mg2SiO4)
    #plt.plot(temperatures, Cp_liq_2)
    plt.show()

    crst_rhos_obs = np.loadtxt('data/V_beta_cristobalite.dat', unpack=True)
    crst_rhos_obs[1] = 0.06008/(crst_rhos_obs[1]*27.3026e-6)

    plt.plot(crst_rhos_obs[0], crst_rhos_obs[1], label='crst (Bourova and Richet, 1998)')
    rho_SiO2_Bacon_et_al_1960 = lambda T: (2.508 - 2.13e-4*(T - 273.15))*1000.
    rho_mask = ((temperatures>2173.) & (temperatures<2473.))
    plt.plot(temperatures[rho_mask],  rho_SiO2_Bacon_et_al_1960(temperatures)[rho_mask], label='liquid (Bacon et al., 1960)')
    plt.plot(temperatures, rho_crst, label='crst model')
    plt.plot(temperatures, rho_qtz, label='qtz model')
    plt.plot(temperatures, rho_liq, label='liquid model')

    plt.plot([1673., 1973.], [ 0.06008/27.3e-6, 0.06008/27.3e-6], marker='o', linestyle='None', label='liquid (Bottinga and Richet, 1995)')  # Liquid volume from Bottinga and Richet, 1995
    plt.plot([1999.], [ 0.06008/27.4e-6], marker='o', linestyle='None', label='crst (Bourova and Richet, 1998)')  # Cristobalite volume from Bourova and Richet, 1998
    plt.legend(loc='upper left')
    plt.show()


    crst.set_state(1.e5, Tm)
    print 'Vf:', (0.06008/rho_SiO2_Bacon_et_al_1960(Tm) - crst.V)*1.e6, 'cm^3/mol'
    print 'Sf:', Sm, 'J/K/mol'
    Vm = 0.06008/rho_SiO2_Bacon_et_al_1960(Tm) - crst.V
    print 'Melting curve of SiO2:', Vm/Sm*1.e9, 'K/GPa'


    # Make a correction to fit the melting curve of SiO2
    SiO2_liq.set_state(Pm, Tm)
    crst.set_state(Pm, Tm)
    
    delta_S_liq = delta_S(Tm)[2]
    delta_G_liq = crst.gibbs - SiO2_liq.gibbs
    delta_E_liq = crst.gibbs - SiO2_liq.gibbs + delta_S_liq*Tm
    delta_V_liq = 2.25e-7 # -1.e-6
    SiO2_liq.property_modifiers = [['linear', {'delta_E': delta_E_liq, 'delta_S': delta_S_liq, 'delta_V': delta_V_liq}]]
    burnman.Mineral.__init__(SiO2_liq)

    SiO2_liq.set_state(Pm, Tm)
    crst.set_state(Pm, Tm)
    print (SiO2_liq.V - crst.V)*1.e6, 'Bourova and Richet suggest -0.1 cm^3/mol'
    print  SiO2_liq.S - crst.S, 'Bourova and Richet suggest X J/K/mol'

    print (SiO2_liq.V - crst.V)/(SiO2_liq.S - crst.S)*1.e8, 'some small negative number K/kbar'


    
    fig1 = mpimg.imread('figures/hp_sio2_melting.png')
    plt.imshow(fig1, extent=[0., 80., 1500., 5000.], aspect='auto')
    fig2 = mpimg.imread('figures/sio2_melting.png')
    plt.imshow(fig2, extent=[0., 15., 1673., 3273.], aspect='auto')
    
    pressures = np.linspace(1.e5, 100.e9, 201)
    temperatures = np.empty_like(pressures)
    S_diff = np.empty_like(pressures)
    Tguess = Tm
    
    for i, P in enumerate(pressures):
        if P < 1.e9:
            solid = crst
        elif P < 4.e9:
            solid = qtz
        elif P < 13.e9:
            solid = coe
        else:
            solid = stv

        # Only an approximate fix
        dG = ( 0.5*Tguess*(2.*a[0]*np.log(Tguess/1999.) - 2.*a[0] + a[1]*(Tguess - 2.*1999.)) -
               0.5*1999.*(2.*a[0]*np.log(1999./1999.) - 2.*a[0] + a[1]*(-1999.)))
        SiO2_liq.property_modifiers = [['linear', {'delta_E': delta_G_liq + dG + Tguess*delta_S(Tguess)[2], 'delta_S': delta_S(Tguess)[2], 'delta_V': delta_V_liq}]]
        burnman.Mineral.__init__(SiO2_liq)
        
        temperatures[i] = burnman.tools.equilibrium_temperature([solid, SiO2_liq], [1., -1.], P, Tguess)
        Tguess = temperatures[i]
        S_diff[i] = SiO2_liq.S - solid.S
        print P, temperatures[i], SiO2_liq.S - solid.S, SiO2_liq.V - solid.V, SiO2_liq.gibbs - solid.gibbs
    
    plt.plot(pressures/1.e9, temperatures)
    plt.show()
    
    plt.plot(pressures, S_diff)
    plt.show()


    
# MIXING ENTROPIES

if test_mix == True:
    temperatures = np.linspace(500, 3680, 101)
    S_mix_fo = np.empty_like(temperatures)
    S_mix_fo_JANAF = np.empty_like(temperatures)
    S_per = np.empty_like(temperatures)
    S_fo = np.empty_like(temperatures)
    S_qtz = np.empty_like(temperatures)



    Mg2SiO4_interp = np.interp(temperatures,  Mg2SiO4_data[0], Mg2SiO4_data[2])
    MgO_interp = np.interp(temperatures,  MgO_data[0], MgO_data[2])
    SiO2_interp = np.interp(temperatures,  SiO2_data[0], SiO2_data[2])
    S_mix_fo_JANAF = Mg2SiO4_interp/3. - (2./3.*MgO_interp + 1./3.*SiO2_interp)

    
    for P in [1.e5, 20.e9, 100.e9]:
        if P < 10.e9:
            temperatures = np.linspace(500, 3680, 101)
        else:
            temperatures = np.linspace(500, 3680, 101)
        for i, T in enumerate(temperatures):
            S = []
            for j, m in enumerate(mins):
                m.set_state(P, T)
                S.append(m.S + delta_S(T)[j])

            if P < 0.1e9:
                S_per[i] = S[0]
                S_fo[i] = S[1]
                S_qtz[i] = S[2]
                print S_fo[i], Mg2SiO4_interp[i]
            
            S_mix_fo[i] = S[1]/3. - (2./3.*S[0] + 1./3.*S[2])
        plt.plot(temperatures, S_mix_fo, label=str(P/1.e9)+' GPa')



    

    
    mask = (S_mix_fo_JANAF > -100)
    plt.plot(temperatures[mask], S_mix_fo_JANAF[mask], label='JANAF prediction of mixing entropy')

    
    #plt.plot(temperatures[mask], S_fo[mask] - Mg2SiO4_interp[mask], label='fo entropy (model-JANAF)')
    #plt.plot(temperatures[mask], S_per[mask] - MgO_interp[mask], label='per entropy (model-JANAF)')
    #plt.plot(temperatures[mask], S_qtz[mask] - SiO2_interp[mask], label='qtz entropy (model-JANAF)')
    plt.legend(loc='lower right')
    plt.show()
    
