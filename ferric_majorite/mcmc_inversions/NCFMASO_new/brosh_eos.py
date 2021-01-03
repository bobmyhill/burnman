import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman
from burnman import Mineral, SolidSolution
from burnman.processchemistry import formula_mass


dummy = burnman.minerals.HP_2011_ds62.per()

class FeNi3C(SolidSolution):

    def __init__(self, molar_fractions=None):
        self.name = '(Fe,Ni)3C'
        self.solution_type = 'symmetric'
        self.endmembers = [[dummy, '[Fe]3C'], [dummy, '[Ni]3C']]
        self.energy_interaction = [[29.4e3]]

        SolidSolution.__init__(self, molar_fractions=molar_fractions)
FeNi_cementite = FeNi3C()

# cementite parameters
def gibbs_P0_Fe7C(temperatures):
    try:
        a=temperatures[0]
        tmp = temperatures
    except:
        tmp = [temperatures]
    G = []
    for T in tmp:
        if T < 103:
            G_CC = -1049.14084-.09009204*T-2.75E-05*T**3
        elif T < 350.:
            G_CC = -988.25091-7.39898691*T+1.76583*T*np.log(T)-.01706952*T**2
        elif T < 6000.:
            G_CC = (-17368.441+170.73*T-24.3*T*np.log(T)-4.723E-04*T**2+2562600*T**(-1)
                    -2.643E+08*T**(-2)+1.2E+10*T**(-3))

        if T < 43.:
            G_Fe3C = +11369.9377-5.64125926*T-8.333E-06*T**4
        elif T < 163.:
            G_Fe3C = +11622.6472-59.5377093*T+15.74232*T*np.log(T)-.27565*T**2
        elif T < 6000.:
            G_Fe3C = -10195.8608+690.949888*T-118.47637*T*np.log(T)-7E-04*T**2+590527*T**(-1)
        G.append(7./3.*G_Fe3C + 2./3.*G_CC)
    return np.array(G)


# cementite parameters
def gibbs_P0_cementite(temperatures):
    try:
        a=temperatures[0]
        tmp = temperatures
    except:
        tmp = [temperatures]
    G = []
    for T in tmp:
        if T < 43.:
            G.append(+11369.9377-5.64125926*T-8.333E-06*T**4)
        elif T < 163.:
            G.append(+11622.6472-59.5377093*T+15.74232*T*np.log(T)-.27565*T**2)
        elif T < 6000.:
            G.append(-10195.8608+690.949888*T-118.47637*T*np.log(T)-7E-04*T**2+590527*T**(-1))
    return np.array(G)

# cementite parameters
def gibbs_P0_diamond(temperatures):
    try:
        a=temperatures[0]
        tmp = temperatures
    except:
        tmp = [temperatures]
    G = []
    for T in tmp:
        if T < 78.:
            G.append(+1366.36944+.0375684493*T-1.795E-08*T**4)
        elif T < 330.:
            G.append(+3386.8147-64.4266811*T+11.71699*T*np.log(T)-.0304*T**2+3.3333E-06*T**3
                     -91196.5*T**(-1)+2797170*T**(-2)-38820740*T**(-3))
        elif T < 6000.:
            G.append(-16359.441+175.61*T-24.31*T*np.log(T)-4.723E-04*T**2+2698000*T**(-1)
                     -2.61E+08*T**(-2)+1.11E+10*T**(-3))
    return np.array(G)

cementite_params = {'V_0': 5.755E-06*4.,
          'K_0': 2e+11,
          'Kprime_0': 5.2,
          'theta_0': 400,
          'grueneisen_0': 1.7,
          'delta': [5., 10.],
          'b': [1., 4.],
          'n': 4.,
          'HS': 11369.9377,
          'SS': 5.64125926,
          'gibbs_P0': gibbs_P0_cementite}


diamond_params = {'V_0': 3.4145e-6,
          'K_0': 447.e9,
          'Kprime_0': 3.5,
          'theta_0': 1650.,
          'grueneisen_0': 0.93,
          'delta': [5., 5.],
          'b': [1., 10.],
          'n': 1.,
          'HS': 1366.36944,
          'SS': -.0375684493,
          'gibbs_P0': gibbs_P0_diamond}


Fe7C3_params = {'V_0': 5.515e-6*10.,
                'K_0': 2.55e11,
                'Kprime_0': 4.,
                'theta_0': 445.,
                'grueneisen_0': 1.7,
                'delta': [4., 10.],
                'b': [1., 4.],
                'n': 10.,
                'HS': 7./3.*11369.9377 + 2./3.*-1049.14084,
                'SS': 7./3.*5.64125926 + 2./3.*.09009204,
                'gibbs_P0': gibbs_P0_Fe7C}




def Tc_cementite(pressures):
    AM1 = 1. - 1e-10*pressures
    AM2 = AM1*AM1 + 1e-4
    AM3 = np.log(AM2)
    AM4 = 0.5*(AM1 + np.exp(0.5*AM3))
    AM5 = 0.75*np.log(AM4)
    return 485.*np.exp(AM5)

pressures = np.linspace(0., 20.e9, 101)
plt.plot(pressures/1.e9, Tc_cementite(pressures))
plt.show()


def gibbs(P, T, params):
    # Implementation of the gibbs free energy
    # copied from TDB file from Fei and Brosh, 2014
    Pf  = P/params['K_0']
    BV  = params['V_0']*params['K_0']/params['n']
    IA  = params['Kprime_0'] - 1./3.
    FC  = 1. + 4./3.*IA*Pf
    LFC = np.log(FC)
    IXC = 1. - (1. - np.exp(0.25*LFC))/IA

    M = np.array([[1.5, -6., 8., -32./9.],
                  [-9., 27., -24., 16./3.],
                  [9., -18., 9., -4./3.],
                  [3., -3., 1., -1./9.]])
    G2, G1, GL, GM1 = M.dot([np.power(params['Kprime_0'], 3.),
                             np.power(params['Kprime_0'], 2.),
                             params['Kprime_0'],
                             1.])
    #G2  = 1.5*np.power(params['Kprime_0'], 3.) - 6.*np.power(params['Kprime_0'], 2.) + 8.*params['Kprime_0'] - 3.55555555
    #G1  = -9*np.power(params['Kprime_0'], 3.) + 27.*np.power(params['Kprime_0'], 2.) - 24.*params['Kprime_0'] + 5.33333333
    #GL  = 9.*np.power(params['Kprime_0'], 3.) - 18.*np.power(params['Kprime_0'], 2.) + 9.*params['Kprime_0'] - 1.33333333
    #GM1 = 3.*np.power(params['Kprime_0'], 3.) - 3.*np.power(params['Kprime_0'], 2.) + params['Kprime_0'] - .111111111
    GPC = G2/(IXC*IXC) + G1/IXC - GL*np.log(IXC) + GM1*IXC - G2 - G1 - GM1
    PT  = Pf*(1. + params['delta'][0])
    IAT = 3.*params['b'][0]-1.
    FT  = 1. + 2./3.*IAT*PT
    LTF = np.log(FT)
    IXT = 1. - (1. - np.exp(0.5*LTF))/IAT
    GPT = ((4.5*params['b'][0] - 3.)/(IXT*IXT) +
           (- 9.*params['b'][0] + 3.)/IXT
           + 4.5*params['b'][0])
    PT2 = Pf*(1. + params['delta'][1])
    IY  = 1. + 2.*params['b'][1]*PT2
    LY  = 0.5*np.log(IY)
    Y   = np.exp(LY)
    IB  = (1.-np.exp(LY))/params['b'][1]
    GBPf = 1. + params['b'][1]*(1. - np.exp(IB)) -Y*np.exp(IB)
    GBM = 1. + params['b'][1]
    GBR = GBPf/GBM
    IGR = (params['delta'][0]+1.)/params['grueneisen_0']
    INT = GPT/IGR
    Tf   = params['theta_0']*np.exp(INT)
    IE  = 1. - np.exp(-Tf/T)
    IE0 = 1. - np.exp(-params['theta_0']/T)

    HSE = params['HS']/params['n']
    SSE = params['SS']/params['n']
    G_P0 = params['gibbs_P0'](T)
    if len(G_P0) == 1.:
        G_P0 = G_P0[0]
    R = 8.31446
    DGT = -HSE + SSE*T + G_P0/params['n'] - 3.*R*T*np.log(IE0)

    G = G_P0 + params['n']*(BV*GPC + 3.*R*T*np.log(IE) - 3.*R*T*np.log(IE0) - DGT*GBR)
    return G

cem_cp_img = mpimg.imread('cementite_cp_dick_2011.png')
plt.imshow(cem_cp_img, extent=[0., 1500., 0., 4.5*8.31446*4.], aspect='auto', alpha=0.3)

temperatures = np.linspace(0.0001, 2000., 1001)
for P in [1.e5, 1.e9, 1.e10]:
    S = -np.gradient(gibbs(P, temperatures, Fe7C3_params), temperatures, edge_order=2)
    Cp = temperatures*np.gradient(S, temperatures, edge_order=2)
    plt.plot(temperatures, Cp)
plt.show()

cem_V_img = mpimg.imread('cementite_PVT_Litasov_2013.png')

#f = 6.023e23 / 1.e30 / 4.
#plt.imshow(cem_V_img, extent=[0., 35., 135.*f, 165.*f], aspect='auto', alpha=0.3)

fcc_iron = burnman.minerals.SE_2015.fcc_iron()
hcp_iron = burnman.minerals.SE_2015.hcp_iron()

fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

pressures = np.linspace(6.e9, 35.e9, 1001)
#for T in np.linspace(300., 1500., 7):
for T in np.linspace(1500., 2100., 4):
    V_Fe7C3 = np.gradient(gibbs(pressures, T, Fe7C3_params), pressures, edge_order=2)
    V_Fe3C = np.gradient(gibbs(pressures, T, cementite_params), pressures, edge_order=2)
    V_C = np.gradient(gibbs(pressures, T, diamond_params), pressures, edge_order=2)
    #plt.plot(pressures/1.e9, (V_Fe7C3 - 3.*V_C)/7.)


    S_Fe7C3 = gibbs(pressures, T-0.5, Fe7C3_params)-gibbs(pressures, T+0.5, Fe7C3_params)
    S_Fe3C = gibbs(pressures, T-0.5, cementite_params)-gibbs(pressures, T+0.5, cementite_params)
    S_C = gibbs(pressures, T-0.5, diamond_params)-gibbs(pressures, T+0.5, diamond_params)

    G_Fe7C3 = gibbs(pressures, T, Fe7C3_params)
    G_Fe3C = gibbs(pressures, T, cementite_params)
    G_C = gibbs(pressures, T, diamond_params)

    gibbs_Fe_equiv = (G_Fe7C3 - 3.*G_C)/7.
    S_Fe_equiv = (S_Fe7C3 - 3.*S_C)/7.
    V_Fe_equiv = (V_Fe7C3 - 3.*V_C)/7.

    E_Fe_equiv = gibbs_Fe_equiv + S_Fe_equiv*T - pressures*V_Fe_equiv

    temperatures = T + 0.*pressures
    G_Fe, E_Fe, S_Fe, V_Fe = fcc_iron.evaluate(['gibbs', 'molar_internal_energy', 'S', 'V'], pressures, temperatures)

    ax[0].plot(pressures/1.e9, gibbs_Fe_equiv - G_Fe, label='{0} K'.format(T))
    ax[1].plot(pressures/1.e9, E_Fe_equiv - E_Fe, label='{0} K'.format(T))
    ax[2].plot(pressures/1.e9, S_Fe_equiv - S_Fe, label='{0} K'.format(T))
    ax[3].plot(pressures/1.e9, V_Fe_equiv - V_Fe, label='{0} K'.format(T))

    FeNi_cementite.set_composition([0.76, 0.24]) # saturation at 10 GPa
    FeNi_cementite.set_state(10.e9, 1700.)

    V_Fe3C = V_Fe3C
    S_Fe3C = S_Fe3C + FeNi_cementite.excess_partial_entropies[0]
    G_Fe3C = G_Fe3C + FeNi_cementite.excess_partial_gibbs[0]

    gibbs_Fe_equiv = (G_Fe3C - G_C)/3.
    S_Fe_equiv = (S_Fe3C - S_C)/3.
    V_Fe_equiv = (V_Fe3C - V_C)/3.

    E_Fe_equiv = gibbs_Fe_equiv + S_Fe_equiv*T - pressures*V_Fe_equiv


    ax[0].plot(pressures/1.e9, gibbs_Fe_equiv - G_Fe, linestyle=':', label='{0} K'.format(T))
    ax[1].plot(pressures/1.e9, E_Fe_equiv - E_Fe, linestyle=':', label='{0} K'.format(T))
    ax[2].plot(pressures/1.e9, S_Fe_equiv - S_Fe, linestyle=':', label='{0} K'.format(T))
    ax[3].plot(pressures/1.e9, V_Fe_equiv - V_Fe, linestyle=':', label='{0} K'.format(T))








for i in range(4):
    ax[i].legend()
ax[1].set_ylim(0., 5.e3)
ax[2].set_ylim(0., 10.)
ax[3].set_ylim(-5e-7, 0.)
plt.show()
