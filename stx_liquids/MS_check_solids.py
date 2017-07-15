import numpy as np
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt


import os, sys
sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import \
    DKS_2013_liquids, \
    DKS_2013_solids, \
    SLB_2011, \
    HP_2011_ds62
from burnman import constants
import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = 8.31446

def enthalpy(T, prm):
    return prm.A + ( prm.a[0]*( T - 298.15 ) +
                     0.5  * prm.a[1]*( T*T - 298.15*298.15 ) +
                     -1.0 * prm.a[2]*( 1./T - 1./298.15 ) +
                     2.0  * prm.a[3]*(np.sqrt(T) - np.sqrt(298.15) ) +
                     -0.5 * prm.a[4]*(1./(T*T) - 1./(298.15*298.15) ) )

def entropy(T, prm):
    return prm.B + ( prm.a[0]*(np.log(T/298.15)) +
                     prm.a[1]*(T - 298.15) +
                     -0.5 * prm.a[2]*(1./(T*T) - 1./(298.15*298.15)) +
                     -2.0 * prm.a[3]*(1./np.sqrt(T) - 1./np.sqrt(298.15)) +
                     -1./3. * prm.a[4]*(1./(T*T*T) - 1./(298.15*298.15*298.15) ) )

class params():
    def __init__(self):
        self.A = 0.
        self.B = 0.
        self.a = [0., 0., 0., 0., 0.]

    
def G_SiO2_liquid(T):
    prm = params()
    if T < 1996.:
        prm.A = -214339.36
        prm.B = 12.148448
        prm.a = [19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8]
    else:
        prm.A = -221471.21
        prm.B = 2.3702523
        prm.a = [20.50000, 0., 0., 0., 0.]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))

def G_crst(T):
    prm = params()
    prm.A = -216629.36
    prm.B = 11.001147
    prm.a = [19.960229, 0.e-3, -5.8684512e5, -89.553776, 0.66938861e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))

def G_MgO_liquid(T):
    prm = params()
    prm.A = -130340.58
    prm.B = 6.4541207
    prm.a = [17.398557, -0.751e-3, 1.2494063e5, -70.793260, 0.013968958e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))

def G_per(T):
    prm = params()
    prm.A = -143761.95
    prm.B = 6.4415388
    prm.a = [14.605557, 0.e-3, -1.4845937e5, -70.793260, 0.013968958e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))


def G_fo(T):
    prm = params()
    prm.A = -520482.62
    prm.B = 22.468913
    prm.a = [57.036654, 0.e-3, 0.e5, -478.31286, -0.27782811e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))/3. # (1 cation basis)

def G_en(T):
    prm = params()
    prm.A = -368906.91
    prm.B = 16.117975
    prm.a = [39.813456, 0.e-3, -5.426786e5, -286.94742, 0.66718532e8]

    return 4.184*(enthalpy(T, prm) - T*entropy(T, prm))/2. # (1 cation basis)






DKS_per = DKS_2013_solids.periclase()
SLB_per = SLB_2011.periclase()
DKS_pv = DKS_2013_solids.perovskite()
SLB_pv = SLB_2011.mg_perovskite()
DKS_stv = DKS_2013_solids.stishovite()
SLB_stv = SLB_2011.stishovite()



mmr = [['per', DKS_per, SLB_per],
       ['pv', DKS_pv, SLB_pv],
       ['stv', DKS_stv, SLB_stv]]

P = 60.e9
temperatures = np.linspace(100., 5000., 101)
G_diff = np.empty_like(temperatures)
G_diff2 = np.empty_like(temperatures)
for (n, m1, m2) in mmr:
    m1.set_state(1.e5, 300.)
    m2.set_state(1.e5, 300.)
    m2.property_modifiers = [['linear', {'delta_E': m1.gibbs - m2.gibbs,
                                         'delta_S': 0.,
                                         'delta_V': 0.}]]
    for i, T in enumerate(temperatures):
        print T
        m1.set_state(P, T)
        m2.set_state(P, T)
        
        G_diff[i] = m1.gibbs
        G_diff2[i] = m2.gibbs

        
        G_diff[i] = m1.S
        G_diff2[i] = m2.S

    plt.plot(temperatures, G_diff, label='DKS')
    plt.plot(temperatures, G_diff2, label='SLB')
    plt.title(n)
    plt.legend(loc='lower right')
    plt.show()




temperatures = np.linspace(100., 2200., 101)
G_diff = np.empty_like(temperatures)
G_diff2 = np.empty_like(temperatures)

HP_fo = HP_2011_ds62.fo()
SLB_fo = SLB_2011.forsterite()
HP_per = HP_2011_ds62.per()
SLB_per = SLB_2011.periclase()
HP_en = HP_2011_ds62.en()
SLB_en = SLB_2011.enstatite()
HP_crst = HP_2011_ds62.crst()

SLB_per.params['F_0'] += -40036.738
SLB_per.params['q_0'] = 0.15
SLB_per.params['grueneisen_0'] = 1.40


#SLB_en.params['F_0'] += -40036.738
#SLB_en.params['Debye_0'] = 1400.
#SLB_en.params['q_0'] = 0.0
#SLB_en.property_modifiers = [['linear', {'delta_E': 0., 'delta_S': 16., 'delta_V': 0.}]]

#SLB_fo.params['F_0'] += -40036.738
#SLB_fo.params['Debye_0'] = 1400.
#SLB_fo.params['q_0'] = 0.
#SLB_fo.property_modifiers = [['linear', {'delta_E': 0., 'delta_S': 20., 'delta_V': 0.}]]

mmr = [['fo', HP_fo, SLB_fo, G_fo, 3.],
       ['per', HP_per, SLB_per, G_per, 1.],
       ['en', HP_en, SLB_en, G_en, 4.],
       ['crst', HP_crst, HP_crst, G_crst, 1.]]

for (n, m1, m2, reaction, f) in mmr:
    m1.set_state(1.e5, 300.)
    m2.set_state(1.e5, 300.)
    m2.property_modifiers = [['linear', {'delta_E': m1.gibbs - m2.gibbs,
                                         'delta_S': 0.,
                                         'delta_V': 0.}]]
    for i, T in enumerate(temperatures):
        print T
        m1.set_state(1.e5, T)
        m2.set_state(1.e5, T)
        
        G_diff[i] = m1.gibbs - reaction(T)*f
        G_diff2[i] = m2.gibbs - reaction(T)*f

        
        G_diff[i] = m1.heat_capacity_p
        G_diff2[i] = m2.heat_capacity_p

    MgO_data = np.loadtxt(fname='data/JANAF_MgO_liq.dat', unpack=True)
    plt.plot(MgO_data[0], MgO_data[1])
    plt.plot(temperatures, G_diff, label='HP')
    plt.plot(temperatures, G_diff2, label='SLB')
    plt.title(n)
    plt.legend(loc='lower right')
    plt.show()


fo = HP_2011_ds62.fo()
per=HP_2011_ds62.per()
crst=HP_2011_ds62.crst()


for i, T in enumerate(temperatures):
    print T
    fo.set_state(1.e5, T)
    per.set_state(1.e5, T)
    crst.set_state(1.e5, T)
    
    G_diff[i] = 2.*per.gibbs + crst.gibbs - fo.gibbs
    G_diff2[i] = 2.*G_per(T) + G_crst(T) - G_fo(T)*3.

plt.plot(temperatures, G_diff)
plt.plot(temperatures, G_diff2)
plt.title('fo')
plt.show()




