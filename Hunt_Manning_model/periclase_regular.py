#!/usr/python
import os, sys
import numpy as np

from scipy.optimize import fsolve, minimize, fmin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R=8.31446

def non_ideal_contributions(proportions, Ws):
    W_MH, W_MB, W_HB = Ws
    XMgO, Xbr, XH2O = proportions
    RTlnyMgO = (1.-XMgO)*XH2O*W_MH + (1.-XMgO)*Xbr*W_MB - XH2O*Xbr*W_HB
    RTlnyH2O = XMgO*(1.-XH2O)*W_MH - XMgO*Xbr*W_MB + (1.-XH2O)*Xbr*W_HB
    RTlnybr = -XMgO*XH2O*W_MH + XMgO*(1.-Xbr)*W_MB + XH2O*(1.-Xbr)*W_HB
    return RTlnyMgO, RTlnybr, RTlnyH2O 

def equilibrium_order(Xbr, XH2Otot, T, deltaGbrmelt, Ws):

    XH2O=XH2Otot*(1.+Xbr)-Xbr
    XMgO=1.-Xbr-XH2O

    RTlnyMgO, RTlnybr, RTlnyH2O = non_ideal_contributions([XMgO, Xbr, XH2O], Ws)

    return deltaGbrmelt + R*T*np.log(Xbr/(XH2O*XMgO)) - RTlnyMgO + RTlnybr - RTlnyH2O

def gibbs_excess(Xbr, XH2Otot, T, deltaGbrmelt, Ws):
    XH2O=XH2Otot*(1.+Xbr)-Xbr
    XMgO=1.-Xbr-XH2O

    RTlnyMgO, RTlnybr, RTlnyH2O = non_ideal_contributions([XMgO, Xbr, XH2O], Ws)

    ideal=Xbr*deltaGbrmelt + R*T*(Xbr*np.log(Xbr) + XH2O*np.log(XH2O) + XMgO*np.log(XMgO))
    nonideal=Xbr*RTlnybr + XH2O*RTlnyH2O + XMgO*RTlnyMgO 

    return ideal+nonideal

def eqm_gibbs_excess(XH2Otot, T, deltaGbrmelt, Ws):
    Xbr=fsolve(equilibrium_order, 0.01, args=(XH2Otot, T, deltaGbrmelt, Ws))
    return gibbs_excess(Xbr, XH2Otot, T, deltaGbrmelt, Ws)

def partial_excesses(XH2Otot, T, deltaGbrmelt, Ws):
    delta=0.0001

    Xbr=fsolve(equilibrium_order, 0.00001, args=(XH2Otot-delta, T, deltaGbrmelt, Ws))
    gibbs0=gibbs_excess(Xbr, XH2Otot-delta, T, deltaGbrmelt, Ws)

    Xbr=fsolve(equilibrium_order, 0.00001, args=(XH2Otot+delta, T, deltaGbrmelt, Ws))
    gibbs1=gibbs_excess(Xbr, XH2Otot+delta, T, deltaGbrmelt, Ws)

    return gibbs0 - XH2Otot*(gibbs1-gibbs0)/(2.*delta), gibbs0 + (1.-XH2Otot)*(gibbs1-gibbs0)/(2.*delta)

def per_eqm(XH2Otot, T, deltaGbrmelt, Ws, dGper):
    return dGper - partial_excesses(XH2Otot, T, deltaGbrmelt, Ws)[0]

def br_eqm(XH2Otot, T, deltaGbrmelt, Ws, dGbr):
    return dGbr - 0.5*(partial_excesses(XH2Otot, T, deltaGbrmelt, Ws)[0] + partial_excesses(XH2Otot, T, deltaGbrmelt, Ws)[1])

def per_br_eqm(data, deltaGbrmelt, Ws, dGbr, dGper): # fit according to deltaGbrmelt and Ws[2]
    XH2Otot, T=data
    per_eqm=dGper(T) - partial_excesses(XH2Otot, T, deltaGbrmelt(T), Ws(T))[0]
    br_eqm=dGbr(T) - 0.5*(partial_excesses(XH2Otot, T, deltaGbrmelt(T), Ws(T))[0] + partial_excesses(XH2Otot, T, deltaGbrmelt(T), Ws(T))[1])
    return [per_eqm[0], br_eqm[0]]

######################
# FITTING PARAMETERS #
######################
deltaGbrmelt=lambda T: -41000. - 10*(T-(1210.+273.15))
Ws=lambda T: [0000., -10000., -22000.] # W_MH, W_MB, W_HB
Smelt=22.
Tmelt=5373.
######################

dGper=lambda T: Smelt*(T-Tmelt)
dGbr=lambda T: 0.5*(Smelt*(T-Tmelt)) +0.5*(- 8330. + 20.*(T-1473.15))



Xbr=0.655 # composition of fluid in eqm with Xbr
Tbr=1210.+273.15 # K



dGH2O_HP=1000.*((-829.82 - -547.59)- -274.10)

#XH2O_HP=np.exp(dGH2O_HP/(R*Tbr))

compositions=np.linspace(0.0001, 0.99, 101)
Gex=np.empty_like(compositions)
Gex_2=np.empty_like(compositions)
for i, X in enumerate(compositions):
    Gex[i]=eqm_gibbs_excess(X, Tbr, deltaGbrmelt(Tbr), Ws(Tbr))
    Gex_2[i]=eqm_gibbs_excess(X, 5000, deltaGbrmelt(Tbr), Ws(Tbr))


plt.plot( compositions, Gex, '-', linewidth=2., label='T=Tbr')
plt.plot( compositions, Gex_2, '-', linewidth=2., label='T=5000 K')
#plt.plot ( 0.0, Smelt*(Tbr-Tmelt), marker='o', label='model per')
plt.plot ( [1.0], [dGH2O_HP], marker='o', label='HP H2O activity')
plt.plot ( [0.0, 1.0], partial_excesses(Xbr, Tbr, deltaGbrmelt(Tbr), Ws(Tbr)), marker='o', label='model H2O activity')
plt.ylabel("Excess Gibbs (J/mol)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()




Xperbr, Tbr = fsolve(per_br_eqm, [0.64, 1210.+273.15], args=(deltaGbrmelt, Ws, dGbr, dGper))



per_temperatures=np.linspace(Tbr, 5273., 101)
per_liquidus=np.empty_like(per_temperatures)
br_temperatures=np.linspace(1373.15, Tbr, 21)
br_liquidus=np.empty_like(br_temperatures)
for i, T in enumerate(per_temperatures):
    per_liquidus[i]=fsolve(per_eqm, 0.6, args=(T, deltaGbrmelt(T), Ws(T), dGper(T)))
    print per_liquidus[i], T

for i, T in enumerate(br_temperatures):
    guess=0.7-0.05*float(i)/float(len(br_temperatures))
    br_liquidus[i]=fsolve(br_eqm, guess, args=(T, deltaGbrmelt(T), Ws(T), dGbr(T)), xtol=1.e-12, factor=.1)
    diff=guess - br_liquidus[i]
    if np.abs(diff) <  0.001:
        br_liquidus[i]=float('nan')
    else:
        print br_liquidus[i], T

plt.plot( per_liquidus, per_temperatures, 'r-', linewidth=2., label='per')
plt.plot( br_liquidus, br_temperatures, 'r-', linewidth=2., label='br')
plt.plot( [0., Xperbr], [Tbr, Tbr], 'r-', linewidth=2., label='br=per+melt')
plt.plot( [0.5, 0.5], [1000., Tbr], 'r-', linewidth=2., label='br')

periclase=[]
brucite=[]
liquid=[]
for line in open('../figures/13GPa_per-H2O.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[2] == 'p' or content[2] == 'sp':
            periclase.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'l':
            liquid.append([float(content[0])+273.15, float(content[1])/100.])
        if content[2] == 'b':
            brucite.append([float(content[0])+273.15, float(content[1])/100.])

periclase=zip(*periclase)
brucite=zip(*brucite)
liquid=zip(*liquid)
plt.plot( periclase[1], periclase[0], marker='.', linestyle='none', label='per+liquid')
plt.plot( brucite[1], brucite[0], marker='.', linestyle='none', label='br+liquid')
plt.plot( liquid[1], liquid[0], marker='.', linestyle='none', label='superliquidus')



plt.xlim(0.,1.)
plt.ylabel("Temperature (K)")
plt.xlabel("X")
plt.legend(loc='lower left')
plt.show()
