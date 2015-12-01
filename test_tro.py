import burnman
import os, sys, numpy as np, matplotlib.pyplot as plt
tro = burnman.minerals.HP_2011_ds62.tro()
tro2 = burnman.minerals.HP_2011_ds62.tro2()

P = 1.e9
temperatures = np.linspace(100., 1000., 101)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
volumes = np.empty_like(temperatures)
Ss2 = np.empty_like(temperatures)
Cps2 = np.empty_like(temperatures)
volumes2 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    tro.set_state(P, T)
    Ss[i] = tro.S
    Cps[i] = tro.C_p
    volumes[i] = tro.V
    tro2.set_state(P, T)
    Ss2[i] = tro2.S
    Cps2[i] = tro2.C_p
    volumes2[i] = tro2.V
    
    K_T2 = tro2.K_T
    alpha2 = tro2.alpha

    dT = 0.001
    tro2.set_state(P, T+dT)    
    G1 = tro2.gibbs
    S1 = tro2.S
    V1 = tro2.V
    tro2.set_state(P, T-dT)    
    G0 = tro2.gibbs
    S0 = tro2.S
    V0 = tro2.V
    S = -(G1 - G0)/(2.*dT)
    #print (Ss2[i]-S)/S, 'good'
    Cp = T*(S1 - S0)/(2.*dT)
    #print (Cps2[i] - Cp)/Cp, 'not ok'
    alpha = (V1 - V0)/(2.*dT)/volumes2[i]
    #print (alpha - alpha2)/alpha, 'not ok'
    dP = 1000.
    tro2.set_state(P+dP, T)    
    G1 = tro2.gibbs
    V1 = tro2.V
    tro2.set_state(P-dP, T)    
    G0 = tro2.gibbs
    V0 = tro2.V
    V = (G1 - G0)/(2.*dP)
    K_T = -(2.*dP)/(V1 - V0) * volumes2[i]
    #print (K_T-K_T2)/K_T, 'not ok'
    #print (volumes2[i] - V)/V, 'ok'


    
    
plt.plot(temperatures, Ss, label='SLB')
plt.plot(temperatures, Ss2, label='HP')
plt.title('Entropies')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, Cps, label='SLB')
plt.plot(temperatures, Cps2, label='HP')
plt.title('Heat capacities')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, volumes, label='SLB')
plt.plot(temperatures, volumes2, label='HP')
plt.title('Volumes')
plt.legend(loc='lower right')
plt.show()
