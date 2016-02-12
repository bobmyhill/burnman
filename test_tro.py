import burnman
import os, sys, numpy as np, matplotlib.pyplot as plt
tro = burnman.minerals.HP_2011_ds62.tro()
tro2 = burnman.minerals.HP_2011_ds62.tro2()

P = 1.e10
temperatures = np.linspace(300., 1400., 101)
Ss = np.empty_like(temperatures)
Cps = np.empty_like(temperatures)
K_Ts = np.empty_like(temperatures)
volumes = np.empty_like(temperatures)
Ss2 = np.empty_like(temperatures)
Cps2 = np.empty_like(temperatures)
volumes2 = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    tro.set_state(P, T)
    Ss[i] = tro.S
    Cps[i] = tro.C_p
    volumes[i] = tro.V
    K_Ts[i] = tro.K_T

    tro2.set_state(P, T)
    Ss2[i] = tro2.S
    Cps2[i] = tro2.C_p
    volumes2[i] = tro2.V
    
    K_T2 = tro.K_T
    alpha2 = tro.alpha

    dT = 0.001
    tro.set_state(P, T+dT)    
    G1 = tro.gibbs
    S1 = tro.S
    V1 = tro.V
    tro.set_state(P, T-dT)    
    G0 = tro.gibbs
    S0 = tro.S
    V0 = tro.V
    S = -(G1 - G0)/(2.*dT)
    #print Ss[i], S, (Ss[i]-S)/S, 'good'
    Cp = T*(S1 - S0)/(2.*dT)
    #print Cps[i], Cp, (Cps[i] - Cp)/Cp, 'good'
    alpha = (V1 - V0)/(2.*dT)/volumes[i]
    #print (alpha - alpha2)/alpha, 'good'
    dP = 1000.
    tro.set_state(P+dP, T)    
    G1 = tro.gibbs
    V1 = tro.V
    tro.set_state(P-dP, T)    
    G0 = tro.gibbs
    V0 = tro.V
    V = (G1 - G0)/(2.*dP)
    K_T = -(2.*dP)/(V1 - V0) * volumes[i]
    #print (K_T - K_T2)/K_T, 'good'
    #print (volumes[i] - V)/V, 'good'


    
    
plt.plot(temperatures, Ss, label='SLB')
plt.plot(temperatures, Ss2, label='HP')
plt.title('Entropies')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, Cps, label='SLB')
plt.plot(temperatures, Cps2, label='HP')
plt.title('Heat capacities')
plt.legend(loc='lower right')
plt.ylim(0., 100.)
plt.show()

plt.plot(temperatures, volumes, label='SLB')
plt.plot(temperatures, volumes2, label='HP')
plt.title('Volumes')
plt.legend(loc='lower right')
plt.show()

plt.plot(temperatures, K_Ts/1.e9, label='SLB')
plt.title('K_T')
plt.legend(loc='lower right')
plt.show()

T = 800.
pressures = np.linspace(1.e9, 10.e9, 101)
K_Ts = np.empty_like(pressures)
for i, P in enumerate(pressures):
    tro.set_state(P, T)
    K_Ts[i] = tro.K_T

plt.plot(pressures/1.e9, K_Ts, label='HP')
plt.title('K_T')
plt.legend(loc='lower right')
plt.show()
