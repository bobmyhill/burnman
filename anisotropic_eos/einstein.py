import numpy as np
from scipy.special import spence
import matplotlib.pyplot as plt
import os, sys
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
from burnman.minerals import HP_2011_ds62, SLB_2011

def molar_heat_capacity_p(T, theta, a, b):
    u = theta/T
    Cp = a*(u*u*np.exp(u)/np.power(np.exp(u) - 1., 2.))
    Cp += b*(-1.-1/(2.*u)-1./(u*(np.exp(-1./u) - 1)))
    return Cp

def entropy(T, theta, a, b):
    u = theta/T
    S = a*((u*(1. + 1./(np.exp(u) - 1.))
             - np.log(np.exp(u) - 1.)))
    S += b*(1./(2.*u) + np.log(1. - np.exp(-1./u)) + np.log(u))
    return S

def gibbs(T, theta, a, b):
    u = theta/T
    G = 0.
    G += -a*(T*(np.log(np.exp(u) - 1.) - u))
    G += b*(T*(np.real((1. - 1./(4.*u) + np.log(u + 0j)) - u*spence(1. + 0j - np.exp(1/u))) + np.pi*np.pi/6.*u))
    return G

temperatures = np.linspace(300., 1900., 101)
plt.plot(temperatures, molar_heat_capacity_p(temperatures, theta=700.*0.806, a=3.*2.*8.31446, b=8.))
#plt.plot(temperatures, temperatures*np.gradient(entropy(temperatures, theta=760., a=0., b=-1), temperatures))
#plt.plot(temperatures, molar_heat_capacity_p(temperatures, theta=760., a=0., b=-1), linestyle=':')
#plt.plot(temperatures, entropy(temperatures, theta=760., a=40., b=1.))
#plt.plot(temperatures, np.gradient(gibbs(temperatures, theta=760., a=40., b=1.), temperatures), linestyle=':')
#plt.plot(temperatures, gibbs(temperatures, theta=760., a=2., b=2.), linestyle=':')

def eqn4(T, a, b, c, d, f):
    return a + b*np.log(T) + c/T + d/T/T + f/T/T/T

per = HP_2011_ds62.per()
per2 = SLB_2011.periclase()
pressures = 0.*temperatures + 1.e5
plt.plot(temperatures, per.evaluate(['C_p'], pressures, temperatures)[0], linestyle=':')
plt.plot(temperatures, per2.evaluate(['C_p'], pressures, temperatures)[0], linestyle='--')
plt.plot(temperatures, eqn4(temperatures, -87.613, 17.615, 22.158e3, -6.206e6, 5.289e8), linestyle='--')
plt.show()
