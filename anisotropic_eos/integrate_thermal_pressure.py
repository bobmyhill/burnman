import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

def P_th(x):
    return 1./(np.exp(1./x) - 1.)

def Pth3(x):
    return np.exp(-x) + x - 1.

temperatures = np.linspace(0.0001, 20., 10001)
plt.plot(temperatures, Pth3(temperatures))
plt.plot(temperatures, P_th(temperatures))
plt.show()
exit()

temperatures = np.linspace(0.0001, 20., 10001)
plt.plot(temperatures, np.exp(1./temperatures))
plt.show()
#def P_th2(x):
#    u = 0.5/x
#    return (-1.-1/(2.*u)-1./(u*(np.exp(-1./u) - 1)))

params = {'Cp': [1./4., 0., 2./4.]}

temperatures = np.linspace(0.0001, 20., 10001)
P_ths = P_th(temperatures)
Ss = cumtrapz(P_ths/temperatures, temperatures, initial=0)
#plt.plot(temperatures, P_th(temperatures))
#plt.plot(temperatures, P_th(temperatures))


S_coords = np.polynomial.chebyshev.chebfit(temperatures, Ss, 15)
G_coords = np.polynomial.chebyshev.chebint(S_coords)


temperatures = np.linspace(0.0001, 1., 10001)
P_ths = P_th(temperatures)
plt.plot(temperatures, cumtrapz(P_ths/temperatures, temperatures, initial=0), linestyle=':')
plt.plot(temperatures, np.polynomial.chebyshev.chebval(temperatures, S_coords))
#plt.plot(temperatures, np.polynomial.chebyshev.chebval(temperatures, G_coords))
plt.show()
