from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def fn_RTlng_Mg2SiO4(temperatures, Tm, b, R, W):
    a = -W / ((b - 1.0) ** 2 * Tm * temperatures)
    return a * (
        (temperatures - Tm) * (temperatures - Tm)
        + (1.0 - b**2) * Tm * (temperatures - Tm)
    )


b = 0.5
Tm = 2560
W = -90000
R = 8.31446
temperatures = np.linspace(b * Tm, 2560, 1001)

lnRTg_Mg2SiO4 = fn_RTlng_Mg2SiO4(temperatures, Tm, b, R, W)
p_H2OL = np.sqrt(lnRTg_Mg2SiO4 / W)
plt.plot(temperatures, p_H2OL)


plt.xlabel("Temperature")
plt.ylabel("proportions")
plt.xlim(0.0, Tm + 100.0)
plt.show()

"""
exit()


def fn_lng_Mg2SiO4(temperatures, Tm, a, b):
    return a*(temperatures - Tm)*(temperatures - Tm) + b*(temperatures - Tm)


def fn_p(temperatures, Tm):
    Th = 8.*(1 - temperatures/Tm)
    return Th/(np.exp(Th) - 1.)

temperatures = np.linspace(0.01, 2560, 1001)
Tm = 2560
W = -90000
R = 8.31446
lng_Mg2SiO4 = fn_lng_Mg2SiO4(temperatures, Tm, -2.82158106e-06,  2.03499012e-03)
p_H2OL = np.sqrt(lng_Mg2SiO4/(W/(R*temperatures)))
plt.plot(temperatures, p_H2OL)

plt.plot(temperatures, 1. - fn_p(temperatures, 2560))

plt.xlabel('Temperature')
plt.ylabel('proportions')
plt.show()
"""
