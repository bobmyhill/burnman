import numpy as np
from scipy.optimize import root


def transition_temperature(pressure, alpha, beta):
    def delta_gibbs(temperature, pressure):
        alpha.set_state(pressure, temperature[0])
        beta.set_state(pressure, temperature[0])
        delta_gibbs = alpha.gibbs - beta.gibbs

        return delta_gibbs

    sol = root(delta_gibbs, [100.0], args=(pressure))
    return sol.x[0]


def transition_temperature_MM1980(pressure):
    pkbar = pressure / 1.0e8
    return (
        573.194
        + 273.15
        + 27.1084 * pkbar
        - 0.23607 * pkbar * pkbar
        + 0.00391 * np.power(pkbar, 3.0)
    )


def transition_temperature_Angel(pressure):
    # note the -10 is mistakenly called d2TdP2 in the table
    # it is in fact 0.5 x d2TdP2
    P_GPa = pressure / 1.0e9
    return 847 + 270 * P_GPa - 10.0 * P_GPa * P_GPa
