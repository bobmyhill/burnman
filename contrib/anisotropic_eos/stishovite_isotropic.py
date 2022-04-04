import numpy as np
import matplotlib.pyplot as plt

import burnman
from burnman import AnisotropicMineral


def Cij0(pressure):
    C110 = 578 + 5.38 * pressure
    C330 = 776 + 4.94 * pressure
    C120 = 86 + 5.38 * pressure
    C130 = 191 + 2.72 * pressure
    C440 = 252 + 1.88 * pressure
    C660 = 323 + 3.10 * pressure

    return np.array([[C110, C120, C130, 0., 0., 0.],
                     [C120, C110, C130, 0., 0., 0.],
                     [C130, C130, C330, 0., 0., 0.],
                     [0., 0., 0., C440, 0., 0.],
                     [0., 0., 0., 0., C440, 0.],
                     [0., 0., 0., 0., 0., C660]])


def make_phase():
    dP = 1.e6
    Sij0 = np.linalg.inv(Cij0(0))
    Sij1 = np.linalg.inv(Cij0(dP/1.e9))  # 1 kPa

    beta0 = np.sum(Sij0[:3, :3])
    beta1 = np.sum(Sij1[:3, :3])
    Kprime = (1./beta1 - 1./beta0)/dP*1.e9

    stv = burnman.minerals.SLB_2011.stishovite()
    stv.params['K_0'] = 1./beta0*1.e9
    stv.params['Kprime_0'] = Kprime
    stv.property_modifiers = []

    cell_parameters = np.array([4.1772, 4.1772, 2.6651, 90, 90, 90])

    cell_parameters[:3] *= np.cbrt(stv.params['V_0']
                                   / np.prod(cell_parameters[:3]))

    f_order = 2
    Pth_order = 0
    constants = np.zeros((6, 6, f_order+1, Pth_order+1))

    # S = beta * dchi / df
    # S/beta = c10 + c20*f
    stv.set_state(0., 300.)
    V0 = stv.V

    stv.set_state(dP, 300.)
    V1 = stv.V

    f = np.log(V1/V0)

    constants[:, :, 1, 0] = Sij0/beta0
    constants[:, :, 2, 0] = (Sij1/beta1 - Sij0/beta0)/f

    stv2 = AnisotropicMineral(stv, cell_parameters, constants)
    return stv2


def plot():

    stv2 = make_phase()
    pressures = np.linspace(1.e5, 125.e9, 26)
    Cijs = np.zeros((26, 9))
    for k, P in enumerate(pressures):
        print(P/1.e9)
        stv2.set_state(P, 300.)
        for m, (i, j) in enumerate([(1, 1),
                    (2, 2),
                    (3, 3),
                    (4, 4),
                    (5, 5),
                    (6, 6),
                    (1, 2),
                    (1, 3),
                    (2, 3)]):
            Cijs[k, m] = stv2.isothermal_stiffness_tensor[i-1, j-1]/1.e9

    for m in range(9):
        plt.plot(pressures/1.e9, Cijs[:,m])


    plt.plot(pressures/1.e9, 578 + 5.38 * pressures/1.e9, linestyle=':')
    plt.plot(pressures/1.e9, 776 + 4.94 * pressures/1.e9, linestyle=':')
    plt.plot(pressures/1.e9, 86 + 5.38 * pressures/1.e9, linestyle=':')
    plt.plot(pressures/1.e9, 191 + 2.72 * pressures/1.e9, linestyle=':')
    plt.plot(pressures/1.e9, 252 + 1.88 * pressures/1.e9, linestyle=':')
    plt.plot(pressures/1.e9, 323 + 3.10 * pressures/1.e9, linestyle=':')

    plt.show()

if __name__ == "__main__":
    plot()