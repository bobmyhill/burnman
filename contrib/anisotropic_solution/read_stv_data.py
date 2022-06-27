import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def F(Q):
    a = 0.01
    b = 0.003
    z = 0.*Q
    psidelta = np.array([[a*Q + b/2.*Q*Q, z, z],
                         [z, -a*Q + b/2.*Q*Q, z],
                         [z, z, - b*Q*Q]])
    exps = np.array([expm(p) for p in psidelta.T])
    return exps


Qs = np.linspace(0., 1., 101)
Fs = F(Qs)
plt.plot(Qs*Qs, Fs[:, 0, 0])
plt.plot(Qs*Qs, Fs[:, 1, 1])
plt.plot(Qs*Qs, Fs[:, 2, 2])
plt.show()

unit_cell = np.loadtxt('data/Zhang_2021_stishovite_unit_cell.dat')


P, Perr = unit_cell[:, :2].T
a, aerr = unit_cell[:, 2:4].T
b, berr = unit_cell[:, 4:6].T
c, cerr = unit_cell[:, 6:8].T
V, Verr = unit_cell[:, 8:10].T
rho, rhoerr = unit_cell[:, 10:12].T

# plt.plot(P, a/c)
# plt.show()
fig = plt.figure(figsize=(12, 4))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

ax[0].scatter(P, a, label='a')
ax[0].scatter(P, b, label='b')
ax[0].scatter(P, c, label='c')

elastic_tensor = np.loadtxt('data/Zhang_2021_stishovite_elastic_tensor.dat')

P, Perr = elastic_tensor[:, :2].T
C11, C11err = elastic_tensor[:, 2:4].T
C12, C12err = elastic_tensor[:, 4:6].T
C13, C13err = elastic_tensor[:, 6:8].T
C22, C22err = elastic_tensor[:, 8:10].T
C23, C23err = elastic_tensor[:, 10:12].T
C33, C33err = elastic_tensor[:, 12:14].T
C44, C44err = elastic_tensor[:, 14:16].T
C55, C55err = elastic_tensor[:, 16:18].T
C66, C66err = elastic_tensor[:, 18:20].T
rho, rhoerr = elastic_tensor[:, 20:22].T
nul = 0.*C11

C = np.array([[C11, C12, C13, nul, nul, nul],
              [C12, C22, C23, nul, nul, nul],
              [C13, C23, C33, nul, nul, nul],
              [nul, nul, nul, C44, nul, nul],
              [nul, nul, nul, nul, C55, nul],
              [nul, nul, nul, nul, nul, C66]]).T


S = np.linalg.inv(C)
w, v = np.linalg.eig(C)

beta_S = np.sum(np.sum(S[:, :3, :3], axis=1), axis=1)
ax[1].scatter(P, 1./beta_S, label='K_S')


ax[1].scatter(P, C11, label='C11')
ax[1].scatter(P, C22, label='C22')
ax[1].scatter(P, C33, label='C33')
ax[1].scatter(P, C12, label='C12')
ax[1].scatter(P, C13, label='C13')
ax[1].scatter(P, C23, label='C23')
ax[1].scatter(P, C44, label='C44')
ax[1].scatter(P, C55, label='C55')
ax[1].scatter(P, C66, label='C66')

ax[1].errorbar(P, C11, xerr=Perr, yerr=C11err, fmt='None')
ax[1].errorbar(P, C22, xerr=Perr, yerr=C22err, fmt='None')
ax[1].errorbar(P, C33, xerr=Perr, yerr=C33err, fmt='None')
ax[1].errorbar(P, C12, xerr=Perr, yerr=C12err, fmt='None')
ax[1].errorbar(P, C13, xerr=Perr, yerr=C13err, fmt='None')
ax[1].errorbar(P, C23, xerr=Perr, yerr=C23err, fmt='None')
ax[1].errorbar(P, C44, xerr=Perr, yerr=C44err, fmt='None')
ax[1].errorbar(P, C55, xerr=Perr, yerr=C55err, fmt='None')
ax[1].errorbar(P, C66, xerr=Perr, yerr=C66err, fmt='None')

for i in range(6):
    ax[2].scatter(P, w[:, i])
ax[2].set_ylim(0., )

for i in range(2):
    ax[i].legend()
plt.show()
