import numpy as np
from burnman.tools.unitcell import molar_volume_from_unit_cell_volume
from burnman import AnisotropicSolution
from quartz import forsterite
import matplotlib.pyplot as plt

from burnman.minerals.HP_2011_ds62 import q
from burnman.minerals.SLB_2011 import quartz



d = np.loadtxt('data/Lakshtanov_et_al_2007_Cijs_quartz.dat', unpack=True)
"""
for i in range(7):
    plt.scatter(d[0], d[i+2])
plt.show()
"""
TC = d[0]
TK = TC + 273.15
C11, C12, C13, C14, C33, C44 = d[2:8]
nul = C11*0.
C = np.array([[C11, C12, C13, C14, nul, nul],
              [C12, C11, C13, -C14, nul, nul],
              [C13, C13, C33, nul, nul, nul],
              [C14, -C14, nul, C44, nul, nul],
              [nul, nul, nul, nul, C44, C14],
              [nul, nul, nul, nul, C14, 0.5*(C11 - C12)]]).T

S = np.linalg.inv(C)
psi = np.einsum('ijk, i->ijk', S, 1./np.sum(S[:,:3,:3], axis=(2, 1)))


for i in [0, -1]:
    print(repr(d[0][i]))
    print(repr(psi[i]))


plt.plot(d[0], psi[:,0,0], label='$S_{11}/\\beta_{RT}$')
plt.plot(d[0], psi[:,2,2], label='$S_{33}/\\beta_{RT}$')
plt.plot(d[0], psi[:,0,1], label='$S_{12}/\\beta_{RT}$')
plt.plot(d[0], psi[:,0,2], label='$S_{13}/\\beta_{RT}$')
plt.plot(d[0], psi[:,0,3], label='$S_{14}/\\beta_{RT}$')
plt.plot(d[0], psi[:,4,4], label='$S_{44}/\\beta_{RT}$')
plt.legend()
plt.show()
exit()


#for i in range(6):
#    for j in range(6):
#        plt.scatter(TC, C[:, i, j])

l = []
m = []
for c in C:
    eigenValues, eigenVectors = np.linalg.eig(c)

    l.append(eigenValues)
    m.append(eigenVectors)

l = np.array(l)
m = np.array(m)

print(m)
print(m.shape)
for i in range(6):
    plt.scatter(TK, m[:, i, 3])
plt.xlim(0., )
plt.ylim(0., )
#plt.plot(V, Qs*Qs)
plt.show()
exit()

# Anisotropic solution
def psi_func(f, Pth, X, params):
    Psi = np.zeros((6, 6))
    dPsidf = np.zeros((6, 6))
    dPsidPth = np.zeros((6, 6))
    dPsidX = np.zeros((6, 6, 2))
    return (Psi, dPsidf, dPsidPth, dPsidX)

def nonconf_helmholtz_func(volume, temperature, molar_amounts):
    return 0.

# Initialised objects
# Scalar solution model
# Psi function
# Reference endmember
# Endmembers to X vector (each component should sum to zero)
# Components of the X vector that are freely varying on seismic timescales

qtz = AnisotropicSolution(
    name="quartz",
    solution_type="function",
    endmembers=[[forsterite(), '[Si]O2'],
                [forsterite(), '[Si]O2']],
    excess_helmholtz_function=nonconf_helmholtz_func,
    master_cell_parameters=np.array([1, 1, 1, 90, 90, 90]),
    anisotropic_parameters={},
    psi_excess_function=psi_func,
    dXdQ=np.array([[-1., 1.]]),
    orthotropic=True,
    relaxed=True,
)

qtz.set_composition([0.5, 0.5])
qtz.set_state(1.e5, 300.)
print(qtz.isentropic_compliance_tensor)
exit()

qtz = q()
#qtz.property_modifiers[0][1]['V_D'] = 1.25e-06
#qtz = quartz()

temperatures = np.linspace(10., 1300., 101)
pressures = temperatures * 0. + 1.e5
Qs = temperatures * 0. + 1.e5
V = qtz.evaluate(['V'], pressures, temperatures)[0]

for i, T in enumerate(temperatures):
    qtz.set_state(1.e5, T)
    Qs[i] = qtz.property_modifier_properties[0]['Q']


d = np.loadtxt('data/Carpenter_1998_quartz_unit_cell.dat', unpack=True)

plt.errorbar(d[0], molar_volume_from_unit_cell_volume(d[-2], 3.),
             xerr=d[1], yerr=molar_volume_from_unit_cell_volume(d[-1], 3.),
             fmt=' ', color='b')
plt.plot(temperatures, V)
plt.show()
