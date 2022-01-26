import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

cs = ['red', 'orange', 'green', 'blue']


Ps = np.linspace(0., 150., 16)
print(Ps)
Cijs = np.zeros((5, 4*len(Ps)))
Cijs[0, 0*len(Ps):1*len(Ps)] = 300.
Cijs[0, 1*len(Ps):2*len(Ps)] = 1000.
Cijs[0, 2*len(Ps):3*len(Ps)] = 2000.
Cijs[0, 3*len(Ps):4*len(Ps)] = 3000.
Cijs[1, 0*len(Ps):1*len(Ps)] = Ps*1.e9
Cijs[1, 1*len(Ps):2*len(Ps)] = Ps*1.e9
Cijs[1, 2*len(Ps):3*len(Ps)] = Ps*1.e9
Cijs[1, 3*len(Ps):4*len(Ps)] = Ps*1.e9

header='T(K) P(Pa) '
for i, Cij in enumerate(['C11', 'C12', 'C44']):
    for j, T in enumerate(['300', '1000', '2000', '3000']):
        d = np.loadtxt(f'Karki_2000_periclase_{Cij}_{T}K.dat')
        spl = interp1d(d[:, 0], d[:, 1], kind='linear')
        Pfines = np.linspace(3., 147., 70)
        cijs = spl(Pfines)

        spl2 = interp1d(Pfines, cijs, fill_value='extrapolate')
        cijs = spl2(Ps)
        plt.scatter(Ps, cijs, c=cs[j])
        plt.plot(Ps, cijs, c=cs[j])
        
        
        print(i, j)
        Cijs[i+2, j*len(Ps):(j+1)*len(Ps)] = cijs*1.e9

    header += Cij + '(Pa) '

np.savetxt('Karki_2000_periclase_CSijs.dat', X=Cijs.T, header=header, fmt='%3e')

plt.show()
