import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d, bisplrep, bisplev

cs = ['red', 'orange', 'green', 'blue']


# Make 2d interp grids for alpha and Cp
P = []
T = []
alpha = []
for i, PGPa in enumerate(['0', '10', '30', '60', '100', '150']):
    d = np.loadtxt(f'Karki_2000_periclase_alpha_{PGPa}GPa.dat')
    P.extend(float(PGPa) + d[:, 0]*0.)
    T.extend(d[:, 0])
    alpha.extend(d[:, 1]*1.e-5)

f_alphaV = interp2d(P, T, alpha, kind='cubic')

P = []
T = []
Cp = []
for i, PGPa in enumerate(['0', '10', '30', '60', '100', '150']):
    d = np.loadtxt(f'Karki_2000_periclase_Cp_{PGPa}GPa.dat')
    ns = 12
    P.extend(float(PGPa) + d[ns:, 0]*0.)
    T.extend(d[ns:, 0])
    Cp.extend(d[ns:, 1])

rep_Cp = bisplrep(P, T, Cp, kx=3, ky=3, tx=5, ty=30)
def f_Cp(P, T):
    return bisplev(P, T, rep_Cp)

Ps = np.linspace(0., 150., 31)
print(Ps)
Cijs = np.zeros((8, 4*len(Ps)))
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
        d_V = np.loadtxt(f'Karki_2000_periclase_V_{T}K.dat')
        spl = interp1d(d[:, 0], d[:, 1], kind='linear')
        spl_V = interp1d(d_V[:, 0], d_V[:, 1], kind='linear')
        Pfines = np.linspace(3., 147., 70)
        cijs = spl(Pfines)
        Vs = spl_V(Pfines)

        PfinesK = np.linspace(3., 147., 20)
        KTs = -spl_V(PfinesK)*np.gradient(PfinesK, spl_V(PfinesK), edge_order=2)

        spl2 = interp1d(Pfines, cijs, fill_value='extrapolate')
        spl2V = interp1d(Pfines, Vs, fill_value='extrapolate')
        spl2K = interp1d(PfinesK, KTs, fill_value='extrapolate')
        cijs = spl2(Ps)*1.e9
        Vs = spl2V(Ps)*1e-30*6.02214e23
        KTs = spl2K(Ps)*1.e9
        
        alphas = []
        Cps = []
        for P in Ps:
            alphas.append(f_alphaV(P, float(T))[0])
            Cps.append(f_Cp(P, float(T)))

        alphas = np.array(alphas)
        Cps = np.array(Cps)
        temperatures = Cps*0. + float(T)

        factor = alphas*alphas*Vs*temperatures/Cps

        plt.scatter(Ps, factor, c=cs[j])
        #plt.scatter(Ps, KTs, c=cs[j])
        #plt.plot(Ps, KTs, c=cs[j])
        #plt.scatter(Ps, cijs, c=cs[j])
        #plt.plot(Ps, cijs, c=cs[j])
        print(i, j)
        Cijs[i+2, j*len(Ps):(j+1)*len(Ps)] = cijs
        Cijs[i+3, j*len(Ps):(j+1)*len(Ps)] = KTs
        Cijs[i+4, j*len(Ps):(j+1)*len(Ps)] = Vs
        Cijs[i+5, j*len(Ps):(j+1)*len(Ps)] = factor

    header += Cij + '(Pa) '

header += 'KT (Pa) V (m^3/mol) betaT-betaS (Pa^-1)'
np.savetxt('Karki_2000_periclase_CSijs.dat', X=Cijs.T, header=header, fmt='%3e')

plt.show()
