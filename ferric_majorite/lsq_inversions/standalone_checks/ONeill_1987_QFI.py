import numpy as np

from input_dataset import *

"""
qtz.set_state(1.e5, 298.15)
print(qtz.S, 41.51, '+/- 0.1 (Hemingway et al., 1991)')

qtz.set_state(1.e5, 1500.)
print(qtz.S)

bcc_iron.set_state(1.e5, 298.15)
print(bcc_iron.S, bcc_iron.C_p)

bcc_iron.set_state(1.e5, 298.15)
H_0 = bcc_iron.H
fcc_iron.set_state(1.e5, 1185.)
print(fcc_iron.H - H_0)
print(fcc_iron.S - bcc_iron.S, fcc_iron.C_p, '(Desai, 1986 suggests S_0 for BCC 0.2 J/K/mol lower)')
"""
print(qtz.params['S_0'])

fa = burnman.minerals.HP_2011_ds62.fa()
fa.params['S_0'] -= 0.3
for (T, m) in [(900., bcc_iron),
               (1000., bcc_iron),
               (1042., bcc_iron),
               (1100., bcc_iron),
               (1200., fcc_iron),
               (1000., qtz),
               (1000., fa),
               (1000., O2)]:
    m.set_state(1.e5, T)
    print(m.S)


def mu_O2_Fe_FeO(T):
    if T < 833.:
        raise Exception('T too low')
    elif T < 1042.:
        return -605812. + 1366.718*T - 182.7955*T*np.log(T) + 0.103592*T*T
    elif T < 1184.:
        return -519357. + 59.427*T + 8.9276*T*np.log(T)
    elif T < 1450.:
        return -551159. + 269.404*T - 16.9484*T*np.log(T)
    else:
        raise Exception('T too high')

def mu_O2_Mo_MoO2(T):
    if T < 1000.:
        raise Exception('T too low')
    elif T < 1450.:
        return -603268. + 337.460*T - 20.6892*T*np.log(T)
    else:
        raise Exception('T too high')

F = 96484.56 # value of Faraday constant from paper

minerals = [fa, qtz, bcc_iron, fcc_iron, O2]


data = np.loadtxt('data/ONeill_1987_QFI_FeFeO_electrode.dat')
diff_gibbs = np.empty_like(data[:,0])

for i, (T, emfmV) in enumerate(data):

    emf = emfmV*1.e-3
    mu_O2_ref = mu_O2_Fe_FeO(T)
    mu_O2 = mu_O2_ref - 4.*F*emf # 4FE = mu_O2B - mu_O2A; reference electrode is electrode B

    for m in minerals:
        m.set_state(1.e5, T)

    if T < 1184:
        iron = bcc_iron
    else:
        iron = fcc_iron
    diff_gibbs[i] = fa.gibbs - (2.*iron.gibbs + qtz.gibbs + O2.gibbs) - mu_O2
                                
mask = [i for i, d in enumerate(data) if d[0] > 50.]

plt.scatter(data[:,0][mask], diff_gibbs[mask])

data = np.loadtxt('data/ONeill_1987_QFI_MoMoO2_electrode.dat')
diff_gibbs = np.empty_like(data[:,0])

for i, (T, emfmV) in enumerate(data):

    emf = emfmV*1.e-3
    mu_O2_ref = mu_O2_Mo_MoO2(T)
    mu_O2 = mu_O2_ref - 4.*F*emf    


    for m in minerals:
        m.set_state(1.e5, T)

    
    if T < 1184:
        iron = bcc_iron
    else:
        iron = fcc_iron
    diff_gibbs[i] = fa.gibbs - (2.*iron.gibbs + qtz.gibbs + O2.gibbs) - mu_O2

mask = [i for i, d in enumerate(data) if d[0] > 50.]
    
plt.scatter(data[:,0][mask], diff_gibbs[mask])
plt.show()
exit()
