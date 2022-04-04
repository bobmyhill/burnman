import matplotlib.pyplot as plt 
import numpy as np


def F3(Q, Trel):
    p = Q*Q
    p_ord = 0.5*(p + 1.)
    p_dis = 1. - p_ord
    Srel = (p_ord*np.log(p_ord) + p_dis*np.log(p_dis))/np.log(0.5)
    return Trel -1*np.power(Q, 4) - Srel*Trel


def F(Q, Trel, n):
    Tc = 1.
    b = n/2
    a = n/Tc/2.
    return 1./2.*a*(Trel - Tc)*Q*Q + 1./n*b*np.power(Q, n)

Qs = np.linspace(0., 1., 1001)
ps = Qs

for Trel in [0., 0.5, 1., 1.5]:
    ln, = plt.plot(ps, F(Qs, Trel, n=6), label=f'$T = {Trel} T_c$')
    plt.plot(ps, F(Qs, Trel, n=4), linestyle=':',
             color=ln.get_color())
    # plt.plot(ps, F3(Qs, Trel), linestyle='--',
    #         color=ln.get_color())

Trels = np.linspace(0., 1.5, 101)
Qs = np.array([np.power(1. - Trel, 0.25)
               if Trel < 1. else 0. for Trel in Trels])
ps = Qs
plt.plot(ps, F(Qs, Trels, n=6), color='black', label='tricritical model (n=6)')

Qs = np.array([np.power(1. - Trel, 0.5)
               if Trel < 1. else 0. for Trel in Trels])
ps = Qs
plt.plot(ps, F(Qs, Trels, n=4), linestyle=':', color='black', label='n=4')

plt.legend()
plt.xlabel('$p_{ord}$ ($Q$)')
plt.ylabel(f'$Fn/2b$')
plt.savefig('landau_tricritical.pdf')
plt.show()

for Trel in [0., 0.5, 1., 1.5]:
    ln, = plt.plot(ps, F(Qs, Trel, n=6) - F(1, Trel, n=6), label=f'$T = {Trel} T_c$')
    plt.plot(ps, F(Qs, Trel, n=4) - F(1, Trel, n=4), linestyle=':',
             color=ln.get_color())

Trels = np.linspace(0., 1.5, 101)
Qs = np.array([np.power(1. - Trel, 0.25)
               if Trel < 1. else 0. for Trel in Trels])
ps = Qs
plt.plot(ps, F(Qs, Trels, n=6) - F(1., Trels, n=6), color='black', label='tricritical model (n=6)')
Qs = np.array([np.power(1. - Trel, 0.5)
               if Trel < 1. else 0. for Trel in Trels])
plt.plot(ps, F(Qs, Trels, n=4) - F(1., Trels, n=4), linestyle=':', color='black', label='n=4')

plt.legend()
plt.xlabel('$p_{ord}$ ($Q$)')
plt.ylabel(f'$Fn/2b$')
plt.savefig('landau_tricritical2.pdf')
plt.show()
