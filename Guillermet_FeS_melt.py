import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve

R = 8.31446
def G(T, y_Fe, y_S):
    # This is a function to calculate the gibbs free energy for one mole of substance
    # which is equal to 2 moles total of Fe, S and vacancies.
    # It is *not* the same as 1 mole of Fe + S.
    # Moving from Fe to FeS essentially replaces vacancies with S atoms
    # (the lowest energy state contains essentially no vacancies until almost at FeS).

    # To normalise, divide through by y_Fe + y_S
    
    # N.B. This formalism doesn't seem valid microscopically;
    # Fe liquid is almost close-packed,
    # and adding S atoms will *generate* vacancies, not destroy them.



    if y_Fe > 1.:
        y_Fe = 1. - 1.e-12
    if y_S > 1.:
        y_S = 1. - 1.e-12
    
    G0_VaVa = 0.
    G0_FeVa = -11274. + 163.878*T - 22.03*T*np.log(T) + 4.1755e-3*T*T
    G0_VaS = -65357. + 165.396*T - 13.513*T*np.log(T)
    G0_FeS = -157578 + 200.119*T - 19.252*T*np.log(T)
    
    L_Va_VaFe = 100.*T
    L_Va_VaS = L_Va_VaFe
    L_Fe_VaS = np.array([31761. - 9.202*T,
                         10761. + 0.477*T]).dot(np.array([1.,
                                                          (1. - y_S) - y_S])) # just after eq. 3
    L_S_VaFe = np.array([79779. - 45.139*T,
                         57510. - 17.082*T]).dot(np.array([1.,
                                                           (1 - y_Fe) - y_Fe])) # just after eq. 3
    
    
    
    
    G_m = ((1. - y_Fe)*(1. - y_S)*G0_VaVa + y_Fe*(1. - y_S)*G0_FeVa + 
           (1. - y_Fe)*y_S*G0_VaS + y_Fe*y_S*G0_FeS +
           R*T*((1. - y_Fe)*np.log(1. - y_Fe) + y_Fe*np.log(y_Fe) +
                (1. - y_S)*np.log(1. - y_S) + y_S*np.log(y_S)) +
           (1. - y_Fe)*y_Fe*(1. - y_S)*L_Va_VaFe +
           (1. - y_Fe)*y_Fe*y_S*L_S_VaFe +
           (1. - y_S)*y_S*(1. - y_Fe)*L_Va_VaS +
           (1. - y_S)*y_S*y_Fe*L_Fe_VaS) # eq. 4
    return G_m




def GLVaVa(T, y_Fe, y_S):
    if y_Fe > 1.:
        y_Fe = 1. - 1.e-12
    if y_S > 1.:
        y_S = 1. - 1.e-12

        
    G0_VaVa = 0.
    G0_FeVa = -11274. + 163.878*T - 22.03*T*np.log(T) + 4.1755e-3*T*T
    G0_VaS = -65357. + 165.396*T - 13.513*T*np.log(T)
    G0_FeS = -157578 + 200.119*T - 19.252*T*np.log(T)
    
    L0_Va_VaFe = 100.*T
    L1_Va_VaFe = 0.
    
    L0_Va_VaS = L0_Va_VaFe
    L1_Va_VaS = 0.
    
    L0_Fe_VaS = 31761. - 9.202*T
    L1_Fe_VaS = 10761. + 0.477*T
    
    L0_S_VaFe = 79779. - 45.139*T
    L1_S_VaFe = 57510. - 17.082*T
    
    GVaVa = (G0_VaVa + y_Fe*y_S*(G0_VaS + G0_FeVa - G0_VaVa - G0_FeS) +
             R*T*np.log((1. - y_Fe)*(1. - y_S)) +
             y_Fe*((1. - y_Fe)*y_S + y_Fe*(1. - y_S))*L0_Va_VaFe +
             y_Fe*y_S*(2.*y_Fe - 1.)*L0_S_VaFe +
             y_S*(y_Fe*(1. - y_S) + y_S*(1 - y_Fe))*L0_Va_VaS +
             y_S*y_Fe*(2.*y_S - 1.)*L0_Fe_VaS +
             y_Fe*((1. - y_S)*(3. - 4*y_Fe))*L1_Va_VaFe +
             y_S*y_Fe*(6*y_Fe*(1. - y_Fe) - 1.)*L1_S_VaFe +
             y_S*(y_Fe*(1. - y_S)*(1. - 2.*y_S) + (1 - y_Fe)*y_S*(3. - 4*y_S))*L1_Va_VaS +
             y_Fe*y_S*(6.*y_S*(1. - y_S) - 1.)*L1_Fe_VaS)

    return GVaVa

def site_occupancies(T, x_S):
    x_Fe = 1. - x_S
    if x_Fe < 0.5:
        V_guess = (1. - 2.*x_Fe)/(1. - x_Fe)
    elif x_Fe > 0.5:
        V_guess = (1. - 2.*(1. - x_Fe))/(1. - (1. - x_Fe))
    else:
        V_guess = 0.0
        
    G_VaVa = lambda x_V, x_Fe, T: GLVaVa(T, (2. - x_V[0])*x_Fe, (2. - x_V[0])*x_Fe*(1./x_Fe - 1.))

    x_V = fsolve(G_VaVa, [V_guess+0.0001], args=(x_Fe, T))[0]

    y_Fe = (2. - x_V)*x_Fe
    y_S = (2. - x_V)*x_Fe*(1./x_Fe - 1.)
    y_VaFe = 1. - y_Fe
    y_VaS = 1 - y_S
    return([y_Fe, y_VaFe, y_S, y_VaS])

def calc_mu(T, y_Fe, y_S):
    
    if y_Fe > 1.:
        y_Fe = 1. - 1.e-12
    if y_S > 1.:
        y_S = 1. - 1.e-12

        
    G0_VaVa = 0.
    G0_FeVa = -11274. + 163.878*T - 22.03*T*np.log(T) + 4.1755e-3*T*T
    G0_VaS = -65357. + 165.396*T - 13.513*T*np.log(T)
    G0_FeS = -157578 + 200.119*T - 19.252*T*np.log(T)
    
    L0_Va_VaFe = 100.*T
    L1_Va_VaFe = 0.
    
    L0_Va_VaS = L0_Va_VaFe
    L1_Va_VaS = 0.
    
    L0_Fe_VaS = 31761. - 9.202*T
    L1_Fe_VaS = 10761. + 0.477*T
    
    L0_S_VaFe = 79779. - 45.139*T
    L1_S_VaFe = 57510. - 17.082*T
    
    mu_S = ((1. - y_Fe)*(G0_VaS - G0_VaVa) +
            y_Fe*(G0_FeS - G0_FeVa) +
            R*T*np.log(y_S/(1. - y_S)) - y_Fe*(1 - y_Fe)*L0_Va_VaFe +
            y_Fe*(1 - y_Fe)*L0_S_VaFe + (1. - y_Fe)*(1 - 2.*y_S)*L0_Va_VaS +
            y_Fe*(1. - 2.*y_S)*L0_Fe_VaS +
            y_Fe*(1 - y_Fe)*(2.*y_Fe - 1.)*L1_Va_VaFe +
            y_Fe*(1. - y_Fe)*(1. - 2.*y_Fe)*L1_S_VaFe +
            (1. - y_Fe)*(1. - 6.*y_S*(1 - y_S))*L1_Va_VaS +
            y_Fe*(1. - 6.*y_S*(1 - y_S))*L1_Fe_VaS)

    
    mu_Fe = ((1. - y_S)*(G0_FeVa - G0_VaVa) +
            y_S*(G0_FeS - G0_VaS) +
            R*T*np.log(y_Fe/(1. - y_Fe)) - y_S*(1 - y_S)*L0_Va_VaS +
            y_S*(1 - y_S)*L0_Fe_VaS + (1. - y_S)*(1 - 2.*y_Fe)*L0_Va_VaFe +
            y_S*(1. - 2.*y_Fe)*L0_S_VaFe +
            y_S*(1 - y_S)*(2.*y_S - 1.)*L1_Va_VaS +
            y_S*(1. - y_S)*(1. - 2.*y_S)*L1_Fe_VaS +
            (1. - y_S)*(1. - 6.*y_Fe*(1 - y_Fe))*L1_Va_VaFe +
             y_S*(1. - 6.*y_Fe*(1 - y_Fe))*L1_S_VaFe)
    return mu_Fe, mu_S

temperatures = np.linspace(1073., 2073., 6)
x_Ss = np.linspace(0.001, 0.999, 200)
Gs = np.empty_like(x_Ss)
Ss = np.empty_like(x_Ss)
mu_Fe = np.empty_like(x_Ss)
mu_S = np.empty_like(x_Ss)

fig = plt.figure()
ax = [fig.add_subplot(1, 2, i+1) for i in range(2)]
for T in temperatures:
    for i, x_S in enumerate(x_Ss):
        deltaT = 0.001
        y_Fe, y_VaFe, y_S, y_VaS = site_occupancies(T-deltaT/2., x_S)
        G0 = G(T-deltaT/2., y_Fe, y_S)/(y_Fe + y_S) # note normalisation

        y_Fe, y_VaFe, y_S, y_VaS = site_occupancies(T+deltaT/2., x_S)
        G1 = G(T+deltaT/2., y_Fe, y_S)/(y_Fe + y_S) # note normalisation

        Ss[i] = (G0 - G1)/deltaT
        
        y_Fe, y_VaFe, y_S, y_VaS = site_occupancies(T, x_S)
        Gs[i] = G(T, y_Fe, y_S)/(y_Fe + y_S) # note normalisation
        
        mu_Fe[i], mu_S[i]  = calc_mu(T, y_Fe, y_S)


    G0_FeVa = -11274. + 163.878*T - 22.03*T*np.log(T) + 4.1755e-3*T*T
    G0_FeS = -157578 + 200.119*T - 19.252*T*np.log(T)
    G0_VaS = -65357. + 165.396*T - 13.513*T*np.log(T)
    
    S0_FeVa = -163.878 + 22.03*(np.log(T) + 1.) - 2.*4.1755e-3*T
    S0_VaS = -165.396 + 13.513*(np.log(T) + 1.)


    ax[0].plot(x_Ss, Gs - ((1. - x_Ss)*G0_FeVa +  x_Ss*G0_VaS), label='{0} K'.format(T))

    #ax[0].plot(x_Ss, Gs, label='{0} K'.format(T))
    #ax[0].plot(x_Ss, (1. - x_Ss)*mu_Fe + x_Ss*mu_S, linestyle=':', linewidth=2)
    
    ax[1].plot(x_Ss, (Ss - ((1. - x_Ss)*S0_FeVa +  x_Ss*S0_VaS))/R, label='{0} K'.format(T))
        
    #plt.plot(x_Ss, np.exp(mu_S/(R*T)))
    #plt.plot(x_Ss, np.exp(mu_Fe/(R*T)))
    #plt.scatter([0.], [G0_FeVa])

ax[1].plot(x_Ss, -x_Ss*np.log(x_Ss) - (1. - x_Ss)*np.log(1. - x_Ss), label='ideal')
ax[1].plot(x_Ss, -x_Ss*np.log(x_Ss) - (1. - x_Ss)*np.log(1. - x_Ss), label='ideal')

ax[1].plot((x_Ss/(1. + x_Ss)), -1.*(x_Ss*np.log(x_Ss) + (1. - x_Ss)*np.log(1. - x_Ss)), label='ideal Fe-FeS')
    
#plt.ylim(0., 0.02)
for i in range(2):
    ax[i].set_xlabel('$x_{S}$')
    ax[i].legend(loc='best')
ax[0].set_ylabel('G (J/mol)')
ax[1].set_ylabel('S (J/K/mol)')
plt.show()

def equilibrate(x_S, T):
    y_Fe, y_VaFe, y_S, y_VaS = site_occupancies(T, x_S)
    mu_Fe, mu_S = calc_mu(T, y_Fe, y_S)

    return 1. - np.exp(mu_Fe/(R*T))


temperatures = np.linspace(1073., 1800., 101)
X_Ss = np.empty_like(temperatures)
guess = [0.45]
for i, T in enumerate(temperatures):
    res = fsolve(equilibrate, guess, args=(T))
    X_Ss[i] = res 
    guess = res[0]
    
plt.plot(X_Ss, temperatures)
plt.xlim(0., 0.5)
plt.show()
