import autograd.numpy as np
import burnman


qtz_HP = burnman.minerals.HP_2011_ds62.q()

eps = np.finfo(float).eps
sqrt_eps = np.sqrt(np.finfo(float).eps)
log_eps = np.log(np.finfo(float).eps)

chebyshev_representation = np.array(
    [2.707737068327440945 / 2.0, 0.340068135211091751, -0.12945150184440869e-01,
     0.7963755380173816e-03, -
     0.546360009590824e-04, 0.39243019598805e-05,
     -0.2894032823539e-06, 0.217317613962e-07, -
     0.16542099950e-08,
     0.1272796189e-09, -
     0.987963460e-11, 0.7725074e-12, -
     0.607797e-13,
     0.48076e-14, -0.3820e-15, 0.305e-16, -0.24e-17])

def _chebval(x, c):
    """
    Evaluate a Chebyshev series at points x.
    This is just a lightly modified copy/paste job from the numpy
    implementation of the same function, copied over here to put a
    jit wrapper around it.
    """
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x

def debye_fn_cheb(x):
    """
    Evaluate the Debye function using a Chebyshev series expansion coupled with
    asymptotic solutions of the function.  Shamelessly adapted from the GSL implementation
    of the same function (Itself adapted from Collected Algorithms from ACM).
    Should give the same result as debye_fn(x) to near machine-precision.
    """
    val_infinity = 19.4818182068004875
    xcut = -log_eps

    assert(x > 0.0)  # check for invalid x

    if x < 2.0 * np.sqrt(2.0) * sqrt_eps:
        return 1.0 - 3.0 * x / 8.0 + x * x / 20.0
    elif x <= 4.0:
        t = x * x / 8.0 - 1.0
        c = _chebval(t, chebyshev_representation)
        return c - 0.375 * x
    elif x < -(np.log(2.0) + log_eps):
        nexp = int(np.floor(xcut / x))
        ex = np.exp(-x)
        xk = nexp * x
        rk = nexp
        sum = 0.0
        for i in range(nexp, 0, -1):
            xk_inv = 1.0 / xk
            sum *= ex
            sum += (((6.0 * xk_inv + 6.0) * xk_inv + 3.0) * xk_inv + 1.0) / rk
            rk -= 1.0
            xk -= x
        return val_infinity / (x * x * x) - 3.0 * sum * ex
    elif x < xcut:
        x3 = x * x * x
        sum = 6.0 + 6.0 * x + 3.0 * x * x + x3
        return (val_infinity - 3.0 * sum * np.exp(-x)) / x3
    else:
        return ((val_infinity / x) / x) / x

def debye_helmholtz_free_energy(T, debye_T, n):
    """
    Helmholtz free energy of lattice vibrations in the Debye model.
    It is important to note that this does NOT include the zero
    point energy of vibration for the lattice.  As long as you are
    calculating relative differences in F, this should cancel anyways.
    In Joules.
    """
    if T <= eps:
        return 0.
    x = debye_T / T
    F = n * burnman.constants.gas_constant * T * \
        (3.0 * np.log(1.0 - np.exp(-x)) - debye_fn_cheb(x))
    return F

def _debye_temperature(x, params):
    """
    Finite strain approximation for Debye Temperature [K]
    x = ref_vol/vol
    """
    f = 1. / 2. * (pow(x, 2. / 3.) - 1.)
    a1_ii = 6. * params['grueneisen_0']  # EQ 47
    a2_iikk = (-12. * params['grueneisen_0']
                + 36. * pow(params['grueneisen_0'], 2.)
                - 18. * params['q_0'] * params['grueneisen_0'])  # EQ 47
    nu_o_nu0_sq = 1. + a1_ii * f + 1. / 2. * a2_iikk * f * f
    if nu_o_nu0_sq > 0.:
        return params['Debye_0'] * np.sqrt(nu_o_nu0_sq)
    else:
        raise Exception(f'This volume (V = {1./x:.2f}*V_0) exceeds the '
                        'valid range of the thermal '
                        'part of the slb equation of state.')

def helmholtz_free_energy_beta(temperature, volume):
    params = qtz_beta.params
    x = params['V_0'] / volume
    f = 1. / 2. * (pow(x, 2. / 3.) - 1.)
    Debye_T = _debye_temperature(params['V_0'] / volume, params)

    F_quasiharmonic = (debye_helmholtz_free_energy(temperature,
                                                    Debye_T,
                                                    params['n'])
                        - debye_helmholtz_free_energy(params['T_0'],
                                                        Debye_T,
                                                        params['n']))

    b_iikk = 9. * params['K_0']  # EQ 28
    b_iikkmm = 27. * params['K_0'] * (params['Kprime_0'] - 4.)  # EQ 29

    F = (params['F_0'] + 0.5 * b_iikk * f * f * params['V_0']
            + (1. / 6.) * params['V_0'] * b_iikkmm * f * f * f
            + F_quasiharmonic)

    return F

def helmholtz_free_energy_alpha(temperature, volume):
    params = qtz_alpha.params
    x = params['V_0'] / volume
    f = 1. / 2. * (pow(x, 2. / 3.) - 1.)
    Debye_T = _debye_temperature(params['V_0'] / volume, params)

    F_quasiharmonic = (debye_helmholtz_free_energy(temperature,
                                                    Debye_T,
                                                    params['n'])
                        - debye_helmholtz_free_energy(params['T_0'],
                                                        Debye_T,
                                                        params['n']))

    b_iikk = 9. * params['K_0']  # EQ 28
    b_iikkmm = 27. * params['K_0'] * (params['Kprime_0'] - 4.)  # EQ 29

    F = (params['F_0'] + 0.5 * b_iikk * f * f * params['V_0']
            + (1. / 6.) * params['V_0'] * b_iikkmm * f * f * f
            + F_quasiharmonic)

    return F

# This is the quartz with the structural state at 1 bar and room temperature
qtz_alpha = burnman.minerals.SLB_2011.quartz()
qtz_alpha.property_modifiers = []
qtz_alpha.params['V_0'] = qtz_HP.params['V_0']
qtz_alpha.params['K_0'] = qtz_HP.params['K_0']
qtz_alpha.params['Kprime_0'] = qtz_HP.params['Kprime_0']
qtz_alpha.params['Debye_0'] = qtz_alpha.params['Debye_0'] + 45.


qtz_beta = burnman.minerals.SLB_2011.quartz()
qtz_beta.property_modifiers = []
qtz_beta.params['Debye_0'] = qtz_beta.params['Debye_0'] - 5.
qtz_beta.params['V_0'] = qtz_HP.params['V_0'] + 1.188e-06 * 0.88
qtz_beta.params['K_0'] = qtz_HP.params['K_0'] - 15.e9
qtz_beta.params['Kprime_0'] = qtz_HP.params['Kprime_0']
qtz_beta.params['F_0'] = qtz_beta.params['F_0'] + 1510.

"""
import matplotlib.pyplot as plt
temperatures = np.linspace(300., 1100., 101)
pressures = 1.e5 + temperatures*0.

plt.plot(temperatures, qtz_alpha.evaluate(['C_p'], pressures, temperatures)[0])
plt.plot(temperatures, qtz_beta.evaluate(['C_p'], pressures, temperatures)[0])
plt.plot(temperatures, qtz_beta.evaluate(['C_p'], pressures + 3.558e9, temperatures)[0], linestyle=':')
plt.show()
exit()
"""

qtz_alpha.set_state(1.e5, 573.94+273.15)
qtz_beta.set_state(1.e5, 573.94+273.15)

print(qtz_beta.V - qtz_alpha.V)
print(qtz_beta.S - qtz_alpha.S)
print(qtz_HP.property_modifiers[0][1]['V_D']*0.899)
print(qtz_HP.property_modifiers[0][1]['S_D']*0.899)
# Bachheimer and Dolino, 1975
# F = (T - T0)*eta*eta/2. + b*np.power(eta, 4.)/4. + c*np.power(eta, 6.)/6.

# Variables:
# Q = 0. becomes a global minimum at T1+delta
# Q = ? is the global minimum at T1-delta
# Q = 1. is the global minimum at T0

# qtz_beta.params['F_0'] changes the position of the minimum at 273.15 K

b = 6.6 # 200 -166.8
c = 5.2 # 300
T1 = 573.94+273.15

# Convert into excess form
qtz_beta.set_state(1.e5, T1)
F_beta0 = helmholtz_free_energy_beta(T1, qtz_beta.V)
F_alpha0 = helmholtz_free_energy_alpha(T1, qtz_beta.V)

a = (F_alpha0 - F_beta0) / (b/4. + c/6.)
b_BD = a*b/4.
c_BD = a*c/6.

def quartz_helmholtz_function(volume, temperature, molar_amounts):

    n_moles = sum(molar_amounts)
    molar_fractions = molar_amounts / n_moles

    F_beta = helmholtz_free_energy_beta(temperature, volume)
    F_alpha = helmholtz_free_energy_alpha(temperature, volume)

    Q = (molar_fractions[0] - molar_fractions[1])
    Qsqr = Q*Q
    F_xs = ((F_alpha - F_beta)*(Qsqr - 1.)
            + b_BD*(Qsqr - 1.)*Qsqr
            + c_BD*(Qsqr*Qsqr - 1.)*Qsqr)
    return n_moles * F_xs


class qtz_ss_scalar(burnman.ElasticSolution):
    def __init__(self, molar_fractions=None):
        self.name = 'quartz'
        self.solution_type = 'function'
        self.endmembers = [[qtz_alpha, '[Si]O2'],
                           [qtz_alpha, '[Si]O2']]
        self.excess_helmholtz_function = quartz_helmholtz_function

        burnman.ElasticSolution.__init__(self, molar_fractions)



def psi_func(f, Pth, params):
    dPsidf = (params['a'] + params['b_1']*params['c_1']
              * np.exp(params['c_1']*f) + params['b_2']
              * params['c_2']*np.exp(params['c_2']*f))
    Psi = (0. + params['a']*f
           + params['b_1'] * np.exp(params['c_1']*f)
           + params['b_2']*np.exp(params['c_2']*f)
           + params['d'] * Pth/1.e9)
    dPsidPth = params['d']/1.e9
    return (Psi, dPsidf, dPsidPth)

# https://serc.carleton.edu/NAGTWorkshops/mineralogy/mineral_physics/tensors.html
# see also Nye, 1959
p11 = 4.79501845e-01
p12 = -6.91663267e-02
p13 = -4.62784420e-02
p14 = -1.64139386e-01
p33 = 3.64442730e-01
p44 = 7.31250186e-01
p66 = 2.*(p11 - p12)
alpha_params = {'a': np.array([[p11,  p12,  p13,  p14,  0.,   0.],
                               [p12,  p11,  p13, -p14,  0.,   0.],
                               [p13,  p13,  p33,  0.,   0.,   0.],
                               [p14, -p14,  0.,   p44,  0.,   0.],
                               [0.,   0.,   0.,   0.,   p44,  2.*p14],
                               [0.,   0.,   0.,   0.,   2.*p14, p66]]),
                'b_1': np.zeros((6, 6)),
                'c_1': np.zeros((6, 6)),
                'b_2': np.zeros((6, 6)),
                'c_2': np.zeros((6, 6)),
                'd': np.zeros((6, 6))}

antialpha_params = {'a': np.array([[p11,  p12,  p13, -p14,  0.,   0.],
                                   [p12,  p11,  p13,  p14,  0.,   0.],
                                   [p13,  p13,  p33,  0.,   0.,   0.],
                                   [-p14, p14,  0.,   p44,  0.,   0.],
                                   [0.,   0.,   0.,   0.,   p44,  -2.*p14],
                                   [0.,   0.,   0.,   0.,   -2.*p14, p66]]),
                    'b_1': np.zeros((6, 6)),
                    'c_1': np.zeros((6, 6)),
                    'b_2': np.zeros((6, 6)),
                    'c_2': np.zeros((6, 6)),
                    'd': np.zeros((6, 6))}

p11b = 0.66714632
p12b = -0.07477777
p13b = -0.25360633
p14b = 0.
p33b = 0.82968823
p44b = 1.93489383
p66b = 2.*(p11b - p12b)
beta_params = {'a': np.array([[p11b, p12b, p13b, 0.,   0.,   0.],
                              [p12b, p11b, p13b, 0.,   0.,   0.],
                              [p13b, p13b, p33b, 0.,   0.,   0.],
                              [0.,   0.,   0.,   p44b, 0.,   0.],
                              [0.,   0.,   0.,   0.,   p44b, 0.],
                              [0.,   0.,   0.,   0.,   0.,   p66b]]),
               'b_1': np.zeros((6, 6)),
               'c_1': np.zeros((6, 6)),
               'b_2': np.zeros((6, 6)),
               'c_2': np.zeros((6, 6)),
               'd': np.zeros((6, 6))}

f = 0.005855603551066963
cell_parameters_alpha = np.array([4.9137*f, 4.9137*f, 5.4047*f, 90.0, 90.0, 120.0])
qtz_alpha_anisotropic = burnman.AnisotropicMineral(qtz_alpha,
                                                   cell_parameters=cell_parameters_alpha,
                                                   anisotropic_parameters=alpha_params,
                                                   psi_function=psi_func,
                                                   orthotropic=True)

qtz_antialpha_anisotropic = burnman.AnisotropicMineral(qtz_alpha,
                                                       cell_parameters=cell_parameters_alpha,
                                                       anisotropic_parameters=antialpha_params,
                                                       psi_function=psi_func,
                                                       orthotropic=True)

"""
a_0_Carpenter = 4.996 + 2.63*1.e-6*298.15
c_0_Carpenter = 5.464 - 5.63*1.e-6*298.15
f = 0.006033902623337949

qtz_beta_anisotropic = burnman.AnisotropicMineral(qtz_beta,
                                                  cell_parameters=np.array([a_0_Carpenter*f, a_0_Carpenter*f, a_0_Carpenter*f,
                                                                            90.0, 90.0, 120.0]),
                                                  anisotropic_parameters=beta_params,
                                                  psi_function=psi_func,
                                                  orthotropic=True)
"""

def psi_func_sol(V, Pth, X, params):
    nul = np.zeros((6, 6))
    nul2 = np.zeros((6, 6, 2))
    return (nul, nul, nul, nul2)



qtz_ss_anisotropic = burnman.AnisotropicSolution(name='quartz',
                                                 solution_type='function',
                                                 endmembers=[[qtz_alpha_anisotropic, '[Si]O2'],
                                                             [qtz_antialpha_anisotropic, '[Si]O2']],
                                                 excess_helmholtz_function=quartz_helmholtz_function,
                                                 master_cell_parameters=cell_parameters_alpha,
                                                 anisotropic_parameters={},
                                                 psi_excess_function=psi_func_sol,
                                                 dXdQ=np.array([[-1., 1.]]),
                                                 relaxed=True)
