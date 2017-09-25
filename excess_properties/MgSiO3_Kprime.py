import os
import sys
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
# Read in data from Kurnosov et al., 2017
# P (GPa) rho C11 C22 C33 C44 C55 C66 C12 C13 C23
data = np.loadtxt('Kurnosov_et_al_2017.dat',
                  unpack=True)

pressures = data[0]*1.e9
densities = data[1]*1.e3
C11, C11e, C22, C22e, C33, C33e, C44, C44e, C55, C55e, C66, C66e, C12, C12e, C13, C13e, C23, C23e = data[2:]*1.e9

# Find the best fitting bulk moduli at each pressure
class OrthorhombicModel(object):
    def __init__(self, mineral, delta_params, param_covariances, dof, noise_variance):
        self.m = mineral
        self.data = np.array([[0.]])
        self.delta_params = delta_params
        self.pcov = param_covariances
        self.dof = dof
        self.noise_variance = noise_variance
        
    def set_params(self, param_values):
        index_lists = [[(0, 0)], # C11
                       [(0, 1)], # C12
                       [(0, 2)], # C13
                       [(1, 1)], # C22
                       [(1, 2)], # C23
                       [(2, 2)], # C33
                       [(3, 3)], # C44
                       [(4, 4)], # C55
                       [(5, 5)]] # C66
        self.m.params['rho_0'] = param_values[0]
        self.m.params['stiffness_tensor_0'] = burnman.anisotropy.voigt_array_from_cijs(param_values[1:],
                                                                                       index_lists)
        burnman.Material.__init__(self.m) # need to reset cached values
        
    def get_params(self):
        params = [self.m.params['rho_0']]
        index_lists = [[(0, 0)], # C11
                       [(0, 1)], # C12
                       [(0, 2)], # C13
                       [(1, 1)], # C22
                       [(1, 2)], # C23
                       [(2, 2)], # C33
                       [(3, 3)], # C44
                       [(4, 4)], # C55
                       [(5, 5)]] # C66
        params.extend([self.m.params['stiffness_tensor_0'][idx[0]] for idx in index_lists])
        return params
    
    def function(self, x, flag):
        return None

KS = np.empty_like(pressures)
KSe = np.empty_like(pressures)
for i in range(len(pressures)):
    density = densities[i]
    
    cijs = np.array([C11[i], C12[i], C13[i],
                     C22[i], C23[i], C33[i],
                     C44[i], C55[i], C66[i]])
    params = np.array([density, C11[i], C12[i], C13[i],
                       C22[i], C23[i], C33[i],
                       C44[i], C55[i], C66[i]])
    param_covariances = np.diag(np.power(np.array([0., C11e[i], C12e[i], C13e[i],
                                                   C22e[i], C23e[i], C33e[i],
                                                   C44e[i], C55e[i], C66e[i]]), 2.))

    static_bdg = burnman.anisotropy.OrthorhombicMaterial(rho = density, cijs = cijs)

    static_bdg.params['stiffness_tensor_0']
    static_bdg.bulk_modulus_reuss
    
    model = OrthorhombicModel(mineral=static_bdg,
                              delta_params = params*1.e-5,
                              param_covariances = param_covariances,
                              dof = 100.,
                              noise_variance = 1.)
    var = burnman.nonlinear_fitting.confidence_prediction_bands(model = model,
                                                                x_array = np.array([[0.]]),
                                                                confidence_interval = 0.95,
                                                                f = lambda x: static_bdg.bulk_modulus_reuss)[0][0]
    KS[i] = static_bdg.bulk_modulus_reuss
    KSe[i] = np.sqrt(var)


for K in [KS-KSe, KS, KS+KSe]:
    spl = interp.splrep(pressures,K,k=3) # no smoothing, 3rd order spline
    dKdP = interp.splev(pressures,spl,der=1)

    plt.plot(pressures/K, 1./dKdP)



P, KS = np.loadtxt('Wentzcovitch_et_al_2004_MgSiO3_pv_KS.dat', unpack=True)

spl = interp.splrep(P,KS,k=3) # no smoothing, 3rd order spline
dKdP = interp.splev(P,spl,der=1)

plt.plot(P/KS, 1./dKdP)


plt.plot([0., 5./8.], [0., 5./8.])
plt.show()
