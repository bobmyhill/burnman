import matplotlib.pyplot as plt
import numpy as np

palettes = ['seaborn-darkgrid', 'seaborn-notebook', 'classic', 'seaborn-ticks', 'grayscale',
            'bmh', 'seaborn-talk', 'dark_background', 'ggplot', 'fivethirtyeight',
            '_classic_test', 'seaborn-colorblind', 'seaborn-deep', 'seaborn-whitegrid', 'seaborn',
            'seaborn-poster', 'seaborn-bright', 'seaborn-muted', 'seaborn-paper', 'seaborn-white',
            'seaborn-pastel', 'seaborn-dark', 'seaborn-dark-palette']

plt.style.use(palettes[6])
plt.rcParams['figure.figsize'] = 12, 6 # inches
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'


from scipy.optimize import brentq

class dummy_solution(object):
    def __init__(self):
        self.molar_mass = 0.
        self.rho = 0.
        self.V = 0.
        self.K_T = 0.
        self.S = 0.
        self.heat_capacity_v = 0.
        self.alpha = 0.
        self.gr = 0.
        self.K_S = 0.
        self.heat_capacity_p = 0.
        self.bulk_sound_velocity = 0.
        self.helmholtz = 0.
        self.gibbs = 0.
        self.internal_energy = 0.
        #self.dFdxV = 0.
        self.partial_pressures = (0., 0.)
        
def _deltaV(pressure, volume, temperature, m):
    m.set_state(pressure, temperature)
    return volume - m.V

    
def _deltaP(volume, pressure, temperature, p_xs, x_a, a, b):
    p_a = brentq(_deltaV,
                  a.method.pressure(temperature, volume, a.params) - 1.e9,
                  a.method.pressure(temperature, volume, a.params) + 1.e9,
                  args=(volume, temperature, a))
    p_b = brentq(_deltaV,
                  b.method.pressure(temperature, volume, b.params) - 1.e9,
                  b.method.pressure(temperature, volume, b.params) + 1.e9,
                  args=(volume, temperature, b))
    return pressure - (x_a*p_a + (1. - x_a)*p_b) - p_xs

def solution(pressure, temperature, F_xs, p_xs, x_a, a, b, cluster_size=1.):

    a.set_state(1.e5, 300.)
    b.set_state(1.e5, 300.)

    F_linear = x_a*a.helmholtz + (1. - x_a)*b.helmholtz
    #dFdxV_linear = (a.helmholtz - b.helmholtz)
    
    brentq(_deltaP, min([a.V, b.V])-1.e-9, max([a.V, b.V])+20.e-9, args=(1.e5, 300., p_xs, x_a, a, b))

    F_disordered_mixing = ((x_a*a.helmholtz + (1. - x_a)*b.helmholtz - F_linear) *
                           (cluster_size - 1.)/cluster_size )

    #dFdxV_disordered_mixing = (((a.helmholtz - b.helmholtz) - dFdxV_linear) *
    #                           (cluster_size - 1.)/cluster_size )
    
    a.set_state(pressure, temperature)
    b.set_state(pressure, temperature)
    
    s = dummy_solution()
    s.V = brentq(_deltaP, min([a.V, b.V])-1.e-9, max([a.V, b.V])+20.e-9, args=(pressure, temperature, p_xs, x_a, a, b))
    # p_xs is an excess pressure at constant volume across the solution, so at constant pressure across the solution, a positive excess pressure implies a positive excess volume.

    s.molar_mass = x_a*a.molar_mass + (1. - x_a)*b.molar_mass
    s.rho = s.molar_mass/s.V
    
    s.partial_pressures = (a.pressure, b.pressure)
    s.S = x_a*a.S + (1. - x_a)*b.S
    s.K_T = x_a*a.K_T + (1. - x_a)*b.K_T
    s.heat_capacity_v = x_a*a.heat_capacity_v + (1. - x_a)*b.heat_capacity_v
    s.alpha = 1./s.K_T*(x_a*a.alpha*a.K_T + (1 - x_a)*b.alpha*b.K_T)
    s.gr = s.alpha*s.K_T*s.V/s.heat_capacity_v
    s.K_S = s.K_T*(1. + s.alpha*s.gr*temperature)
    s.bulk_sound_velocity = np.sqrt(s.K_S/s.rho)
    s.heat_capacity_p = s.heat_capacity_v*s.K_T/s.K_S
    s.helmholtz = ((x_a*a.helmholtz + (1. - x_a)*b.helmholtz) -
                   F_disordered_mixing + 
                   F_xs - p_xs*s.V)
    #s.dFdxV = (a.helmholtz - b.helmholtz) - dFdxV_disordered_mixing # ONLY GOOD IF F_XS and P_XS = 0.

    
    s.gibbs = s.helmholtz + pressure*s.V
    s.internal_energy = s.helmholtz + temperature*s.S
    s.H = s.internal_energy + pressure*s.V
    
    a.set_state(pressure, temperature)
    b.set_state(pressure, temperature)
    return s
