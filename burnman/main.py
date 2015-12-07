# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU GPL v2 or later.


import numpy as np
import scipy.integrate as integrate

import burnman.geotherm
import burnman.seismic
import burnman.averaging_schemes


class ElasticProperties(object):
    """
    Class that contains volume, density, and moduli. This is generated for
    different pressures and temperatures using
    :func:`calculate_moduli`. Several instances of elastic_properties can be
    averaged using :func:`average_moduli`.

    :var float V: volume :math:`[m^3]`
    :var float rho: density :math:`[kg/m^3]`
    :var float K: bulk modulus :math:`K` :math:`[Pa]`
    :var float G: shear modulus :math:`G` :math:`[Pa]`
    """

    def __init__(self, V=None, rho=None, K=None, G=None, fraction=None):
        """
        create an object with the given parameters.
        """
        self.V = V
        self.rho = rho
        self.K = K
        self.G = G
        self.fraction = fraction


def calculate_moduli(rock, pressures, temperatures):
    """
    Given a composite and a list of pressures :math:`[Pa]` and temperatures :math:`[K]`,
    calculate the elastic moduli and densities of the individual phases.

    :param burnman.abstract_material rock: this is a rock

    :type pressures: list of float
    :param pressures: list of pressures you want to evaluate the rock at. :math:`[Pa]`

    :type temperatures: list of float
    :param temperatures: list of temperatures you want to evaluate the rock at. :math:`[K]`

    :returns:
      answer -- an array of (n_evaluation_points by n_phases) of
      elastic_properties(), so the result is of the form
      answer[pressure_idx][phase_idx].V
    :rtype: list of list of :class:`burnman.elastic_properties`
    """

    answer = [[] for p in pressures]

    for idx in range(len(pressures)):
        rock.set_state(pressures[idx], temperatures[idx])
        (fractions,minerals) = rock.unroll()
        for (fraction,mineral) in zip(fractions,minerals):
            e = ElasticProperties()
            e.V = fraction * mineral.molar_volume()
            e.K = mineral.adiabatic_bulk_modulus()
            e.G = mineral.shear_modulus()
            e.rho = mineral.molar_mass() / mineral.molar_volume()
            e.fraction = fraction
            answer[idx].append(e)

    return answer

def average_moduli(moduli_list, averaging_scheme=burnman.averaging_schemes.VoigtReussHill):
    """
    Given an array of  of :class:`elastic_properties` of size: n_evaluation_points 
    by n_phases (e.g. that generated by
    :func:`calculate_moduli`), calculate the bulk properties, according to
    some averaging scheme. The averaging scheme defaults to Voigt-Reuss-Hill
    (see :class:`burnman.averaging_schemes.voigt_reuss_hill`), but the user
    may specify, Voigt, Reuss, the Hashin-Shtrikman bounds, or any user
    defined scheme that satisfies the interface
    :class:`burnman.averaging_schemes.averaging_scheme` (also see
    :doc:`averaging`).

    :type moduli_list: list of list of :class:`burnman.elastic_properties`
    :param moduli_list: List of end-member moduli to be averaged.

    :type averaging_scheme: :class:`burnman.averaging_schemes.averaging_scheme`
    :param averaging_scheme: Averaging scheme to use.

    :returns: A list of n_evaluation_points instances of elastic_properties.
    :rtype: list of :class:`burnman.elastic_properties`
    """
    n_pressures = len(moduli_list)
    result = [ElasticProperties() for i in range(n_pressures)]

    for idx in range(n_pressures):
        fractions = np.array([e.fraction for e in moduli_list[idx]])

        V_frac = np.array([m.V for m in moduli_list[idx]])
        K_ph = np.array([m.K for m in moduli_list[idx]])
        G_ph = np.array([m.G for m in moduli_list[idx]])
        rho_ph = np.array([m.rho for m in moduli_list[idx]])

        result[idx].V = sum(V_frac)
        result[idx].K = averaging_scheme.average_bulk_moduli(V_frac, K_ph, G_ph)
        result[idx].G = averaging_scheme.average_shear_moduli(V_frac, K_ph, G_ph)
        result[idx].rho = averaging_scheme.average_density(V_frac, rho_ph)
        result[idx].fraction = 1.0

    return result

def compute_velocities(moduli):
    """
    Given a list of elastic_properties, compute the seismic velocities :math:`V_p, V_s,` 
    and :math:`V_{\phi}` :math:`[m/s]` for each entry in the list.


    :type moduli: list of :class:`ElasticProperties`
    :param moduli: input elastic properties.

    :returns: lists of :math:`V_p, V_s,` and :math:`V_{\phi}` :math:`[m/s]`
    :rtype: list of float, list of float, list of float


    """
    mat_vs = np.ndarray(len(moduli))
    mat_vp = np.ndarray(len(moduli))
    mat_vphi = np.ndarray(len(moduli))

    for i in range(len(moduli)):

        mat_vs[i] = np.sqrt( moduli[i].G / moduli[i].rho)
        mat_vp[i] = np.sqrt( (moduli[i].K + 4./3.*moduli[i].G) / moduli[i].rho)
        mat_vphi[i] = np.sqrt( moduli[i].K / moduli[i].rho)

    return mat_vp, mat_vs, mat_vphi


def velocities_from_rock(rock, pressures, temperatures, averaging_scheme=burnman.averaging_schemes.VoigtReussHill()):
    """
    A function that rolls several steps into one: given a rock and a list of
    pressures and temperatures, it calculates the elastic moduli of the
    individual phases using calculate_moduli(), averages them using
    average_moduli(), and calculates the seismic velocities using
    compute_velocities().


    :param burnman.abstract_material rock: this is a rock

    :type pressures: list of float
    :param pressures: list of pressures you want to evaluate the rock at. :math:`[Pa]`

    :type temperatures: list of float
    :param temperatures: list of temperatures you want to evaluate the rock at. :math:`[K]`


    :type averaging_scheme: :class:`burnman.averaging_schemes.averaging_scheme`
    :param averaging_scheme: Averaging scheme to use.

    :returns: :math:`\\rho` :math:`[kg/m^3]` , :math:`V_p, V_s,` and :math:`V_{\phi}` :math:`[m/s]`, bulk modulus :math:`K` :math:`[Pa]`,shear modulus :math:`G` :math:`[Pa]`
    :rtype: lists of floats

    """
    moduli_list = calculate_moduli(rock, pressures, temperatures)
    moduli = average_moduli(moduli_list, averaging_scheme)
    mat_vp, mat_vs, mat_vphi = compute_velocities(moduli)
    mat_rho = np.array([m.rho for m in moduli])
    mat_K = np.array([m.K for m in moduli])
    mat_G = np.array([m.G for m in moduli])
    return mat_rho, mat_vp, mat_vs, mat_vphi, mat_K, mat_G

def depths_for_rock(rock,pressures, temperatures,averaging_scheme=burnman.averaging_schemes.VoigtReussHill()):
    """
    Function computes the self-consistent depths (to avoid using the PREM depth-pressure conversion) :cite:`Cammarano2013`.
    It is simplified by taking :math:`g` from PREM.

    :param burnman.abstract_material rock: this is a rock

    :type pressures: list of float
    :param pressures: list of pressures you want to evaluate the rock at. :math:`[Pa]`

    :type temperatures: list of float
    :param temperatures: list of temperatures you want to evaluate the rock at. :math:`[K]`

    :type averaging_scheme: :class:`burnman.averaging_schemes.averaging_scheme`
    :param averaging_scheme: Averaging scheme to use.

    :returns: depth :math:`[m]`
    :rtype: list of floats
    """
    moduli_list = calculate_moduli(rock, pressures, temperatures)
    moduli = average_moduli(moduli_list, averaging_scheme)
    mat_rho = np.array([m.rho for m in moduli])
    seismic_model = burnman.seismic.PREM()
    depthsref = np.array(list(map(seismic_model.depth,pressures)))
    pressref = np.zeros_like(pressures)
    g  = seismic_model.gravity(depthsref) # G for prem
    depths  = np.hstack((depthsref[0],depthsref[0]+integrate.cumtrapz(1./(g*mat_rho),pressures)))
    return depths

def pressures_for_rock(rock, depths, T0, averaging_scheme=burnman.averaging_schemes.VoigtReussHill()):
    """
    Function computes the self-consistent pressures (to avoid using the PREM depth pressure conversion) :cite:`Cammarano2013`.
    Only simplification is using :math:`g` from PREM.

    :param burnman.abstract_material rock: this is a rock

    :type depths: list of float
    :param depths: list of depths you want to evaluate the rock at :math:`[m]`.

    :type temperatures: list of float
    :param temperatures: list of temperatures you want to evaluate the rock at. :math:`[K]`.

    :type averaging_scheme: :class:`burnman.averaging_schemes.averaging_scheme`
    :param averaging_scheme: Averaging scheme to use.

    :returns: pressures :math:`[Pa]`
    :rtype: list of floats

    """
    # use PREM pressures as inital guestimate
    seismic_model = burnman.seismic.PREM()
    pressures,_,_,_,_ = seismic_model.evaluate_all_at(depths)
    pressref = np.zeros_like(pressures)
    #gets table with PREM gravities
    g = seismic_model.gravity(depths)
    #optimize pressures for this composition
    while nrmse(len(pressures),pressures,pressref)>1.e-6:
        # calculate density
        temperatures = burnman.geotherm.adiabatic(pressures,T0,rock)
        moduli_list = calculate_moduli(rock, pressures, temperatures)
        moduli = average_moduli(moduli_list, averaging_scheme)
        mat_rho = np.array([m.rho for m in moduli])
        # calculate pressures
        pressref = pressures
        pressures = np.hstack((pressref[0], pressref[0]+integrate.cumtrapz(g*mat_rho,depths)))
    return pressures

def apply_attenuation_correction(v_p,v_s,v_phi,Qs,Qphi):
    """
    Returns lists of corrected velocities  for a given :math:`Q_s, Q_{\\mu}` and :math:`Q_{\\phi}`



    :type Vp: list of float
    :param Vp: list of :math:`V_p`. :math:`[m/s]`
    :type Vs: list of float
    :param Vs: list of :math:`V_s`. :math:`[m/s]`
    :type Vphi: list of float
    :param Vphi: list of :math:`V_{\phi}`. :math:`[m/s]`

    :returns: :math:`V_p` ,:math:`V_s`, `V_{\phi}`. :math:`[m/s]`
    :rtype: list of floats

    """

    length = len(v_p)
    ret_v_p = np.zeros(length)
    ret_v_s = np.zeros(length)
    ret_v_phi = np.zeros(length)
    for i in range(length):
        ret_v_p[i],ret_v_s[i],ret_v_phi[i] = \
            burnman.seismic.attenuation_correction(v_p[i], v_s[i], v_phi[i],Qs,Qphi)

    return ret_v_p, ret_v_s, ret_v_phi


def compare_l2(depth,calc, obs):
    """

    Computes the L2 norm for N profiles at a time (assumed to be linear between points).

    .. math:: math does not work yet...
       \sum_{i=1}^{\\infty} x_{i}

    :type depths: array of float
    :param depths: depths. :math:`[m]`
    :type calc: list of arrays of float
    :param calc: N arrays calculated values, e.g. [mat_vs,mat_vphi]
    :type obs: list of arrays of float
    :param obs: N arrays of values (observed or calculated) to compare to , e.g. [seis_vs, seis_vphi]

    :returns: array of L2 norms of length N
    :rtype: array of floats
    """
    err=[]
    for l in range(len(calc)):
        err.append(l2(depth,calc[l],obs[l]))

    return err

def compare_chifactor(calc, obs):
    """

    Computes the chi factor for N profiles at a time. Assumes a 1% a priori uncertainty on the seismic model.


    :type calc: list of arrays of float
    :param calc: N arrays calculated values, e.g. [mat_vs,mat_vphi]
    :type obs: list of arrays of float
    :param obs: N arrays of values (observed or calculated) to compare to , e.g. [seis_vs, seis_vphi]

    :returns: error array of length N
    :rtype: array of floats
    """
    err=[]
    for l in range(len(calc)):
        err.append(chi_factor(calc[l],obs[l]))

    return err

def l2(x,funca,funcb):
    """

    Computes the L2 norm for one profile(assumed to be linear between points).

    :type x: array of float
    :param x: depths :math:`[m]`.
    :type funca: list of arrays of float
    :param funca: array calculated values
    :type funcb: list of arrays of float
    :param funcb: array of values (observed or calculated) to compare to

    :returns: L2 norm
    :rtype: array of floats
    """
    diff=np.array(funca-funcb)
    diff=diff*diff
    return integrate.trapz(diff,x)


def nrmse(x,funca,funcb):
    """
    Normalized root mean square error for one profile
    :type x: array of float
    :param x: depths in m.
    :type funca: list of arrays of float
    :param funca: array calculated values
    :type funcb: list of arrays of float
    :param funcb: array of values (observed or calculated) to compare to

    :returns: RMS error
    :rtype: array of floats

    """
    diff=np.array(funca-funcb)
    diff=diff*diff
    rmse=np.sqrt(np.sum(diff)/x)
    nrmse=rmse/(np.max(funca)-np.min(funca))
    return nrmse

def chi_factor(calc,obs):
    """
    :math:`\\chi` factor for one profile assuming 1% uncertainty on the reference model (obs)
    :type calc: list of arrays of float
    :param calc: array calculated values
    :type obs: list of arrays of float
    :param obs: array of reference values to compare to

    :returns: :math:`\\chi` factor
    :rtype: array of floats

    """

    err=np.empty_like(calc)
    for i in range(len(calc)):
        err[i]=pow((calc[i]-obs[i])/(0.01*np.mean(obs)),2.)

    err_tot=np.sum(err)/len(err)

    return err_tot
