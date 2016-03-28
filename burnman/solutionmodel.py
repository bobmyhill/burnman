# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import numpy as np
import warnings
from .processchemistry import *
from . import constants
from scipy.optimize import fsolve

"""
kronecker delta function for integers
"""
kd = lambda x, y: 1 if x == y else 0


class SolutionModel(object):

    """
    This is the base class for a solution model,  intended for use
    in defining solid solutions and performing thermodynamic calculations
    on them.  All minerals of type :class:`burnman.SolidSolution` use
    a solution model for defining how the endmembers in the solid solution
    interact.

    A user wanting a new solution model should define the functions below.
    In the base class all of these return zero, so if the solution model
    does not implement them, they essentially have no effect, and
    then the Gibbs free energy and molar volume of a solid solution are
    just the weighted arithmetic averages of the different endmember values.
    """

    def __init__(self):
        """
        Does nothing.
        """
        pass

    def excess_gibbs_free_energy(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess Gibbs free energy of the solution.
        The base class implementation assumes that the excess gibbs
        free energy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        G_excess : float
            The excess Gibbs free energy
        """
        return np.dot(np.array(molar_fractions), self.excess_partial_gibbs_free_energies(pressure, temperature, molar_fractions))

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess Gibbs free energy for each endmember of the solution.
        The base class implementation assumes that the excess gibbs
        free energy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        partial_G_excess : numpy array
            The excess Gibbs free energy of each endmember
        """
        return np.empty_like(np.array(molar_fractions))

    def excess_volume(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess volume of the solution.
        The base class implementation assumes that the excess volume is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        V_excess : float
            The excess volume of the solution
        """
        return 0.0

    def excess_enthalpy(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess enthalpy of the solution.
        The base class implementation assumes that the excess enthalpy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        H_excess : float
            The excess enthalpy of the solution
        """
        return 0.0

    def excess_entropy(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess entropy of the solution.
        The base class implementation assumes that the excess entropy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        S_excess : float
            The excess entropy of the solution
        """
        return 0.0

    def set_state(self, pressure, temperature):
        return None


class IdealSolution (SolutionModel):

    """
    A very simple class representing an ideal solution model.
    Calculate the excess gibbs free energy due to configurational
    entropy, all the other excess terms return zero.
    """

    def __init__(self, endmembers):
        self.n_endmembers = len(endmembers)
        self.formulas = [e[1] for e in endmembers]

        # Process solid solution chemistry
        self.solution_formulae, self.n_sites, self.sites, self.n_occupancies, self.endmember_occupancies, self.site_multiplicities = \
            process_solution_chemistry(self.formulas)

        self._calculate_endmember_configurational_entropies()

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        return self._ideal_excess_partial_gibbs(temperature, molar_fractions)

    def _calculate_endmember_configurational_entropies(self):
        self.endmember_configurational_entropies = np.zeros(
            shape=(self.n_endmembers))
        for idx, endmember_occupancy in enumerate(self.endmember_occupancies):
            for occ in range(self.n_occupancies):
                if endmember_occupancy[occ] > 1e-10:
                    self.endmember_configurational_entropies[idx] = \
                        self.endmember_configurational_entropies[idx] - \
                        constants.gas_constant * self.site_multiplicities[
                            occ] * endmember_occupancy[occ] * np.log(endmember_occupancy[occ])

    def _endmember_configurational_entropy_contribution(self, molar_fractions):
        return np.dot(molar_fractions, self.endmember_configurational_entropies)

    def _configurational_entropy(self, molar_fractions):
        site_occupancies = np.dot(molar_fractions, self.endmember_occupancies)
        conf_entropy = 0
        for idx, occupancy in enumerate(site_occupancies):
            if occupancy > 1e-10:
                conf_entropy = conf_entropy - constants.gas_constant * \
                    occupancy * \
                    self.site_multiplicities[idx] * np.log(occupancy)

        return conf_entropy

    def _ideal_excess_partial_gibbs(self, temperature, molar_fractions):
        return constants.gas_constant * temperature * self._log_ideal_activities(molar_fractions)

    def _log_ideal_activities(self, molar_fractions):
        site_occupancies = np.dot(molar_fractions, self.endmember_occupancies)
        lna = np.empty(shape=(self.n_endmembers))

        for e in range(self.n_endmembers):
            lna[e] = 0.0
            for occ in range(self.n_occupancies):
                if self.endmember_occupancies[e][occ] > 1e-10 and site_occupancies[occ] > 1e-10:
                    lna[e] = lna[e] + self.endmember_occupancies[e][occ] * \
                        self.site_multiplicities[
                            occ] * np.log(site_occupancies[occ])

            normalisation_constant = self.endmember_configurational_entropies[
                e] / constants.gas_constant
            lna[e] = lna[e] + self.endmember_configurational_entropies[
                e] / constants.gas_constant
        return lna

    def _ideal_activities(self, molar_fractions):
        site_occupancies = np.dot(molar_fractions, self.endmember_occupancies)
        activities = np.empty(shape=(self.n_endmembers))

        for e in range(self.n_endmembers):
            activities[e] = 1.0
            for occ in range(self.n_occupancies):
                if self.endmember_occupancies[e][occ] > 1e-10:
                    activities[e] = activities[e] * np.power(
                        site_occupancies[occ], self.endmember_occupancies[e][occ] * self.site_multiplicities[occ])
            normalisation_constant = np.exp(
                self.endmember_configurational_entropies[e] / constants.gas_constant)
            activities[e] = normalisation_constant * activities[e]
        return activities

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        return np.ones_like(molar_fractions)

    def activities(self, pressure, temperature, molar_fractions):
        return self._ideal_activities(molar_fractions)


class AsymmetricRegularSolution (IdealSolution):

    """
    Solution model implementing the asymmetric regular solution model formulation (Holland and Powell, 2003)
    """

    def __init__(self, endmembers, alphas, enthalpy_interaction, volume_interaction=None, entropy_interaction=None):

        self.n_endmembers = len(endmembers)

        # Create array of van Laar parameters
        self.alpha = np.array(alphas)

        # Create 2D arrays of interaction parameters
        self.Wh = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Ws = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Wv = np.zeros(shape=(self.n_endmembers, self.n_endmembers))

        # setup excess enthalpy interaction matrix
        for i in range(self.n_endmembers):
            for j in range(i + 1, self.n_endmembers):
                self.Wh[i][j] = 2. * enthalpy_interaction[
                    i][j - i - 1] / (self.alpha[i] + self.alpha[j])

        if entropy_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    self.Ws[i][j] = 2. * entropy_interaction[
                        i][j - i - 1] / (self.alpha[i] + self.alpha[j])

        if volume_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    self.Wv[i][j] = 2. * volume_interaction[
                        i][j - i - 1] / (self.alpha[i] + self.alpha[j])

        # initialize ideal solution model
        IdealSolution.__init__(self, endmembers)

    def _phi(self, molar_fractions):
        phi = np.array([self.alpha[i] * molar_fractions[i]
                       for i in range(self.n_endmembers)])
        phi = np.divide(phi, np.sum(phi))
        return phi

    def _non_ideal_interactions(self, molar_fractions):
        # -sum(sum(qi.qj.Wij*)
        # equation (2) of Holland and Powell 2003
        phi = self._phi(molar_fractions)

        q = np.zeros(len(molar_fractions))
        Hint = np.zeros(len(molar_fractions))
        Sint = np.zeros(len(molar_fractions))
        Vint = np.zeros(len(molar_fractions))

        for l in range(self.n_endmembers):
            q = np.array([kd(i, l) - phi[i] for i in range(self.n_endmembers)])

            Hint[l] = 0. - self.alpha[l] * np.dot(q, np.dot(self.Wh, q))
            Sint[l] = 0. - self.alpha[l] * np.dot(q, np.dot(self.Ws, q))
            Vint[l] = 0. - self.alpha[l] * np.dot(q, np.dot(self.Wv, q))

        return Hint, Sint, Vint

    def _non_ideal_excess_partial_gibbs(self, pressure, temperature, molar_fractions):
        Hint, Sint, Vint = self._non_ideal_interactions(molar_fractions)
        return Hint - temperature * Sint + pressure * Vint

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs(
            self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(
            pressure, temperature, molar_fractions)
        return ideal_gibbs + non_ideal_gibbs

    def excess_volume(self, pressure, temperature, molar_fractions):
        phi = self._phi(molar_fractions)
        V_excess = np.dot(self.alpha.T, molar_fractions) * np.dot(
            phi.T, np.dot(self.Wv, phi))
        return V_excess

    def excess_entropy(self, pressure, temperature, molar_fractions):
        phi = self._phi(molar_fractions)
        S_conf = -constants.gas_constant * \
            np.dot(IdealSolution._log_ideal_activities(
                self, molar_fractions), molar_fractions)
        S_excess = np.dot(self.alpha.T, molar_fractions) * np.dot(
            phi.T, np.dot(self.Ws, phi))
        return S_conf + S_excess

    def excess_enthalpy(self, pressure, temperature, molar_fractions):
        phi = self._phi(molar_fractions)
        H_excess = np.dot(self.alpha.T, molar_fractions) * np.dot(
            phi.T, np.dot(self.Wh, phi))
        return H_excess + pressure * self.excess_volume(pressure, temperature, molar_fractions)

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        if temperature > 1.e-10:
            return np.exp(self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions) / (constants.gas_constant * temperature))
        else:
            raise Exception("Activity coefficients not defined at 0 K.")

    def activities(self, pressure, temperature, molar_fractions):
        return IdealSolution.activities(self, pressure, temperature, molar_fractions) * self.activity_coefficients(pressure, temperature, molar_fractions)


class SymmetricRegularSolution (AsymmetricRegularSolution):

    """
    Solution model implementing the symmetric regular solution model
    """

    def __init__(self, endmembers, enthalpy_interaction, volume_interaction=None, entropy_interaction=None):
        alphas = np.ones(len(endmembers))
        AsymmetricRegularSolution.__init__(
            self, endmembers, alphas, enthalpy_interaction, volume_interaction, entropy_interaction)


class SubregularSolution (IdealSolution):

    """
    Solution model implementing the subregular solution model formulation (Helffrich and Wood, 1989)
    """

    def __init__(self, endmembers, enthalpy_interaction, volume_interaction=None, entropy_interaction=None):

        self.n_endmembers = len(endmembers)

        # Create 2D arrays of interaction parameters
        self.Wh = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Ws = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Wv = np.zeros(shape=(self.n_endmembers, self.n_endmembers))

        # setup excess enthalpy interaction matrix
        for i in range(self.n_endmembers):
            for j in range(i + 1, self.n_endmembers):
                self.Wh[i][j] = enthalpy_interaction[i][j - i - 1][0]
                self.Wh[j][i] = enthalpy_interaction[i][j - i - 1][1]

        if entropy_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    self.Ws[i][j] = entropy_interaction[i][j - i - 1][0]
                    self.Ws[j][i] = entropy_interaction[i][j - i - 1][1]

        if volume_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    self.Wv[i][j] = volume_interaction[i][j - i - 1][0]
                    self.Wv[j][i] = volume_interaction[i][j - i - 1][1]

        # initialize ideal solution model
        IdealSolution.__init__(self, endmembers)

    def _non_ideal_function(self, W, molar_fractions):
        # equation (6') of Helffrich and Wood, 1989
        n = len(molar_fractions)
        RTlny = np.zeros(n)
        for l in range(n):
            val = 0.
            for i in range(n):
                if i != l:
                    val += 0.5 * molar_fractions[i] * (W[l][i] * (1 - molar_fractions[l] + molar_fractions[i] + 2. * molar_fractions[l] * (molar_fractions[l] - molar_fractions[i] - 1)) + W[
                                                       i][l] * (1. - molar_fractions[l] - molar_fractions[i] - 2. * molar_fractions[l] * (molar_fractions[l] - molar_fractions[i] - 1)))
                    for j in range(i + 1, n):
                        if j != l:
                            val += molar_fractions[i] * molar_fractions[j] * (
                                W[i][j] * (molar_fractions[i] - molar_fractions[j] - 0.5) + W[j][i] * (molar_fractions[j] - molar_fractions[i] - 0.5))
            RTlny[l] = val
        return RTlny

    def _non_ideal_interactions(self, molar_fractions):
        # equation (6') of Helffrich and Wood, 1989
        Hint = self._non_ideal_function(self.Wh, molar_fractions)
        Sint = self._non_ideal_function(self.Ws, molar_fractions)
        Vint = self._non_ideal_function(self.Wv, molar_fractions)
        return Hint, Sint, Vint

    def _non_ideal_excess_partial_gibbs(self, pressure, temperature, molar_fractions):
        Hint, Sint, Vint = self._non_ideal_interactions(molar_fractions)
        return Hint - temperature * Sint + pressure * Vint

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs(
            self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(
            pressure, temperature, molar_fractions)
        return ideal_gibbs + non_ideal_gibbs

    def excess_volume(self, pressure, temperature, molar_fractions):
        V_excess = np.dot(
            molar_fractions, self._non_ideal_function(self.Wv, molar_fractions))
        return V_excess

    def excess_entropy(self, pressure, temperature, molar_fractions):
        S_conf = -constants.gas_constant * \
            np.dot(IdealSolution._log_ideal_activities(
                self, molar_fractions), molar_fractions)
        S_excess = np.dot(
            molar_fractions, self._non_ideal_function(self.Ws, molar_fractions))
        return S_conf + S_excess

    def excess_enthalpy(self, pressure, temperature, molar_fractions):
        H_excess = np.dot(
            molar_fractions, self._non_ideal_function(self.Wh, molar_fractions))
        return H_excess + pressure * self.excess_volume(pressure, temperature, molar_fractions)

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        if temperature > 1.e-10:
            return np.exp(self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions) / (constants.gas_constant * temperature))
        else:
            raise Exception("Activity coefficients not defined at 0 K.")

    def activities(self, pressure, temperature, molar_fractions):
        return IdealSolution.activities(self, pressure, temperature, molar_fractions) * self.activity_coefficients(pressure, temperature, molar_fractions)



class FullSubregularSolution (IdealSolution):

    """
    Solution model implementing the subregular solution model formulation (Helffrich and Wood, 1989)
    """

    def __init__(self, endmembers, P0, T0, n, energy_interaction=None, volume_interaction=None, kprime_interaction=None, thermal_pressure_interaction=None):
        
        self.n_endmembers = len(endmembers)
        self.P_0 = P0
        self.T_0 = T0
        self.n_atoms = n

        self.ideal_std = [[0. for i in xrange(self.n_endmembers)] for j in xrange(self.n_endmembers)]
        self.nonideal_std = [[0. for i in xrange(self.n_endmembers)] for j in xrange(self.n_endmembers)]
        
        # Create 2D arrays of interaction parameters
        self.We = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Ws = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Wv = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        
        We = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        Wv = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        Wkprime = 7.*np.ones(shape=(self.n_endmembers, self.n_endmembers))
        Wp = np.ones(shape=(self.n_endmembers, self.n_endmembers))

        # setup excess enthalpy interaction matrix
        if energy_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    We[i][j] = energy_interaction[i][j - i - 1][0]
                    We[j][i] = energy_interaction[i][j - i - 1][1]
        
        if volume_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    Wv[i][j] = volume_interaction[i][j - i - 1][0]
                    Wv[j][i] = volume_interaction[i][j - i - 1][1]

        if kprime_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    Wkprime[i][j] = kprime_interaction[i][j - i - 1][0]
                    Wkprime[j][i] = kprime_interaction[i][j - i - 1][1]

        if thermal_pressure_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    Wp[i][j] = thermal_pressure_interaction[i][j - i - 1][0]
                    Wp[j][i] = thermal_pressure_interaction[i][j - i - 1][1]

        # Find standard properties
        for i in range(self.n_endmembers):
            endmembers[i][0].set_state(self.P_0, self.T_0)
            
        # Ideal properties
        for i in range(self.n_endmembers):
            for j in range(i + 1, self.n_endmembers):
                V0 = 0.5*(endmembers[i][0].V + endmembers[j][0].V)
                K0 = 2.0*V0/(endmembers[i][0].V/endmembers[i][0].K_T \
                             + endmembers[j][0].V/endmembers[j][0].K_T)
                # Fill dictionary
                self.ideal_std[i][j]= {
                    'V_0': V0,
                    'K_0': K0 }
                self.ideal_std[j][i] = self.ideal_std[i][j]

        # Nonideal properties
        for i in range(self.n_endmembers):
            for j in range(self.n_endmembers):
                if i != j:
                    V0 = self.ideal_std[i][j]['V_0']
                    V0ni = V0 + Wv[i][j]/4.
                    Kprime = Wkprime[i][j]
                    self.nonideal_std[i][j] = {
                        'energy_xs': We[i][j],
                        'V_0': V0ni,
                        'Kprime_xs': Kprime,
                        'f_Pth': Wp[i][j] }
        
        # initialize ideal solution model
        IdealSolution.__init__(self, endmembers)

    def set_state(self, pressure, temperature, endmembers):
        def _findPideal(P, V, T0, m1, m2):
            V_m1 = m1.method.volume(P[0], T0, m1.params)
            V_m2 = m2.method.volume(P[0], T0, m2.params)
            return V - 0.5*(V_m1 + V_m2)
    
        def _volume_excess(Vnonideal0, Videal0, Kideal0, Kprime, pressure):
            Vexcess0 = Vnonideal0 - Videal0
            Knonideal0 = Kideal0*np.power(Videal0/Vnonideal0, Kprime)
            bideal = Kprime/Kideal0
            bnonideal = Kprime/Knonideal0
            c = 1./Kprime
            return Vnonideal0*np.power((1.+bnonideal*pressure), -c) \
                - Videal0*np.power((1.+bideal*(pressure)), -c)

        def _intVdP_excess(Vnonideal0, Videal0, Kideal0, Kprime, pressure):
            if np.abs(pressure) < 1.:
                pressure = 1.
            Vexcess0 = Vnonideal0 - Videal0
            Knonideal0 = Kideal0*np.power(Videal0/Vnonideal0, Kprime)
            bideal = Kprime/Kideal0
            bnonideal = Kprime/Knonideal0
            c = 1./Kprime
            return -pressure*(Vnonideal0*np.power((1.+bnonideal*pressure), 1.-c) \
                              / (bnonideal*(c - 1.)*pressure) \
                              - Videal0*np.power((1.+bideal*pressure), 1.-c) \
                              /(bideal*(c - 1.)*pressure))
        
        def _V_Pth_ideal(V0_ideal, P_0, T_0, T, f_Pth, m1, m2):
            # First, heres Pth (\int aK_T dT | V) for the ideal phase
            # when V=V0 and T=T1 
            P_V0ideal_T = fsolve(_findPideal, [P_0 + 5.e6*(T - T_0)], args=(V0_ideal, T, m1, m2))[0]
            Pth_V0ideal_T =  P_V0ideal_T - P_0
            
            # Make the assumption that Pth(V0, T)_nonideal = Pth(V0, T)_ideal*f_Pth 
            Pth_V0nonideal_T = Pth_V0ideal_T*f_Pth
            V_V0nonideal_ideal = 0.5*(m1.method.volume(P_0 + Pth_V0nonideal_T, T, m1.params) \
                                      + m2.method.volume(P_0 + Pth_V0nonideal_T, T, m2.params))
            return Pth_V0nonideal_T, V_V0nonideal_ideal
                    
        for i in range(self.n_endmembers):
            for j in range(self.n_endmembers):
                if i != j:
                    # Properties at pressure
                    Pth_V0nonideal_T, V_V0nonideal_ideal = _V_Pth_ideal(self.ideal_std[i][j]['V_0'],
                                                                        self.P_0, self.T_0, temperature,
                                                                        self.nonideal_std[i][j]['f_Pth'],
                                                                        endmembers[i][0], endmembers[j][0])
                    
                    
                
                    # Make the further assumption that the form of the excess volume curve is temperature independent
                    self.Wv[i][j] = 4.*_volume_excess(self.nonideal_std[i][j]['V_0'],
                                                      V_V0nonideal_ideal,
                                                      self.ideal_std[i][j]['K_0'],
                                                      self.nonideal_std[i][j]['Kprime_xs'],
                                                      pressure - Pth_V0nonideal_T - self.P_0)

                    # Calculate contributions to the gibbs free energy
                    # 1. The isothermal path along T0 from P0 to infinite pressure 
                    Gxs_T0 = -4.*_intVdP_excess(self.nonideal_std[i][j]['V_0'],
                                              self.ideal_std[i][j]['V_0'],
                                              self.ideal_std[i][j]['K_0'],
                                              self.nonideal_std[i][j]['Kprime_xs']
                                              , 0.)
                    # 2. The isobaric path at infinite pressure from T0 to T has no excess contribution
                    # 3. The isothermal path from infinite pressure down to P, T
                    Gxs_T = 4.*_intVdP_excess(self.nonideal_std[i][j]['V_0'],
                                           V_V0nonideal_ideal,
                                           self.ideal_std[i][j]['K_0'],
                                           self.nonideal_std[i][j]['Kprime_xs'],
                                           pressure - Pth_V0nonideal_T - self.P_0)

                    # gibbs at (P_0, T_0+1)
                    Pth_V0nonideal_T01, V_V0nonideal_ideal01 = _V_Pth_ideal(self.ideal_std[i][j]['V_0'],
                                                                            self.P_0, self.T_0, self.T_0+1.,
                                                                            self.nonideal_std[i][j]['f_Pth'],
                                                                            endmembers[i][0], endmembers[j][0])
                    Gxs_T01 = 4.*_intVdP_excess(self.nonideal_std[i][j]['V_0'],
                                             V_V0nonideal_ideal01,
                                             self.ideal_std[i][j]['K_0'],
                                             self.nonideal_std[i][j]['Kprime_xs'],
                                             - Pth_V0nonideal_T01)
                    
                    # gibbs at (P, T+1)
                    Pth_V0nonideal_T1, V_V0nonideal_ideal1 = _V_Pth_ideal(self.ideal_std[i][j]['V_0'],
                                                                          self.P_0, self.T_0, temperature+1.,
                                                                          self.nonideal_std[i][j]['f_Pth'],
                                                                          endmembers[i][0], endmembers[j][0])
                    Gxs_T1 = 4.*_intVdP_excess(self.nonideal_std[i][j]['V_0'],
                                            V_V0nonideal_ideal1,
                                            self.ideal_std[i][j]['K_0'], 
                                            self.nonideal_std[i][j]['Kprime_xs'],
                                            pressure - Pth_V0nonideal_T1 - self.P_0)

                    Gxs0 = self.nonideal_std[i][j]['energy_xs'] \
                           + (self.T_0*(Gxs_T01 - Gxs_T0) - self.P_0*(self.nonideal_std[i][j]['V_0'] - self.ideal_std[i][j]['V_0']))
                    Gxs = 0. + Gxs_T0 + Gxs_T
                                        
                    self.Ws[i][j] = Gxs_T - Gxs_T1
                    self.We[i][j] = Gxs + temperature*self.Ws[i][j] - pressure*self.Wv[i][j]
                    
    
    def _non_ideal_function(self, W, molar_fractions):
        # equation (6') of Helffrich and Wood, 1989
        n = len(molar_fractions)
        RTlny = np.zeros(n)
        for l in range(n):
            val = 0.
            for i in range(n):
                if i != l:
                    val += 0.5 * molar_fractions[i] * (W[l][i] * (1 - molar_fractions[l] + molar_fractions[i] + 2. * molar_fractions[l] * (molar_fractions[l] - molar_fractions[i] - 1)) + W[
                                                       i][l] * (1. - molar_fractions[l] - molar_fractions[i] - 2. * molar_fractions[l] * (molar_fractions[l] - molar_fractions[i] - 1)))
                    for j in range(i + 1, n):
                        if j != l:
                            val += molar_fractions[i] * molar_fractions[j] * (
                                W[i][j] * (molar_fractions[i] - molar_fractions[j] - 0.5) + W[j][i] * (molar_fractions[j] - molar_fractions[i] - 0.5))
            RTlny[l] = val
        return RTlny

    def _non_ideal_interactions(self, molar_fractions):
        # equation (6') of Helffrich and Wood, 1989
        Eint = self._non_ideal_function(self.We, molar_fractions)
        Sint = self._non_ideal_function(self.Ws, molar_fractions)
        Vint = self._non_ideal_function(self.Wv, molar_fractions)
        return Eint, Sint, Vint

    def _non_ideal_excess_partial_gibbs(self, pressure, temperature, molar_fractions):
        Eint, Sint, Vint = self._non_ideal_interactions(molar_fractions)
        return Eint - temperature * Sint + pressure * Vint

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs(
            self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(
            pressure, temperature, molar_fractions)
        return ideal_gibbs + non_ideal_gibbs

    def excess_volume(self, pressure, temperature, molar_fractions):
        V_excess = np.dot(
            molar_fractions, self._non_ideal_function(self.Wv, molar_fractions))
        return V_excess

    def excess_entropy(self, pressure, temperature, molar_fractions):
        S_conf = -constants.gas_constant * \
            np.dot(IdealSolution._log_ideal_activities(
                self, molar_fractions), molar_fractions)
        S_excess = np.dot(
            molar_fractions, self._non_ideal_function(self.Ws, molar_fractions))
        return S_conf + S_excess

    def excess_enthalpy(self, pressure, temperature, molar_fractions):
        E_excess = np.dot(
            molar_fractions, self._non_ideal_function(self.We, molar_fractions))
        return E_excess + pressure * self.excess_volume(pressure, temperature, molar_fractions)

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        if temperature > 1.e-10:
            return np.exp(self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions) / (constants.gas_constant * temperature))
        else:
            raise Exception("Activity coefficients not defined at 0 K.")

    def activities(self, pressure, temperature, molar_fractions):
        return IdealSolution.activities(self, pressure, temperature, molar_fractions) * self.activity_coefficients(pressure, temperature, molar_fractions)
