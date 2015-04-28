# BurnMan - a lower mantle toolkit
# Copyright (C) 2012-2014, Myhill, R., Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

import numpy as np
import warnings
import burnman
from burnman.processchemistry import *
import constants

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

    def excess_gibbs_free_energy( self, pressure, temperature, molar_fractions):
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
        return np.dot(np.array(molar_fractions), self.excess_partial_gibbs_free_energies( pressure, temperature, molar_fractions))

    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fractions):
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
        return np.empty_like( np.array(molar_fractions) )

    def excess_volume( self, pressure, temperature, molar_fractions):
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

    def excess_enthalpy( self, pressure, temperature, molar_fractions):
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

    def excess_entropy( self, pressure, temperature, molar_fractions):
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

    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fractions ):
        return self._ideal_excess_partial_gibbs( temperature, molar_fractions )

    def _calculate_endmember_configurational_entropies( self ):
        self.endmember_configurational_entropies=np.zeros(shape=(self.n_endmembers))
        for idx, endmember_occupancy in enumerate(self.endmember_occupancies):
            for occ in range(self.n_occupancies):
                if endmember_occupancy[occ] > 1e-10:
                    self.endmember_configurational_entropies[idx] = \
                        self.endmember_configurational_entropies[idx] - \
                        constants.gas_constant*self.site_multiplicities[occ]*endmember_occupancy[occ]*np.log(endmember_occupancy[occ])

    def _endmember_configurational_entropy_contribution(self, molar_fractions):
        return np.dot(molar_fractions, self.endmember_configurational_entropies)

    def _configurational_entropy (self, molar_fractions):
        site_occupancies=np.dot(molar_fractions, self.endmember_occupancies)
        conf_entropy=0
        for idx, occupancy in enumerate(site_occupancies):
            if occupancy > 1e-10:
                conf_entropy=conf_entropy-constants.gas_constant*occupancy*self.site_multiplicities[idx]*np.log(occupancy)

        return conf_entropy


    def _ideal_excess_partial_gibbs( self, temperature, molar_fractions ):
        return  constants.gas_constant*temperature * self._log_ideal_activities(molar_fractions)

    def _log_ideal_activities ( self, molar_fractions ):
        site_occupancies=np.dot(molar_fractions, self.endmember_occupancies)
        lna=np.empty(shape=(self.n_endmembers))

        for e in range(self.n_endmembers):
            lna[e]=0.0
            for occ in range(self.n_occupancies):
                if self.endmember_occupancies[e][occ] > 1e-10 and site_occupancies[occ] > 1e-10:
                    lna[e]=lna[e] + self.endmember_occupancies[e][occ]*self.site_multiplicities[occ]*np.log(site_occupancies[occ])

            normalisation_constant=self.endmember_configurational_entropies[e]/constants.gas_constant
            lna[e]=lna[e] + self.endmember_configurational_entropies[e]/constants.gas_constant
        return lna

    def _ideal_activities ( self, molar_fractions ):
        site_occupancies=np.dot(molar_fractions, self.endmember_occupancies)
        activities=np.empty(shape=(self.n_endmembers))

        for e in range(self.n_endmembers):
            activities[e]=1.0
            for occ in range(self.n_occupancies):
                if self.endmember_occupancies[e][occ] > 1e-10:
                    activities[e]=activities[e]*np.power(site_occupancies[occ],self.endmember_occupancies[e][occ]*self.site_multiplicities[occ])
            normalisation_constant=np.exp(self.endmember_configurational_entropies[e]/R)
            activities[e]=normalisation_constant*activities[e]
        return activities

 
class AsymmetricRegularSolution (IdealSolution):
    """
    Solution model implementing the asymmetric regular solution model formulation (Holland and Powell, 2003)
    """

    def __init__( self, endmembers, alphas, enthalpy_interaction, volume_interaction = None, entropy_interaction = None ): 

        self.n_endmembers = len(endmembers)

        # Create array of van Laar parameters
        self.alpha=np.array(alphas)

        # Create 2D arrays of interaction parameters
        self.Wh=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Ws=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Wv=np.zeros(shape=(self.n_endmembers,self.n_endmembers))

        #setup excess enthalpy interaction matrix
        for i in range(self.n_endmembers):
            for j in range(i+1, self.n_endmembers):
                self.Wh[i][j]=2.*enthalpy_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        if entropy_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Ws[i][j]=2.*entropy_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        if volume_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Wv[i][j]=2.*volume_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])
                        

        #initialize ideal solution model
        IdealSolution.__init__(self, endmembers )
        
    def _phi( self, molar_fractions):
        phi=np.array([self.alpha[i]*molar_fractions[i] for i in range(self.n_endmembers)])
        phi=np.divide(phi, np.sum(phi))
        return phi

    def _non_ideal_interactions( self, molar_fractions ):
        # -sum(sum(qi.qj.Wij*)
        # equation (2) of Holland and Powell 2003

        phi=self._phi(molar_fractions)

        q=np.zeros(len(molar_fractions))
        Hint=np.zeros(len(molar_fractions))
        Sint=np.zeros(len(molar_fractions))
        Vint=np.zeros(len(molar_fractions))

        for l in range(self.n_endmembers):
            q=np.array([kd(i,l)-phi[i] for i in range(self.n_endmembers)])

            Hint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Wh,q))
            Sint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Ws,q))
            Vint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Wv,q))
     
        return Hint, Sint, Vint

    def _non_ideal_excess_partial_gibbs( self, pressure, temperature, molar_fractions) :

        Hint, Sint, Vint = self._non_ideal_interactions( molar_fractions )
        return Hint - temperature*Sint + pressure*Vint


    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fractions ):

        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs (self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions)
        return ideal_gibbs + non_ideal_gibbs

    def excess_volume ( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        V_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Wv,phi))
        return V_excess

    def excess_entropy( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        S_conf=-constants.gas_constant*np.dot(IdealSolution._log_ideal_activities(self, molar_fractions), molar_fractions)
        S_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Ws,phi))
        return S_conf + S_excess

    def excess_enthalpy( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        H_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Wh,phi))
        return H_excess + pressure*self.excess_volume ( pressure, temperature, molar_fractions )


class SymmetricRegularSolution (AsymmetricRegularSolution):
    """
    Solution model implementing the symmetric regular solution model
    """
    def __init__( self, endmembers, enthalpy_interaction, volume_interaction = None, entropy_interaction = None ):
        alphas = np.ones( len(endmembers) )
        AsymmetricRegularSolution.__init__(self, endmembers, alphas, enthalpy_interaction, volume_interaction, entropy_interaction )

class AsymmetricRegularSolution_w_magnetism (IdealSolution):
    """
    Solution model implementing the asymmetric regular solution model formulation (Holland and Powell, 2003)
    """

    def __init__( self, endmembers, alphas, magnetic_parameters, enthalpy_interaction, volume_interaction = None, entropy_interaction = None ): 

        self.n_endmembers = len(endmembers)

        # Create array of van Laar parameters
        self.alpha=np.array(alphas)

        # Create 2D arrays of interaction parameters
        self.Wh=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Ws=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Wv=np.zeros(shape=(self.n_endmembers,self.n_endmembers))

        #setup excess enthalpy interaction matrix
        for i in range(self.n_endmembers):
            for j in range(i+1, self.n_endmembers):
                self.Wh[i][j]=2.*enthalpy_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        if entropy_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Ws[i][j]=2.*entropy_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        if volume_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Wv[i][j]=2.*volume_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        # Magnetism
        self.structural_parameter=np.array(magnetic_parameters[0])
        self.magnetic_moments=np.array(magnetic_parameters[1])
        self.Tcs=np.array(magnetic_parameters[2])
        self.magnetic_moment_excesses=np.array(magnetic_parameters[3])
        self.Tc_excesses=np.array(magnetic_parameters[4])
        self.WTc=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.WB=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        try:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WTc[i][j] = 2.*self.Tc_excesses[i][j-i-1]/(self.alpha[i]+self.alpha[j])
        except AttributeError:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WTc[i][j] = 0.
                        
        try:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WB[i][j] = 2.*self.magnetic_moment_excesses[i][j-i-1]/(self.alpha[i]+self.alpha[j])
        except AttributeError:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WB[i][j] = 0.
                        

        #initialize ideal solution model
        IdealSolution.__init__(self, endmembers )
        
    def _phi( self, molar_fractions):
        phi=np.array([self.alpha[i]*molar_fractions[i] for i in range(self.n_endmembers)])
        phi=np.divide(phi, np.sum(phi))
        return phi

    def _non_ideal_interactions( self, molar_fractions ):
        # -sum(sum(qi.qj.Wij*)
        # equation (2) of Holland and Powell 2003

        phi=self._phi(molar_fractions)

        q=np.zeros(len(molar_fractions))
        Hint=np.zeros(len(molar_fractions))
        Sint=np.zeros(len(molar_fractions))
        Vint=np.zeros(len(molar_fractions))

        for l in range(self.n_endmembers):
            q=np.array([kd(i,l)-phi[i] for i in range(self.n_endmembers)])

            Hint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Wh,q))
            Sint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Ws,q))
            Vint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Wv,q))
     
        return Hint, Sint, Vint

    def _non_ideal_excess_partial_gibbs( self, pressure, temperature, molar_fractions) :

        Hint, Sint, Vint = self._non_ideal_interactions( molar_fractions )
        return Hint - temperature*Sint + pressure*Vint

    def _magnetic_params(self, structural_parameter):
        A = (518./1125.) + (11692./15975.)*((1./structural_parameter) - 1.)
        b = (474./497.)*(1./structural_parameter - 1.)
        a = [-79./(140.*structural_parameter), -b/6., -b/135., -b/600., -1./10., -1./315., -1./1500.]
        return A, b, a

    def _magnetic_entropy(self, tau, magnetic_moment, structural_parameter):
        """
        Returns the magnetic contribution to the Gibbs free energy [J/mol]
        Expressions are those used by Chin, Hertzman and Sundman (1987)
        as reported in Sundman in the Journal of Phase Equilibria (1991)
        """

        A, b, a = self._magnetic_params(structural_parameter)
        if tau < 1: 
            f=1.+(1./A)*(a[0]/tau + b*(a[1]*np.power(tau, 3.) + a[2]*np.power(tau, 9.) + a[3]*np.power(tau, 15.)))
        else:
            f=(1./A)*(a[4]*np.power(tau,-5) + a[5]*np.power(tau,-15) + a[6]*np.power(tau, -25))

        return -constants.gas_constant*np.log(magnetic_moment + 1.)*f
        

    def _magnetic_excess_partial_gibbs( self, pressure, temperature, molar_fractions) :
        # This function calculates the partial *excess* gibbs free energies due to magnetic ordering
        # Note that the endmember contributions are subtracted from the excesses 
        # i.e. the partial *excess* gibbs free energy at any endmember composition is zero

        phi=self._phi(molar_fractions)

        # Tc, magnetic moment and magnetic Gibbs contribution at P,T,X
        Tc=np.dot(self.Tcs.T, molar_fractions) + np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.WTc,phi))

        if Tc > 1.e-12: 
            tau=temperature/Tc
            magnetic_moment=np.dot(self.magnetic_moments, molar_fractions) + np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.WB,phi))
            Gmag=-temperature*self._magnetic_entropy(tau, magnetic_moment, self.structural_parameter)
            f = Gmag/(constants.gas_constant*temperature*np.log(magnetic_moment + 1.))
            # Partial excesses
            A, b, a = self._magnetic_params(self.structural_parameter)
            dtaudtc=-temperature/(Tc*Tc)
            if tau < 1: 
                dfdtau=(1./A)*(-a[0]/(tau*tau) + 3.*a[1]*np.power(tau, 2.) + 9.*a[2]*np.power(tau, 8.) + 15.*a[3]*np.power(tau, 14.))
            else:
                dfdtau=(1./A)*(-5.*a[4]*np.power(tau,-6) - 15.*a[5]*np.power(tau,-16) - 25.*a[6]*np.power(tau, -26))
        else:
            Gmag=dfdtau=dtaudtc=magnetic_moment=f=0.0

        # Calculate the effective partial Tc and magnetic moments at the endmembers
        # (in order to calculate the partial derivatives of Tc and magnetic moment at the composition of interest)
        partial_B=np.zeros(self.n_endmembers)
        partial_Tc=np.zeros(self.n_endmembers)
        endmember_Gmag=np.zeros(self.n_endmembers)
        for l in range(self.n_endmembers):
            if self.Tcs[l]>1.e-12:
                endmember_Gmag[l] = -temperature*self._magnetic_entropy(temperature/self.Tcs[l], self.magnetic_moments[l], self.structural_parameter)
            else:
                endmember_Gmag[l]=0.

            q=np.array([kd(i,l)-phi[i] for i in range(self.n_endmembers)])
            partial_Tc[l]=self.Tcs[l]-self.alpha[l]*np.dot(q,np.dot(self.WTc,q))
            partial_B[l]=self.magnetic_moments[l]-self.alpha[l]*np.dot(q,np.dot(self.WB,q))
            
        tc_diff = partial_Tc - Tc
        magnetic_moment_diff= partial_B - magnetic_moment
        
        # Use the chain and product rules on the expression for the magnetic gibbs free energy
        XdGdX=constants.gas_constant*temperature*(magnetic_moment_diff*f/(magnetic_moment + 1.) + dfdtau*dtaudtc*tc_diff*np.log(magnetic_moment + 1.))

        endmember_contributions=np.dot(endmember_Gmag, molar_fractions) 

        return Gmag - endmember_contributions + XdGdX


    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fractions ):
        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs (self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions)
        magnetic_gibbs = self._magnetic_excess_partial_gibbs(pressure, temperature, molar_fractions)    
        return ideal_gibbs + non_ideal_gibbs + magnetic_gibbs

    def excess_volume ( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        V_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Wv,phi))
        return V_excess

    def excess_entropy( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        S_conf=-constants.gas_constant*np.dot(IdealSolution._log_ideal_activities(self, molar_fractions), molar_fractions)
        S_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Ws,phi))
        S_mag = -np.dot(self._magnetic_excess_partial_gibbs(pressure, temperature, molar_fractions), molar_fractions)/temperature
        return S_conf + S_excess + S_mag

    def excess_enthalpy( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        H_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Wh,phi))
        return H_excess + pressure*self.excess_volume ( pressure, temperature, molar_fractions )


class SymmetricRegularSolution_w_magnetism (AsymmetricRegularSolution_w_magnetism):
    """
    Solution model implementing the symmetric regular solution model
    """
    def __init__( self, endmembers, magnetic_parameters, enthalpy_interaction, volume_interaction = None, entropy_interaction = None ):
        alphas = np.ones( len(endmembers) )
        AsymmetricRegularSolution_w_magnetism.__init__(self, endmembers, alphas, magnetic_parameters, enthalpy_interaction, volume_interaction, entropy_interaction )

