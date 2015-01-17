# BurnMan - a lower mantle toolkit
# Copyright (C) 2012-2014, Myhill, R., Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

import numpy as np
import math
from sympy.matrices import *
from copy import deepcopy
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

    def excess_gibbs_free_energy( self, pressure, temperature, molar_fraction):
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

        molar_fraction : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        G_excess : float
            The excess Gibbs free energy
        """
        return np.dot(np.array(molar_fraction), self.excess_partial_gibbs_free_energies( pressure, temperature, molar_fraction))

    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fraction):
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

        molar_fraction : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        partial_G_excess : numpy array
            The excess Gibbs free energy of each endmember
        """
        return np.empty_like( np.array(molar_fraction) )

    def excess_volume( self, pressure, temperature, molar_fraction):
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

        molar_fraction : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        V_excess : float
            The excess volume of the solution
        """
        return 0.0

    def excess_enthalpy( self, pressure, temperature, molar_fraction):
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

        molar_fraction : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        H_excess : float
            The excess enthalpy of the solution
        """
        return 0.0

    def excess_entropy( self, pressure, temperature, molar_fraction):
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

        molar_fraction : list of floats
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

    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fraction ):
        return self._ideal_excess_partial_gibbs( temperature, molar_fraction )

    def _calculate_endmember_configurational_entropies( self ):
        self.endmember_configurational_entropies=np.zeros(shape=(self.n_endmembers))
        for idx, endmember_occupancy in enumerate(self.endmember_occupancies):
            for occ in range(self.n_occupancies):
                if endmember_occupancy[occ] > 1e-10:
                    self.endmember_configurational_entropies[idx] = \
                        self.endmember_configurational_entropies[idx] - \
                        constants.gas_constant*self.site_multiplicities[occ]*endmember_occupancy[occ]*np.log(endmember_occupancy[occ])

    def _endmember_configurational_entropy_contribution(self, molar_fraction):
        return np.dot(molar_fraction, self.endmember_configurational_entropies)

    def _configurational_entropy (self, molar_fraction):
        site_occupancies=np.dot(molar_fraction, self.endmember_occupancies)
        conf_entropy=0
        for idx, occupancy in enumerate(site_occupancies):
            if occupancy > 1e-10:
                conf_entropy=conf_entropy-constants.gas_constant*occupancy*self.site_multiplicities[idx]*np.log(occupancy)

        return conf_entropy


    def _ideal_excess_partial_gibbs( self, temperature, molar_fraction ):
        return  constants.gas_constant*temperature * self._log_ideal_activities(molar_fraction)

    def _log_ideal_activities ( self, molar_fraction ):
        site_occupancies=np.dot(molar_fraction, self.endmember_occupancies)
        lna=np.empty(shape=(self.n_endmembers))

        for e in range(self.n_endmembers):
            lna[e]=0.0
            for occ in range(self.n_occupancies):
                if self.endmember_occupancies[e][occ] > 1e-10 and site_occupancies[occ] > 1e-10:
                    lna[e]=lna[e] + self.endmember_occupancies[e][occ]*self.site_multiplicities[occ]*np.log(site_occupancies[occ])

            normalisation_constant=self.endmember_configurational_entropies[e]/constants.gas_constant
            lna[e]=lna[e] + self.endmember_configurational_entropies[e]/constants.gas_constant
        return lna

    def _ideal_activities ( self, molar_fraction ):
        site_occupancies=np.dot(molar_fraction, self.endmember_occupancies)
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
        self.vanlaar=np.array(alphas)

        # Create 2D arrays of interaction parameters
        self.Wh=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Ws=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Wv=np.zeros(shape=(self.n_endmembers,self.n_endmembers))

        #setup excess enthalpy interaction matrix
        self.enthalpy_interaction=enthalpy_interaction
        for i in range(self.n_endmembers):
            for j in range(i+1, self.n_endmembers):
                self.Wh[i][j]=2.*enthalpy_interaction[i][j-i-1]/(self.vanlaar[i]+self.vanlaar[j])

        if entropy_interaction is not None:
            self.entropy_interaction=entropy_interaction
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Ws[i][j]=2.*entropy_interaction[i][j-i-1]/(self.vanlaar[i]+self.vanlaar[j])

        if volume_interaction is not None:
            self.volume_interaction=volume_interaction
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Wv[i][j]=2.*volume_interaction[i][j-i-1]/(self.vanlaar[i]+self.vanlaar[j])

        #initialize ideal solution model
        IdealSolution.__init__(self, endmembers )
        
    def _phi( self, molar_fraction):
        phi=np.array([self.vanlaar[i]*molar_fraction[i] for i in range(self.n_endmembers)])
        phi=np.divide(phi, np.sum(phi))
        return phi

    def _non_ideal_interactions( self, molar_fraction ):
        # -sum(sum(qi.qj.Wij*)
        # equation (2) of Holland and Powell 2003

        phi=self._phi(molar_fraction)

        q=np.zeros(len(molar_fraction))
        Hint=np.zeros(len(molar_fraction))
        Sint=np.zeros(len(molar_fraction))
        Vint=np.zeros(len(molar_fraction))

        for l in range(self.n_endmembers):
            q=np.array([kd(i,l)-phi[i] for i in range(self.n_endmembers)])

            Hint[l]=0.-self.vanlaar[l]*np.dot(q,np.dot(self.Wh,q))
            Sint[l]=0.-self.vanlaar[l]*np.dot(q,np.dot(self.Ws,q))
            Vint[l]=0.-self.vanlaar[l]*np.dot(q,np.dot(self.Wv,q))
     
        return Hint, Sint, Vint

    def _non_ideal_excess_partial_gibbs( self, pressure, temperature, molar_fraction) :

        Hint, Sint, Vint = self._non_ideal_interactions( molar_fraction )
        return Hint - temperature*Sint + pressure*Vint

    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fraction ):

        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs (self, temperature, molar_fraction )
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fraction)
        return ideal_gibbs + non_ideal_gibbs

    def excess_volume ( self, pressure, temperature, molar_fraction ):
        phi=self._phi(molar_fraction)
        V_excess=np.dot(self.vanlaar.T,molar_fraction)*np.dot(phi.T,np.dot(self.Wv,phi))
        return V_excess

    def excess_entropy( self, pressure, temperature, molar_fraction ):
        phi=self._phi(molar_fraction)
        S_conf=np.dot(IdealSolution._ideal_excess_partial_gibbs(self, temperature, molar_fraction),molar_fraction)
        S_excess=np.dot(self.vanlaar.T,molar_fraction)*np.dot(phi.T,np.dot(self.Ws,phi))
        return S_conf + S_excess

    def excess_enthalpy( self, pressure, temperature, molar_fraction ):
        phi=self._phi(molar_fraction)
        H_excess=np.dot(self.vanlaar.T,molar_fraction)*np.dot(phi.T,np.dot(self.Wh,phi))
        return H_excess + pressure*self.excess_volume ( pressure, temperature, molar_fraction )


    def replace_endmember(self, old_index, new_mineral, endmembers, molar_fraction=None):
        mbrsa = [str(i) for i in range(len(self.formulas))]
        mbrsb = [str(i) for i in range(len(self.formulas))]

        dalpha=Matrix([[0]*len(self.vanlaar)]*len(self.vanlaar))
        valpha=Matrix([0]*len(self.vanlaar))
        for i in range(len(self.vanlaar)):
            dalpha[i,i]=float(self.vanlaar[i])
            valpha[i]=float(self.vanlaar[i])

        
        # Input endmember occupancy matrix
        c=Matrix(self.endmember_occupancies)


        new_endmembers = deepcopy(endmembers)
        new_endmembers[old_index][0] = burnman.Mineral()
        new_endmembers[old_index][1] = new_mineral[1]
        
        new_endmembers[old_index][0].params['name']=new_mineral[0]
        eos=endmembers[old_index][0].params['equation_of_state']
        new_endmembers[old_index][0].params['equation_of_state']=eos

        new_endmembers[old_index][0].name=new_mineral[0]
        self.formulas[old_index]=new_mineral[1]
        self.solution_formulae, self.n_sites, self.sites, self.n_occupancies, self.endmember_occupancies, self.site_multiplicities = process_solution_chemistry(self.formulas)

        # Output endmember occupancy matrix
        cp=Matrix(self.endmember_occupancies)

        A=np.linalg.lstsq(c.T, cp.T)[0].round(10)

        alphap=A.T*Matrix(self.vanlaar).T
        dalphap=Matrix([[0]*len(self.vanlaar)]*len(self.vanlaar))
        dinvalphap=Matrix([[0]*len(self.vanlaar)]*len(self.vanlaar))

        for i in range(len(alphap)):
            dalphap[i,i]=alphap[i]
            dinvalphap[i,i]=1./alphap[i]

        B=dalpha*A*dinvalphap            
        vpa=Matrix(molar_fraction)
        pp=(np.linalg.lstsq(cp.T, c.T)[0].round(10))*vpa

        for i in range(len(mbrsb)):
            molar_fraction[i] = pp[i]
            self.vanlaar[i] =  alphap[i]


        new_alphas=np.array([alpha for alpha in alphap.T])

        new_enthalpy_interaction=deepcopy(self.enthalpy_interaction)
        new_entropy_interaction=deepcopy(self.enthalpy_interaction)
        new_volume_interaction=deepcopy(self.enthalpy_interaction)
        for i in range(self.n_endmembers-1):
            for j in range(self.n_endmembers-i-1):
                new_enthalpy_interaction[i][j]=0.
                new_entropy_interaction[i][j]=0.
                new_volume_interaction[i][j]=0.

        # Other properties
        molar_proportions_old_in_new=A.T[old_index]
        if eos == 'hp_tmt' or eos == 'mt':
            new_endmembers[old_index][0].params['Cp'] = [sum(endmembers[i][0].params['Cp'][j] * molar_proportions_old_in_new[i] for i in range(self.n_endmembers)) for j in range(len(endmembers[old_index][0].params['Cp']))]
        print new_endmembers[old_index][0].params['Cp']


        listvars=[(new_enthalpy_interaction, self.Wh,'H_0'), (new_entropy_interaction, self.Ws,'S_0'), (new_volume_interaction, self.Wv,'V_0')]
        for (interaction, W, var) in listvars:

            vGa=[]
            for i in range(len(mbrsa)):
                vGa.append(endmembers[i][0].params[var])
            vGa=Matrix([vGa]).T

            Wa=Matrix(W)
            Q=B.T*Wa*B
            Q3=Matrix(np.diag(np.diag(Q)))
            Qp3=Matrix([[0]*len(W)]*len(W)) # N.B Qp3=-Qf
            for i in range(len(W)):
                for j in range(len(W)):    
                    if j < i:
                        Qp3[j,i] += Q3[i,i]
                    elif j > i:
                        Qp3[i,j] += Q3[i,i]    

            Wbans=np.triu(Q,1) + np.tril(Q,-1).T - Qp3

            for i in range(len(W)):
                for j in range(len(W)):
                    if j > i:
                        interaction[i][j-i-1] = (alphap[i] + alphap[j])/2.*Wbans[i,j]

            vGbans=A.T*vGa + Q3*valpha
            new_endmembers[old_index][0].params[var]=vGbans[i]

        new_endmembers[old_index][0].params['a_0'] = [(1/new_endmembers[old_index][0].params['V_0']) * sum(endmembers[i][0].params['a_0'] * endmembers[i][0].params['V_0'] * molar_proportions_old_in_new[i] for i in range(self.n_endmembers))]


        warnings.warn('Still need to deal with derived parameters...', stacklevel=2)             

        # Start making new model
        new_model=burnman.solutionmodel.AsymmetricRegularSolution(new_endmembers, new_alphas, new_enthalpy_interaction, new_volume_interaction, new_entropy_interaction)

        return new_endmembers, new_model


class SymmetricRegularSolution (AsymmetricRegularSolution):
    """
    Solution model implementing the symmetric regular solution model
    """
    def __init__( self, endmembers, enthalpy_interaction, volume_interaction = None, entropy_interaction = None ):
        alphas = np.ones( len(endmembers) )
        AsymmetricRegularSolution.__init__(self, endmembers, alphas, enthalpy_interaction, volume_interaction, entropy_interaction )

