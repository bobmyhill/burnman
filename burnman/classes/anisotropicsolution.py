# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2022 by the BurnMan team, released under the GNU
# GPL v2 or later.
import numpy as np
from .elasticsolution import ElasticSolution
from .anisotropicmineral import AnisotropicMineral
from .anisotropicmineral import contract_stresses
from .anisotropicmineral import expand_stresses
from .material import material_property, Material
from ..utils.unitcell import cell_parameters_to_vectors
from scipy.linalg import logm


IpmIqn = np.einsum('pm, qn->pqmn', np.eye(3), np.eye(3))


class AnisotropicSolution(AnisotropicMineral):
    """
    A class implementing the anisotropic solution model described
    in :cite:`Myhill2022b`.
    This class is derived from AnisotropicMineral,
    and inherits most of the methods from that class.

    Instantiation of an AnisotropicSolution is similar to that of an
    ElasticSolution, except that each of the endmembers must be an instance of
    an AnisotropicMineral, and an additional function is passed to the
    constructor of AnisotropicSolution that describes excess contributions
    to the anisotropic state tensor (Psixs) and its derivatives with respect to
    volume, temperature and composition.
    The function arguments should be V, Pth, X and params,
    in that order. The output variables Psixs, dPsixsdf, dPsixsdPth and
    dPsixsdX must be returned in that order in a tuple. The user should
    explicitly state whether the solution is always orthotropic or not
    by supplying a boolean to the orthotropic argument.

    Finally, a set of vectors should be passed that represent rapid
    deformation modes. For example, a solution of MgO, FeHSO and FeLSO
    (high and low spin wuestite) can rapidly change proportion of
    high spin and low spin iron, and so a single vector should be passed:
    np.array([[0., -1., 1.]]) or some multiple thereof.

    States of the mineral can only be queried after setting the
    pressure and temperature using set_state() and the composition using
    set_composition().

    This class is available as ``burnman.AnisotropicSolution``.
    """
    def __init__(self,
                 name=None,
                 solution_type=None,
                 endmembers=None,
                 energy_interaction=None,
                 pressure_interaction=None,
                 entropy_interaction=None,
                 energy_ternary_terms=None,
                 pressure_ternary_terms=None,
                 entropy_ternary_terms=None,
                 alphas=None,
                 excess_helmholtz_function=None,
                 master_cell_parameters=np.array([1., 1., 1.,
                                                  90., 90., 90]),
                 anisotropic_parameters={},
                 psi_excess_function=None,
                 dXdQ=None,
                 orthotropic=None,
                 relaxed=True,
                 molar_fractions=None):
        """
        Set up matrices to speed up calculations for when P, T, X is defined.
        """
        Material.__init__(self)
        self.T_0 = 298.15
        self.scalar_solution = ElasticSolution(name=name,
                                 solution_type=solution_type,
                                 endmembers=endmembers,
                                 energy_interaction=energy_interaction,
                                 pressure_interaction=pressure_interaction,
                                 entropy_interaction=entropy_interaction,
                                 energy_ternary_terms=energy_ternary_terms,
                                 pressure_ternary_terms=pressure_ternary_terms,
                                 entropy_ternary_terms=entropy_ternary_terms,
                                 alphas=alphas,
                                 excess_helmholtz_function=excess_helmholtz_function,
                                 molar_fractions=molar_fractions)

        self.orthotropic = orthotropic
        self.anisotropic_params = anisotropic_parameters
        self.psi_excess_function = psi_excess_function

        if dXdQ is None:
            self.dXdQ = np.zeros((self.n_endmembers, 0))
            self.relaxed = False
        else:
            assert len(dXdQ) == self.scalar_solution.n_endmembers
            self.dXdQ = dXdQ
            self.relaxed = relaxed

        cv = cell_parameters_to_vectors(master_cell_parameters)
        self.cell_vectors_0 = cv
        self.cell_volume_0 = np.linalg.det(self.cell_vectors_0)
        self.lnF0s = np.array([logm(np.linalg.solve(self.cell_vectors_0,
                                                    mbr[0].cell_vectors_0))
                               for mbr in self.scalar_solution.endmembers])

    def set_state(self, pressure, temperature):
        # Set solution conditions
        if not hasattr(self.scalar_solution, "molar_fractions"):
            raise Exception('To use this EoS, '
                            'you must first set the composition')
        self.scalar_solution.set_state(pressure, temperature)

        # 1) Compute dPthdf|T
        # relatively large dP needed for accurate estimate of dPthdf
        self.scalar_solution.set_state(pressure, temperature)
        dP = 1.e4 # self.isothermal_bulk_modulus_reuss*1.e-5

        self.scalar_solution.set_state(pressure-dP/2., temperature)
        V1 = self.scalar_solution.V
        Pth1 = pressure-dP/2. - self.scalar_solution._find_pressure(self.T_0,
                                                               V1, self.scalar_solution.molar_fractions)

        self.scalar_solution.set_state(pressure+dP/2., temperature)
        V2 = self.scalar_solution.V
        Pth2 = pressure+dP/2. - self.scalar_solution._find_pressure(self.T_0,
                                                          V2, self.scalar_solution.molar_fractions)

        self.dPthdf = (Pth2 - Pth1) / np.log(V2/V1)
        self.scalar_solution.set_state(pressure, temperature)

        # 2) Compute other properties needed for anisotropic equation of state
        V = self.scalar_solution.molar_volume
        f = np.log(V/self.cell_volume_0)
        self._f = f

        self.Pth = (Pth1 + Pth2)/2.

        # Get endmember values of psi and derivatives
        self.Psi_Voigt_mbr = np.array([mbr[0].Psi_Voigt
                                       for mbr in self.scalar_solution.endmembers])
        self.dPsidf_Voigt_mbr = np.array([mbr[0].dPsidf_Voigt
                                          for mbr in self.scalar_solution.endmembers])
        self.dPsidPth_Voigt_mbr = np.array([mbr[0].dPsidPth_Voigt
                                            for mbr in self.scalar_solution.endmembers])

        Material.set_state(self, pressure, temperature)


    def set_state_with_volume(self, volume, temperature):
        self.scalar_solution.set_state_with_volume(volume, temperature)
        self.set_state(self.scalar_solution.pressure, temperature)

    def set_composition(self, molar_fractions):

        self.scalar_solution.set_composition(molar_fractions)


    def set_relaxation(self, relax_bool):
        if self.relaxed is not relax_bool:
            self.relaxed = relax_bool
            Material.reset(self)


    @material_property
    def molar_fractions(self):
        return self.scalar_solution.molar_fractions

    @material_property
    def partial_gibbs(self):
        return self.scalar_solution.partial_gibbs

    @material_property
    def molar_gibbs(self):
        return self.scalar_solution.molar_gibbs

    @material_property
    def molar_volume(self):
        return self.scalar_solution.V

    @material_property
    def molar_entropy(self):
        return self.scalar_solution.S

    @material_property
    def _Psi_excess_tuple(self):
        psi_xs = self.psi_excess_function(self._f, self.Pth,
                                          self.molar_fractions,
                                          self.anisotropic_params)
        return psi_xs

    @material_property
    def Psi_Voigt(self):
        Psi0_mech_full = np.einsum('ijk, i, lm->jklm',
                                   self.lnF0s, self.molar_fractions,
                                   np.eye(3)/3.)
        Psi0_mech = self._contract_compliances(Psi0_mech_full)
        Psi_mech = (np.einsum('ijk, i->jk',
                              self.Psi_Voigt_mbr,
                              self.molar_fractions)
                    + Psi0_mech)
        return Psi_mech + self._Psi_excess_tuple[0]

    @material_property
    def dPsidf_Voigt(self):
        dPsidf_mech = (np.einsum('ijk, i->jk',
                                 self.dPsidf_Voigt_mbr,
                                 self.molar_fractions))
        return dPsidf_mech + self._Psi_excess_tuple[1]

    @material_property
    def dPsidPth_Voigt(self):
        dPsidPth_mech = (np.einsum('ijk, i->jk',
                                   self.dPsidPth_Voigt_mbr,
                                   self.molar_fractions))
        return dPsidPth_mech + self._Psi_excess_tuple[2]

    @material_property
    def dPsidX(self):
        dPsidX_mech = np.einsum('ijk, lm->jklmi',
                                self.lnF0s, np.eye(3)/3.)

        x = np.array([self._voigt_notation_to_compliance_tensor(A)
                      for A in self._Psi_excess_tuple[3].T]).T

        return dPsidX_mech + x

    @material_property
    def depsdQ(self):
        return np.einsum('ijklm, mn, kl->ijn',
                         self.dPsidX, self.dXdQ, np.eye(3))

    @material_property
    def dSdQ(self):
        return np.einsum('i, ij->j',
                         self.scalar_solution._partial_entropies, self.dXdQ)

    @material_property
    def dPdQ(self):
        return np.einsum('i, ij->j',
                         self.scalar_solution._partial_pressures, self.dXdQ)

    @material_property
    def dFdQ(self):
        return np.einsum('i, ij->j',
                         self.scalar_solution._partial_helmholtz,
                         self.dXdQ)

    @material_property
    def d2FdQdQ(self):
        return np.einsum('ij, ik, jl->kl',
                         self.scalar_solution._helmholtz_hessian,
                         self.dXdQ,
                         self.dXdQ)

    @material_property
    def d2FdQdZ(self):
        beta_T = np.einsum('ijkl, kl', 
                           self.full_isothermal_compliance_tensor_unrelaxed,
                           np.eye(3))
        beta_RT = np.einsum('ij, ij', 
                            beta_T,
                            np.eye(3))
        dd = IpmIqn - np.einsum('pq, mn->pqmn', beta_T/beta_RT, np.eye(3))
        C_T = self.full_isothermal_stiffness_tensor_unrelaxed
        d2FdQdeps = -self.V*(np.einsum('mn, i->imn', np.eye(3), self.dPdQ)
                             + np.einsum('pqmn, kli, klpq->imn',
                                         dd, self.depsdQ, C_T))
        d2FdQdeps = np.array([contract_stresses(arr) for arr in d2FdQdeps])
        d2FdQdT = -self.dSdQ

        return np.concatenate((d2FdQdeps, d2FdQdT[:, np.newaxis]), axis=1)

    @material_property
    def d2FdQdQ_relaxed(self):
        C_T = self.full_isothermal_stiffness_tensor_unrelaxed
        return (self.d2FdQdQ
                + self.V*np.einsum('kli, klmn, mnj->ij',
                                   self.depsdQ,
                                   C_T,
                                   self.depsdQ))

    @material_property
    def d2FdQdQ_relaxed_pinv(self):
        return np.linalg.pinv(self.d2FdQdQ_relaxed)

    @material_property
    def d2FdZdZ(self):
        CT = self.isothermal_stiffness_tensor_unrelaxed
        pi = contract_stresses(self.thermal_stress_tensor_unrelaxed)
        c_eps = self.molar_isometric_heat_capacity_unrelaxed
        V = self.molar_volume
        T = self.temperature
        return np.block([[V * CT, V * pi[:, np.newaxis]],
                         [V * pi, -c_eps/T]])

    @material_property
    def d2FdZdZ_relaxed(self):
        return self.d2FdZdZ - np.einsum('kl, ki, lj->ij',
                                        self.d2FdQdQ_relaxed_pinv,
                                        self.d2FdQdZ,
                                        self.d2FdQdZ)

    @material_property
    def isothermal_stiffness_tensor(self):
        if self.relaxed:
            C_T = self.d2FdZdZ_relaxed[:6, :6] / self.V
            if self.orthotropic:
                return C_T
            else:
                R = self.rotation_matrix
                C = self._voigt_notation_to_stiffness_tensor(C_T)
                C_rotated = np.einsum('mi, nj, ok, pl, ijkl->mnop',
                                      R, R, R, R, C)
                return self._contract_stiffnesses(C_rotated)

        else:
            return self.isothermal_stiffness_tensor_unrelaxed

    @material_property
    def full_isothermal_stiffness_tensor(self):
        CT = self.isothermal_stiffness_tensor
        return self._voigt_notation_to_stiffness_tensor(CT)

    @material_property
    def thermal_stress_tensor(self):
        if self.relaxed:
            pi = expand_stresses(self.d2FdZdZ_relaxed[6, :6] /
                                 self.V)

            if self.orthotropic:
                return pi
            else:
                R = self.rotation_matrix
                return np.einsum('mi, nj, ij->mn', R, R, pi)

        else:
            return self.thermal_stress_tensor_unrelaxed

    @material_property
    def molar_isometric_heat_capacity(self):
        if self.relaxed:
            return -self.d2FdZdZ_relaxed[6, 6] * self.temperature
        else:
            return self.molar_isometric_heat_capacity_unrelaxed

    @material_property
    def isothermal_compliance_tensor(self):
        """
        Returns
        -------
        isothermal_compliance_tensor : 2D numpy array
            The isothermal compliance tensor [1/Pa]
            in Voigt form (:math:`\\mathbb{S}_{\\text{T} pq}`).
        """
        if self.relaxed:
            return np.linalg.inv(self.isothermal_stiffness_tensor)
        else:
            return self.isothermal_compliance_tensor_unrelaxed

    @material_property
    def full_isothermal_compliance_tensor(self):
        S_Voigt = self.isothermal_compliance_tensor
        return self._voigt_notation_to_compliance_tensor(S_Voigt)

    @material_property
    def full_isentropic_stiffness_tensor(self):
        return (self.full_isothermal_stiffness_tensor
                + (self.molar_volume * self.temperature
                   * np.einsum('ij, kl', self.thermal_stress_tensor, self.thermal_stress_tensor)
                   / self.molar_isometric_heat_capacity))

    @material_property
    def isentropic_stiffness_tensor(self):
        C_full = self.full_isentropic_stiffness_tensor
        return self._contract_stiffnesses(C_full)


    @material_property
    def isentropic_compliance_tensor(self):
        return np.linalg.inv(self.isentropic_stiffness_tensor)

    @material_property
    def full_isentropic_compliance_tensor(self):
        S_Voigt = self.isentropic_compliance_tensor
        return self._voigt_notation_to_compliance_tensor(S_Voigt)

    @material_property
    def thermal_expansivity_tensor(self):
        """
        Returns
        -------
        thermal_expansivity_tensor : 2D numpy array
            The tensor of thermal expansivities [1/K].
        """
        if self.relaxed:
            alpha = -np.einsum('ijkl, kl',
                               self.full_isothermal_compliance_tensor,
                               self.thermal_stress_tensor)
            return alpha
        else:
            return self.thermal_expansivity_tensor_unrelaxed

    @material_property
    def isothermal_compliance_tensor_unrelaxed(self):
        """
        Returns
        -------
        isothermal_compliance_tensor : 2D numpy array
            The isothermal compliance tensor [1/Pa]
            in Voigt form (:math:`\\mathbb{S}_{\\text{T} pq}`).
        """
        S_T = ((1./self.isothermal_K_RT_unrelaxed)
               * (self.dPsidf_Voigt + self.dPsidPth_Voigt * self.dPthdf))
        if self.orthotropic:
            return S_T
        else:
            R = self.rotation_matrix
            S = self._voigt_notation_to_compliance_tensor(S_T)
            S_rotated = np.einsum('mi, nj, ok, pl, ijkl->mnop', R, R, R, R, S)
            return self._contract_compliances(S_rotated)

    @material_property
    def thermal_expansivity_tensor_unrelaxed(self):
        """
        Returns
        -------
        thermal_expansivity_tensor : 2D numpy array
            The tensor of thermal expansivities [1/K].
        """
        a = (self.alpha_unrelaxed
             * (self.dPsidf_Voigt
                + self.dPsidPth_Voigt
                * (self.dPthdf
                   + self.isothermal_K_RT_unrelaxed)))
        alpha = np.einsum('ijkl, kl',
                          self._voigt_notation_to_compliance_tensor(a),
                          np.eye(3))

        if self.orthotropic:
            return alpha
        else:
            R = self.rotation_matrix
            return np.einsum('mi, nj, ij->mn', R, R, alpha)

    # Derived properties start here
    @material_property
    def isothermal_stiffness_tensor_unrelaxed(self):
        """
        Returns
        -------
        isothermal_stiffness_tensor : 2D numpy array
            The isothermal stiffness tensor [Pa]
            in Voigt form (:math:`\\mathbb{C}_{\\text{T} pq}`).
        """
        return np.linalg.inv(self.isothermal_compliance_tensor_unrelaxed)

    @material_property
    def full_isothermal_stiffness_tensor_unrelaxed(self):
        """
        Returns
        -------
        full_isothermal_stiffness_tensor : 4D numpy array
            The isothermal stiffness tensor [Pa]
            in standard form (:math:`\\mathbb{C}_{\\text{T} ijkl}`).
        """
        CT = self.isothermal_stiffness_tensor_unrelaxed
        return self._voigt_notation_to_stiffness_tensor(CT)

    @material_property
    def full_isothermal_compliance_tensor_unrelaxed(self):
        """
        Returns
        -------
        full_isothermal_stiffness_tensor : 4D numpy array
            The isothermal compliance tensor [1/Pa]
            in standard form (:math:`\\mathbb{S}_{\\text{T} ijkl}`).
        """
        S_Voigt = self.isothermal_compliance_tensor_unrelaxed
        return self._voigt_notation_to_compliance_tensor(S_Voigt)

    @material_property
    def full_isentropic_compliance_tensor_unrelaxed(self):
        """
        Returns
        -------
        full_isentropic_stiffness_tensor : 4D numpy array
            The isentropic compliance tensor [1/Pa]
            in standard form (:math:`\\mathbb{S}_{\\text{N} ijkl}`).
        """
        return (self.full_isothermal_compliance_tensor_unrelaxed
                - np.einsum('ij, kl->ijkl',
                            self.thermal_expansivity_tensor_unrelaxed,
                            self.thermal_expansivity_tensor_unrelaxed)
                * self.V * self.temperature / self.molar_heat_capacity_p_unrelaxed)

    @material_property
    def isentropic_compliance_tensor_unrelaxed(self):
        """
        Returns
        -------
        isentropic_compliance_tensor : 2D numpy array
            The isentropic compliance tensor [1/Pa]
            in Voigt form (:math:`\\mathbb{S}_{\\text{N} pq}`).
        """
        S_full = self.full_isentropic_compliance_tensor_unrelaxed
        return self._contract_compliances(S_full)

    @material_property
    def isentropic_stiffness_tensor_unrelaxed(self):
        """
        Returns
        -------
        isentropic_stiffness_tensor : 2D numpy array
            The isentropic stiffness tensor [Pa]
            in Voigt form (:math:`\\mathbb{C}_{\\text{N} pq}`).
        """
        return np.linalg.inv(self.isentropic_compliance_tensor_unrelaxed)

    @material_property
    def full_isentropic_stiffness_tensor_unrelaxed(self):
        """
        Returns
        -------
        full_isentropic_stiffness_tensor : 4D numpy array
            The isentropic stiffness tensor [Pa]
            in standard form (:math:`\\mathbb{C}_{\\text{N} ijkl}`).
        """
        C_Voigt = self.isentropic_stiffness_tensor_unrelaxed
        return self._voigt_notation_to_stiffness_tensor(C_Voigt)

    @material_property
    def thermal_stress_tensor_unrelaxed(self):
        """
        Returns
        -------
        thermal stress : 2D numpy array
            The change in stress with temperature at constant strain.
        """
        pi = -np.einsum('ijkl, kl',
                        self.full_isothermal_stiffness_tensor_unrelaxed,
                        self.thermal_expansivity_tensor_unrelaxed)
        return pi

    @material_property
    def molar_isometric_heat_capacity_unrelaxed(self):
        """
        Returns
        -------
        molar_isometric_heat_capacity : float
            The molar heat capacity at constant strain.
        """

        pi = self.thermal_stress_tensor_unrelaxed
        pipiV = np.einsum('ij, kl -> ijkl', pi, pi)*self.V
        indices = np.where(np.abs(pipiV) > 1.e-5)
        values = ((self.full_isentropic_stiffness_tensor_unrelaxed
                   - self.full_isothermal_stiffness_tensor_unrelaxed)[indices]
                  / pipiV[indices])
        if not np.allclose(values, np.ones_like(values)*values[0],
                           rtol=1.e-5):
            """
            raise Exception('Could not calculate the molar heat '
                            'capacity at constant strain. '
                            'There is an inconsistency in the '
                            'equation of state.')
            """
            print('not ok')

        C_isometric = self.temperature/values[0]

        return C_isometric

    @material_property
    def molar_heat_capacity_p_unrelaxed(self):
        """
        Returns molar heat capacity at constant pressure
        of the solution [J/K/mol].
        Aliased with self.C_p.
        """
        return (self.molar_heat_capacity_v_unrelaxed
                + self.molar_volume * self.temperature
                * self.alpha_unrelaxed * self.alpha_unrelaxed
                * self.isothermal_K_RT_unrelaxed)
    
    @material_property
    def isothermal_bulk_modulus_voigt(self):
        return np.sum(self.isothermal_stiffness_tensor[:3,:3])

    @material_property
    def isothermal_bulk_modulus_reuss(self):
        return 1./np.sum(self.isothermal_compliance_tensor[:3,:3])

    @material_property
    def isothermal_bulk_modulus(self):
        """
        Anisotropic minerals do not have a single isothermal bulk modulus.
        This function returns a NotImplementedError. Users should instead
        consider either using isothermal_bulk_modulus_reuss,
        isothermal_bulk_modulus_voigt,
        or directly querying the elements in the isothermal_stiffness_tensor.
        """
        raise NotImplementedError("isothermal_bulk_modulus is not "
                                  "sufficiently explicit for an "
                                  "anisotropic mineral. Did you mean "
                                  "isothermal_bulk_modulus_reuss?")
    
    @material_property
    def thermal_expansivity(self):
        return np.trace(self.thermal_expansivity_tensor)

    @material_property
    def molar_heat_capacity_p(self):
        """
        Returns
        -------
        molar_heat_capacity_p : float
            The molar isobaric heat capacity.
        """

        alpha = self.thermal_expansivity_tensor
        aaV = np.einsum('ij, kl -> ijkl', alpha, alpha)*self.V
        indices = np.where(np.abs(aaV) > 1.e-30)
        values = ((self.full_isentropic_compliance_tensor
                   - self.full_isothermal_compliance_tensor)[indices]
                  / aaV[indices])
        if not np.allclose(values, np.ones_like(values)*values[0],
                           rtol=1.e-5):
            #raise Exception('Could not calculate the molar heat '
            #                'capacity at constant strain. '
            #                'There is an inconsistency in the '
            #                'equation of state.')
            print('C_p inconsistency')
        C_p = -self.temperature/values[0]
        return C_p

    @material_property
    def molar_heat_capacity_v(self):
        return (self.molar_heat_capacity_p - self.molar_volume * self.temperature \
                * self.thermal_expansivity * self.thermal_expansivity \
                * self.isothermal_bulk_modulus_reuss)


    @material_property
    def alpha_unrelaxed(self):
        return self.scalar_solution.thermal_expansivity

    @material_property
    def isothermal_K_RT_unrelaxed(self):
        return self.scalar_solution.isothermal_bulk_modulus

    @material_property
    def molar_heat_capacity_v_unrelaxed(self):
        return self.scalar_solution.molar_heat_capacity_v