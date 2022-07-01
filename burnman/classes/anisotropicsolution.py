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


dd = (np.einsum('pm, qn->pqmn', np.eye(3), np.eye(3))
      - np.einsum('pq, mn->pqmn', np.eye(3), np.eye(3))/3.)


class AnisotropicSolution(ElasticSolution, AnisotropicMineral):
    """
    A class implementing the anisotropic solution model described
    in :cite:`Myhill2022b`.
    This class is derived from both ElasticSolution and AnisotropicMineral,
    and inherits most of the methods from these classes.

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
                 volume_interaction=None,
                 entropy_interaction=None,
                 energy_ternary_terms=None,
                 volume_ternary_terms=None,
                 entropy_ternary_terms=None,
                 alphas=None,
                 excess_gibbs_function=None,
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
        ElasticSolution.__init__(self,
                                 name=name,
                                 solution_type=solution_type,
                                 endmembers=endmembers,
                                 energy_interaction=energy_interaction,
                                 volume_interaction=volume_interaction,
                                 entropy_interaction=entropy_interaction,
                                 energy_ternary_terms=energy_ternary_terms,
                                 volume_ternary_terms=volume_ternary_terms,
                                 entropy_ternary_terms=entropy_ternary_terms,
                                 alphas=alphas,
                                 excess_gibbs_function=excess_gibbs_function,
                                 molar_fractions=molar_fractions)

        self.orthotropic = orthotropic
        self.anisotropic_params = anisotropic_parameters
        self.psi_excess_function = psi_excess_function

        if dXdQ is None:
            self.dXdQ = np.zeros((self.n_endmembers, 0))
            self.relaxed = False
        else:
            assert len(dXdQ[0]) == self.n_endmembers
            self.dXdQ = dXdQ
            self.relaxed = relaxed

        cv = cell_parameters_to_vectors(master_cell_parameters)
        self.cell_vectors_0 = cv
        self.F0s = np.array([np.linalg.solve(self.cell_vectors_0,
                                             mbr[0].cell_vectors_0)
                             for mbr in self.endmembers])

    def set_state(self, pressure, temperature):

        # Set solution conditions
        ElasticSolution.set_state(self, pressure, temperature)

        # Get endmember values of psi and derivatives
        self.Psi_Voigt_mbr = np.array([mbr[0].Psi_Voigt
                                       for mbr in self.endmembers])
        self.dPsidf_Voigt_mbr = np.array([mbr[0].dPsidf_Voigt_mbr
                                          for mbr in self.endmembers])
        self.dPsidPth_Voigt_mbr = np.array([mbr[0].dPsidPth_Voigt_mbr
                                            for mbr in self.endmembers])

    def set_composition(self, molar_fractions):

        ElasticSolution.set_composition(self, molar_fractions)

    def set_relaxation(self, relax_bool):
        if self.relaxed is not relax_bool:
            self.relaxed = relax_bool
            Material.reset(self)

    @material_property
    def _Psi_excess_tuple(self):
        self.psi_excess_function(self.V, self.ddd,
                                 self.molar_fractions,
                                 self.anisotropic_params)

    @material_property
    def Psi_Voigt(self):
        Psi0_mech_full = np.einsum('ijk, i, lm->jklm',
                                  self.F0s, self.molar_fractions, np.eye(3)/3.)
        Psi0_mech = self._contract_stiffnesses(Psi0_mech_full)
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
    def dPsidX_Voigt(self):
        dPsidX_mech_full = np.einsum('ijk, lm->ijklm',
                                     self.F0s, np.eye(3)/3.)
        dPsidX_mech = np.array([self._contract_stiffnesses(dPsidX)
                                for dPsidX in dPsidX_mech_full])
        return dPsidX_mech + self._Psi_excess_tuple[3]

    @material_property
    def depsdQ(self):
        return np.einsum('ijklm, mn, kl->ijn',
                         self.dPsidX_Voigt, self.dXdQ, np.eye(3))

    @material_property
    def dSdQ(self):
        return np.einsum('i, ij->j',
                         self._partial_entropies, self.dXdQ)

    @material_property
    def dPdQ(self):
        return np.einsum('i, ij->j',
                         self._partial_pressures, self.dXdQ)

    @material_property
    def d2FdQ2(self):
        return np.einsum('ij, ik, jl->kl',
                         self._helmholtz_hessian,
                         self.dXdQ,
                         self.dXdQ)

    @material_property
    def unrelaxed_full_stiffness(self):
        relaxed_bool = self.relaxed
        self.relaxed = False
        C_T = self.full_isothermal_stiffness_tensor
        self.relaxed = relaxed_bool
        return C_T

    @material_property
    def d2FdQdZ(self):
        C_T = self.unrelaxed_full_stiffness
        d2FdQdeps = -self.V*(np.einsum('mn, i->imn', np.eye(3), self.dPdQ)
                             + np.einsum('pqmn, kli, klpq->imn',
                                         dd, self.depsdQ, C_T))
        d2FdQdeps = np.array([contract_stresses(arr) for arr in d2FdQdeps])
        d2FdQdT = -self.dSdQ

        return np.concatenate((d2FdQdeps, d2FdQdT[:, np.newaxis]), axis=1)

    @material_property
    def d2FdQdQ_relaxed(self):
        C_T = self.unrelaxed_full_stiffness
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
                C = self._voigt_notation_to_compliance_tensor(C_T)
                C_rotated = np.einsum('mi, nj, ok, pl, ijkl->mnop',
                                      R, R, R, R, C)
                return self._contract_compliances(C_rotated)

        else:
            return super().isothermal_stiffness_tensor

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
            return super().thermal_stress_tensor

    @material_property
    def molar_isometric_heat_capacity(self):
        if self.relaxed:
            return -self.d2FdZdZ_relaxed[6, 6] * self.temperature
        else:
            return super().molar_isometric_heat_capacity

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
            return super().isothermal_compliance_tensor

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
            return super().thermal_expansivity_tensor
