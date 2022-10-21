# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit
# for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2022 by the BurnMan team, released under the GNU
# GPL v2 or later.
import numpy as np
from .anisotropicmineral import AnisotropicMineral
from .material import material_property, Material
from ..utils.anisotropy import (
    contract_stresses,
    expand_stresses,
    voigt_notation_to_compliance_tensor,
    voigt_notation_to_stiffness_tensor,
    contract_compliances,
    contract_stiffnesses,
)

IpmIqn = np.einsum("pm, qn->pqmn", np.eye(3), np.eye(3))


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
    to the anisotropic state tensor (Psi_xs) and its derivatives with respect to
    volume, temperature and composition.
    The function arguments should be ln(V), Pth (a vector),
    X (a vector) and params,
    in that order. The output variables Psi_xs, dPsi_xsdlnV, dPsi_xsdPth and
    dPsi_xsdX must be returned in that order in a tuple. The user should
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

    def __init__(
        self,
        name=None,
        scalar_solution=None,
        anisotropic_parameters={},
        psi_excess_function=None,
        dXdQ=None,
        orthotropic=None,
        relaxed=True,
    ):
        """
        Set up matrices to speed up calculations for when P, T, X is defined.
        """
        Material.__init__(self)
        self.T_0 = 298.15

        self.name = name
        self.scalar_solution = scalar_solution

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

    def set_state(self, pressure, temperature):
        # Set solution conditions
        if not hasattr(self.scalar_solution, "molar_fractions"):
            raise Exception("To use this EoS, " "you must first set the composition")

        self.scalar_solution.set_state(pressure, temperature)
        V = self.scalar_solution.molar_volume
        f = np.log(V)
        self._f = f

        for mbr in self.scalar_solution.endmembers:
            mbr[0].set_state_with_volume(V, temperature)

        self._Pth_mbr = np.array(
            [mbr[0].Pth for mbr in self.scalar_solution.endmembers]
        )

        self._dPthdf_mbr = np.array(
            [mbr[0].dPthdf for mbr in self.scalar_solution.endmembers]
        )

        self._aKT_mbr = np.array(
            [mbr[0].alpha / mbr[0].beta_T for mbr in self.scalar_solution.endmembers]
        )

        # Get endmember values of psi and derivatives
        self._Psi_mbr = np.moveaxis(
            np.array([mbr[0].Psi for mbr in self.scalar_solution.endmembers]), 0, -1
        )
        self._dPsidf_Voigt_mbr = np.moveaxis(
            np.array([mbr[0].dPsidf_Voigt for mbr in self.scalar_solution.endmembers]),
            0,
            -1,
        )
        self._dPsidPth_Voigt_mbr = np.moveaxis(
            np.array(
                [mbr[0].dPsidPth_Voigt for mbr in self.scalar_solution.endmembers]
            ),
            0,
            -1,
        )
        self.scalar_solution.set_state(pressure, temperature)

        Material.set_state(self, pressure, temperature)

    def set_state_with_volume(self, volume, temperature):
        self.scalar_solution.set_state_with_volume(volume, temperature)
        self.set_state(self.scalar_solution.pressure, temperature)

    def set_composition(self, molar_fractions):
        self.scalar_solution.set_composition(molar_fractions)
        if self.pressure is not None:
            self.set_state(self.pressure, self.temperature)

    def set_relaxation(self, relax_bool):
        if self.relaxed is not relax_bool:
            self.relaxed = relax_bool
            Material.reset(self)

    @material_property
    def molar_fractions(self):
        return self.scalar_solution.molar_fractions

    @material_property
    def structural_parameters(self):
        return self.scalar_solution.molar_fractions.dot(self.dXdQ)

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
        """
        Returns a tuple containing
        Psi_xs, dPsi_xsdlnV, dPsi_xsdPth and dPsi_xsdX
        """
        psi_xs = self.psi_excess_function(
            self._f, self._Pth_mbr, self.molar_fractions, self.anisotropic_params
        )
        return psi_xs

    @material_property
    def Psi(self):
        """
        Psi
        """
        Psi_mech = np.einsum("ijklm, m->ijkl", self._Psi_mbr, self.molar_fractions)
        return Psi_mech + voigt_notation_to_compliance_tensor(self._Psi_excess_tuple[0])

    @material_property
    def _dPsidf_Voigt(self):
        """
        Gradient in Psi in Voigt notation with respect
        to ln(V) at fixed Pthi
        """
        dPsidf_mech = np.einsum(
            "ijk, k->ij", self._dPsidf_Voigt_mbr, self.molar_fractions
        )
        return dPsidf_mech + self._Psi_excess_tuple[1]

    @material_property
    def dPsidPth_Voigt(self):
        """
        Gradient in Psi in Voigt notation with respect
        to the thermal pressure Pth for each endmember
        in the solution at fixed Pthj!=i and fixed ln(V).

        Return array is therefore three dimensional.
        """
        return self._dPsidPth_Voigt_mbr + self._Psi_excess_tuple[2]

    @material_property
    def dPsidX(self):
        """
        Gradient in Psi with respect to compositional and structural state
        at fixed volume and temperature
        (or equivalently volume and Pthi(volume, temperature))
        """
        Psi_mbr = np.moveaxis(self._Psi_mbr, -1, 0)
        dPsidX_xs = np.moveaxis(self._Psi_excess_tuple[3], -1, 0)
        dPsidX = np.array(
            [
                Psi_mbr[i] + voigt_notation_to_compliance_tensor(dPsidX_xs[i])
                for i in range(len(Psi_mbr))
            ]
        )
        return np.einsum("mijkl->ijklm", dPsidX)

    @material_property
    def dPsidT_fixed_f_Voigt(self):
        """
        Gradient in Psi with respect to temperature
        at fixed volume and chemical/structural state
        """
        dPsidT_Voigt = np.einsum(
            "ijk, k->ij", self.dPsidPth_Voigt, self._aKT_mbr * self.molar_fractions
        )
        return dPsidT_Voigt

    @material_property
    def dPsidf_fixed_T_Voigt(self):
        """
        Gradient in Psi with respect to temperature
        at fixed volume and chemical/structural state
        """
        dPsidf_Voigt = (
            np.einsum(
                "ijk, k->ij",
                self.dPsidPth_Voigt,
                self._dPthdf_mbr * self.molar_fractions,
            )
            + self._dPsidf_Voigt
        )
        return dPsidf_Voigt

    @material_property
    def dPsidT_fixed_P_Voigt(self):
        """
        Gradient in Psi with respect to temperature
        at fixed pressure and chemical/structural state
        """
        dPsidT_Voigt = (
            self.scalar_solution.alpha * self.dPsidf_fixed_T_Voigt
            + self.dPsidT_fixed_f_Voigt
        )
        return dPsidT_Voigt

    @material_property
    def dPsidP_fixed_T_Voigt(self):
        """
        Gradient in Psi with respect to temperature
        at fixed volume and chemical/structural state
        """
        dPsidP_Voigt = -self.scalar_solution.beta_T * self.dPsidf_fixed_T_Voigt
        return dPsidP_Voigt

    @material_property
    def depsdQ(self):
        """
        Gradient in strain with respect to structural state
        at fixed volume and temperature
        """
        return np.einsum("ijklm, mn, kl->ijn", self.dPsidX, self.dXdQ, np.eye(3))

    @material_property
    def depsdT(self):
        """
        Gradient in strain with respect to temperature
        at fixed volume and chemical/structural state
        """
        dPsidT = voigt_notation_to_compliance_tensor(self.dPsidT_fixed_f_Voigt)
        return np.einsum("ijkl, kl->ij", dPsidT, np.eye(3))

    @material_property
    def dSdX(self):
        """
        Partial entropies calculated at constant volume and temperature
        under hydrostatic conditions.
        """
        dSdX = (
            self.scalar_solution.partial_entropies
            - self.scalar_solution.alpha
            * self.scalar_solution.isothermal_bulk_modulus_reuss
            * self.scalar_solution.partial_volumes
        )
        return dSdX

    @material_property
    def dPdX(self):
        """
        Partial pressures calculated at constant volume and temperature
        under hydrostatic conditions.
        """
        dPdX = (
            self.scalar_solution.isothermal_bulk_modulus_reuss
            / self.scalar_solution.V
            * self.scalar_solution.partial_volumes
        )
        return dPdX

    @material_property
    def d2FdXdX(self):
        """
        Helmholtz compositional hessian calculated at constant volume and temperature
        under hydrostatic conditions.
        """
        d2FdXdX = (
            self.scalar_solution.gibbs_hessian
            + np.einsum(
                "i, j->ij",
                self.scalar_solution.partial_volumes,
                self.scalar_solution.partial_volumes,
            )
            * self.scalar_solution.isothermal_bulk_modulus_reuss
            / self.scalar_solution.V
        )
        return d2FdXdX

    @material_property
    def dSdQ(self):
        """
        Entropy gradient with respect to structure parameters
        calculated at constant volume and temperature
        under hydrostatic conditions.
        """
        return np.einsum("i, ij->j", self.dSdX, self.dXdQ)

    @material_property
    def dPdQ(self):
        """
        Pressure gradient with respect to structure parameters
        calculated at constant volume and temperature
        under hydrostatic conditions.
        """
        return np.einsum("i, ij->j", self.dPdX, self.dXdQ)

    @material_property
    def d2FdQdQ_fixed_volume(self):
        """
        Helmholtz compositional hessian
        calculated at constant volume and temperature
        under hydrostatic conditions.
        """
        return np.einsum(
            "ij, ik, jl->kl",
            self.d2FdXdX,
            self.dXdQ,
            self.dXdQ,
        )

    @material_property
    def d2FdQdZ(self):
        """
        Second derivatives of the Helmholtz energy at constant strain and/or temperature.
        """
        beta_T = np.einsum(
            "ijkl, kl", self.full_isothermal_compliance_tensor_unrelaxed, np.eye(3)
        )
        beta_RT = np.einsum("ij, ij", beta_T, np.eye(3))
        dd = IpmIqn - np.einsum("pq, mn->pqmn", beta_T / beta_RT, np.eye(3))
        C_T = self.full_isothermal_stiffness_tensor_unrelaxed
        d2FdQdeps = -self.V * (
            np.einsum("i, mn->imn", self.dPdQ, np.eye(3))
            + np.einsum("kli, klpq, pqmn->imn", self.depsdQ, C_T, dd)
        )
        d2FdQdeps = np.array([contract_stresses(arr) for arr in d2FdQdeps])
        d2FdQdT = -self.dSdQ + self.V * np.einsum(
            "kli, klmn, mn->i", self.depsdQ, C_T, self.depsdT
        )

        return np.concatenate((d2FdQdeps, d2FdQdT[:, np.newaxis]), axis=1)

    @material_property
    def d2FdQdQ_fixed_strain(self):
        """
        Second structure parameter derivative of the Helmholtz energy
        at constant strain and temperature.
        """
        C_T = self.full_isothermal_stiffness_tensor_unrelaxed
        return self.d2FdQdQ_fixed_volume + self.V * np.einsum(
            "kli, klmn, mnj->ij", self.depsdQ, C_T, self.depsdQ
        )

    @material_property
    def d2FdQdQ_fixed_strain_pinv(self):
        return np.linalg.pinv(self.d2FdQdQ_fixed_strain)

    @material_property
    def d2FdZdZ(self):
        """
        Block matrix of V*C_T, V*pi, -c_eps/T
        at fixed Q
        """
        CT = self.isothermal_stiffness_tensor_unrelaxed
        pi = contract_stresses(self.thermal_stress_tensor_unrelaxed)
        c_eps = self.molar_isometric_heat_capacity_unrelaxed
        V = self.molar_volume
        T = self.temperature
        return np.block([[V * CT, V * pi[:, np.newaxis]], [V * pi, -c_eps / T]])

    @material_property
    def d2FdZdZ_relaxed(self):
        """
        Block matrix of V*C_T, V*pi, -c_eps/T
        under Helmholtz-minimizing varying Q
        """
        return self.d2FdZdZ - np.einsum(
            "kl, ki, lj->ij", self.d2FdQdQ_fixed_strain_pinv, self.d2FdQdZ, self.d2FdQdZ
        )

    @material_property
    def isothermal_stiffness_tensor(self):
        if self.relaxed:
            C_T = self.d2FdZdZ_relaxed[:6, :6] / self.V
            if self.orthotropic:
                return C_T
            else:
                R = self.rotation_matrix
                C = voigt_notation_to_stiffness_tensor(C_T)
                C_rotated = np.einsum("mi, nj, ok, pl, ijkl->mnop", R, R, R, R, C)
                return contract_stiffnesses(C_rotated)

        else:
            return self.isothermal_stiffness_tensor_unrelaxed

    @material_property
    def full_isothermal_stiffness_tensor(self):
        CT = self.isothermal_stiffness_tensor
        return voigt_notation_to_stiffness_tensor(CT)

    @material_property
    def thermal_stress_tensor(self):
        if self.relaxed:
            pi = expand_stresses(self.d2FdZdZ_relaxed[6, :6] / self.V)

            if self.orthotropic:
                return pi
            else:
                R = self.rotation_matrix
                return np.einsum("mi, nj, ij->mn", R, R, pi)

        else:
            return self.thermal_stress_tensor_unrelaxed

    @material_property
    def molar_isometric_heat_capacity(self):
        """
        Returns
        -------
        molar_isometric_heat_capacity : 2D numpy array
            The heat capacity at constant strain [J/K/mol]
        """
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
        return voigt_notation_to_compliance_tensor(S_Voigt)

    @material_property
    def full_isentropic_stiffness_tensor(self):
        return self.full_isothermal_stiffness_tensor + (
            self.molar_volume
            * self.temperature
            * np.einsum(
                "ij, kl", self.thermal_stress_tensor, self.thermal_stress_tensor
            )
            / self.molar_isometric_heat_capacity
        )

    @material_property
    def isentropic_stiffness_tensor(self):
        C_full = self.full_isentropic_stiffness_tensor
        return contract_stiffnesses(C_full)

    @material_property
    def isentropic_compliance_tensor(self):
        return np.linalg.inv(self.isentropic_stiffness_tensor)

    @material_property
    def full_isentropic_compliance_tensor(self):
        S_Voigt = self.isentropic_compliance_tensor
        return voigt_notation_to_compliance_tensor(S_Voigt)

    @material_property
    def isothermal_compliance_tensor_unrelaxed(self):
        """
        Returns
        -------
        isothermal_compliance_tensor : 2D numpy array
            The isothermal compliance tensor [1/Pa]
            in Voigt form (:math:`\\mathbb{S}_{\\text{T} pq}`).
        """
        S_T = -self.dPsidP_fixed_T_Voigt
        if self.orthotropic:
            return S_T
        else:
            R = self.rotation_matrix
            S = voigt_notation_to_compliance_tensor(S_T)
            S_rotated = np.einsum("mi, nj, ok, pl, ijkl->mnop", R, R, R, R, S)
            return contract_compliances(S_rotated)

    @material_property
    def thermal_expansivity_tensor_unrelaxed(self):
        """
        Returns
        -------
        thermal_expansivity_tensor : 2D numpy array
            The tensor of thermal expansivities [1/K].
        """
        alpha = np.einsum(
            "ijkl, kl",
            voigt_notation_to_compliance_tensor(self.dPsidT_fixed_P_Voigt),
            np.eye(3),
        )

        if self.orthotropic:
            return alpha
        else:
            R = self.rotation_matrix
            return np.einsum("mi, nj, ij->mn", R, R, alpha)

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
        return voigt_notation_to_stiffness_tensor(CT)

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
        return voigt_notation_to_compliance_tensor(S_Voigt)

    @material_property
    def full_isentropic_compliance_tensor_unrelaxed(self):
        """
        Returns
        -------
        full_isentropic_stiffness_tensor : 4D numpy array
            The isentropic compliance tensor [1/Pa]
            in standard form (:math:`\\mathbb{S}_{\\text{N} ijkl}`).
        """
        return (
            self.full_isothermal_compliance_tensor_unrelaxed
            - np.einsum(
                "ij, kl->ijkl",
                self.thermal_expansivity_tensor_unrelaxed,
                self.thermal_expansivity_tensor_unrelaxed,
            )
            * self.V
            * self.temperature
            / self.scalar_solution.molar_heat_capacity_p
        )

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
        return contract_compliances(S_full)

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
        return voigt_notation_to_stiffness_tensor(C_Voigt)

    @material_property
    def thermal_stress_tensor_unrelaxed(self):
        """
        Returns
        -------
        thermal stress : 2D numpy array
            The change in stress with temperature at constant strain.
        """
        pi = -np.einsum(
            "ijkl, kl",
            self.full_isothermal_stiffness_tensor_unrelaxed,
            self.thermal_expansivity_tensor_unrelaxed,
        )
        return pi

    @material_property
    def molar_isometric_heat_capacity_unrelaxed(self):
        """
        Returns
        -------
        molar_isometric_heat_capacity : float
            The molar heat capacity at constant strain.
        """
        alpha = self.thermal_expansivity_tensor_unrelaxed
        pi = self.thermal_stress_tensor_unrelaxed
        C_isometric = (
            self.scalar_solution.molar_heat_capacity_p
            + self.V * self.temperature * np.einsum("ij, ij", alpha, pi)
        )

        return C_isometric

    @material_property
    def isothermal_bulk_modulus_voigt(self):
        return np.sum(self.isothermal_stiffness_tensor[:3, :3])

    @material_property
    def isothermal_bulk_modulus_reuss(self):
        return 1.0 / np.sum(self.isothermal_compliance_tensor[:3, :3])

    @material_property
    def isothermal_bulk_modulus(self):
        """
        Anisotropic minerals do not have a single isothermal bulk modulus.
        This function returns a NotImplementedError. Users should instead
        consider either using isothermal_bulk_modulus_reuss,
        isothermal_bulk_modulus_voigt,
        or directly querying the elements in the isothermal_stiffness_tensor.
        """
        raise NotImplementedError(
            "isothermal_bulk_modulus is not "
            "sufficiently explicit for an "
            "anisotropic mineral. Did you mean "
            "isothermal_bulk_modulus_reuss?"
        )

    @material_property
    def thermal_expansivity_tensor(self):
        """
        Returns
        -------
        thermal_expansivity_tensor : 2D numpy array
            The tensor of thermal expansivities [1/K].
        """
        if self.relaxed:
            alpha = -np.einsum(
                "ijkl, kl",
                self.full_isothermal_compliance_tensor,
                self.thermal_stress_tensor,
            )
            return alpha
        else:
            return self.thermal_expansivity_tensor_unrelaxed

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
        pi = self.thermal_stress_tensor
        C_p = (
            self.molar_isometric_heat_capacity
            - self.V * self.temperature * np.einsum("ij, ij", alpha, pi)
        )
        return C_p

    @material_property
    def molar_heat_capacity_v(self):
        return (
            self.molar_heat_capacity_p
            - self.molar_volume
            * self.temperature
            * self.thermal_expansivity
            * self.thermal_expansivity
            * self.isothermal_bulk_modulus_reuss
        )

    @material_property
    def isentropic_thermal_gradient(self):
        """
        Returns
        -------
        dTdP : float
            The change in temperature with pressure at constant entropy [Pa/K]
        """
        return (
            self.molar_volume * self.temperature * self.thermal_expansivity
        ) / self.molar_heat_capacity_p
