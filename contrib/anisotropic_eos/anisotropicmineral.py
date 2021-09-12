# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.
import numpy as np
from scipy.linalg import expm
from numpy.linalg import cond
from warnings import warn
from burnman import Mineral, Material
from burnman.anisotropy import AnisotropicMaterial

def cell_parameters_to_vectors(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    Convert cell parameters from a, b, c, alpha, beta, gamma (in degrees)
    to the unit cell vectors
    Scrounged from https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography
    """
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    n2 = (np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma)
    M  = np.array([[a,0,0],
                   [b*np.cos(gamma),b*np.sin(gamma),0],
                   [c*np.cos(beta),c*n2,c*np.sqrt(np.sin(beta)**2-n2**2)]])
    return M

def cell_vectors_to_parameters(M):
    """
    Convert unit cell vectors to
    cell parameters in the format a, b, c, alpha, beta, gamma (in degrees)
    """

    assert M[0,1] == 0
    assert M[0,2] == 0
    assert M[1,2] == 0

    a = M[0,0]
    b = np.sqrt(np.power(M[1,0], 2.) + np.power(M[1,1], 2.))
    c = np.sqrt(np.power(M[2,0], 2.) + np.power(M[2,1], 2.) + np.power(M[2,2], 2.))

    gamma = np.arccos(M[1,0] / b)
    beta = np.arccos(M[2,0] / c)
    alpha = np.arccos(M[2,1]/c*np.sin(gamma) + np.cos(gamma)*np.cos(beta))

    gamma_deg = np.degrees(gamma)
    beta_deg = np.degrees(beta)
    alpha_deg = np.degrees(alpha)

    return (a, b, c, alpha_deg, beta_deg, gamma_deg)


class AnisotropicMineral(Mineral, AnisotropicMaterial):
    """

    Myhill (2021)
    """

    def __init__(self, isotropic_material, cell_parameters, anisotropic_parameters):

        assert (np.all(anisotropic_parameters[:,:,0,0] == 0)), "anisotropic_parameters_pqmn should be set to zero for all m = n = 0"
        sum_ijij_block = np.sum(anisotropic_parameters[:3,:3,:,:], axis=(0,1))
        assert (np.abs(sum_ijij_block[1,0] - 1.) < 1.e-5), f'The sum of the upper 3x3 pq-block of anisotropic_parameters_pqmn must equal 1 for m=1, n=0 for consistency with the volume. Value is {np.abs(sum_ijij_block[1,0]) - 1.}'
        assert (np.all(np.abs(sum_ijij_block[2:,0]) < 1.e-10)), "The sum of the upper 3x3 pq-block of anisotropic_parameters_pqmn must equal 0 for all m > 1 for consistency with the volume"
        assert (np.all(np.abs(sum_ijij_block[:,1:]) < 1.e-10)), "The sum of the upper 3x3 pq-block of anisotropic_parameters_pqmn must equal 0 for all n > 0 for consistency with the volume"

        assert (cond(anisotropic_parameters[:,:,1,0]) < 1/np.finfo(float).eps), "anisotropic_parameters[:,:,1,0] is singular"

        sum_lower_off_diagonal_block = np.sum(anisotropic_parameters[3:,:3,:,:], axis=1)
        for i, s in enumerate(sum_lower_off_diagonal_block):
            if not np.all(s == 0):
                warn(f'This material appears to be monoclinic or triclinic. Rotations are not yet accounted for.', stacklevel=2)

        self.cell_vectors_0 = cell_parameters_to_vectors(*cell_parameters)
        assert (np.abs(np.linalg.det(self.cell_vectors_0) - isotropic_material.params['V_0']) < np.finfo(float).eps)

        self.c = anisotropic_parameters

        if 'name' in isotropic_material.params:
            self.name = isotropic_material.params['name']

        Mineral.__init__(self, isotropic_material.params,
                         isotropic_material.property_modifiers)

    def set_state(self, pressure, temperature):

        # 1) Compute dPthdf
        dP = 1000.
        Mineral.set_state(self, pressure-dP/2., temperature)
        V1 = self.V
        Pth1 = pressure-dP/2. - self.method.pressure(self.params['T_0'],
                                                     V1, self.params)

        Mineral.set_state(self, pressure+dP/2., temperature)
        V2 = self.V
        Pth2 = pressure+dP/2. - self.method.pressure(self.params['T_0'],
                                                     V2, self.params)

        self.dPthdf = (Pth2 - Pth1) / np.log(V2/V1)
        Mineral.set_state(self, pressure, temperature)

        # 2) Compute other properties needed for anisotropic equation of state
        V = self.V
        V_0 = self.params['V_0']
        Vrel = V/V_0
        f = np.log(Vrel)

        Pth = pressure - self.method.pressure(self.params['T_0'], self.V,
                                              self.params)

        # Compute X, dXdPth, dXdf, needed by most anisotropic properties
        ns = np.arange(self.c.shape[-1])
        x = self.c[:,:,0,:] + self.c[:,:,1,:]*f
        dxdf = self.c[:,:,1,:]

        for i in list(range(2, self.c.shape[2])):
            # non-intuitively, the += operator doesn't simply add in-place,
            # so here we overwrite the arrays with new ones
            x = x + self.c[:,:,i,:]*np.power(f, float(i))/float(i)
            dxdf = dxdf + self.c[:,:,i,:] * np.power(f, float(i)-1.)

        self._Vrel = Vrel
        self._f = f
        self.X_Voigt = np.einsum('ikn, n->ik', x, np.power(Pth, ns))
        self.dXdPth_Voigt = np.einsum('ikn, n->ik',
                                      x[:,:,1:], ns[1:]*np.power(Pth, ns[1:]-1))
        self.dXdf_Voigt = np.einsum('ikn, n->ik', dxdf, np.power(Pth, ns))

    def _contract_compliances(self, compliances):
        try: # numpy.block was new in numpy version 1.13.0.
            block = np.block([[ np.ones((3, 3)), 2.*np.ones((3, 3))],
                              [2.*np.ones((3, 3)), 4.*np.ones((3, 3))]])
        except:
            block = np.array(np.bmat( [[[[1.]*3]*3, [[2.]*3]*3], [[[2.]*3]*3, [[4.]*3]*3]] ))

        voigt_notation = np.zeros((6, 6))
        for m in range(6):
            i, j = self._voigt_index_to_ij(m)
            for n in range(6):
                k, l = self._voigt_index_to_ij(n)
                voigt_notation[m,n] = compliances[i,j,k,l]
        return np.multiply(voigt_notation, block)


    @property
    def deformation_gradient_tensor(self):
        F = expm(np.einsum('ijkl, kl',
                           self._voigt_notation_to_compliance_tensor(self.X_Voigt),
                           np.eye(3)))
        return F

    @property
    def cell_vectors(self):
        return self.deformation_gradient_tensor.dot(self.cell_vectors_0)

    @property
    def cell_parameters(self):
        return cell_vectors_to_parameters(self.cell_vectors)

    @property
    def shear_modulus(self):
        raise NotImplementedError("anisotropic materials do not have a shear modulus property. Return elements of the stiffness tensor instead")

    @property
    def isothermal_bulk_modulus(self):
        raise NotImplementedError("isothermal_bulk_modulus is not sufficiently explicit for an anisotropic material. Did you mean isothermal_bulk_modulus_reuss?")

    @property
    def isentropic_bulk_modulus(self):
        raise NotImplementedError("isentropic_bulk_modulus is not sufficiently explicit for an anisotropic material. Did you mean isentropic_bulk_modulus_reuss?")

    isothermal_bulk_modulus_reuss = Mineral.isothermal_bulk_modulus

    @property
    def isothermal_compressibility_reuss(self):
        return 1./self.isothermal_bulk_modulus_reuss

    @property
    def isothermal_compliance_tensor(self):
        return self.isothermal_compressibility_reuss * (self.dXdf_Voigt + self.dXdPth_Voigt * self.dPthdf)

    @property
    def isothermal_stiffness_tensor(self):
        return np.linalg.inv(self.isothermal_compliance_tensor)

    @property
    def full_isothermal_compliance_tensor(self):
        return self._voigt_notation_to_compliance_tensor(self.isothermal_compliance_tensor)


    @property
    def full_isothermal_stiffness_tensor(self):
        return self._expand_stiffnesses(self.isothermal_stiffness_tensor)

    @property
    def thermal_expansivity_tensor(self):
        a = self.alpha * (self.dXdf_Voigt
                          + self.dXdPth_Voigt * (self.dPthdf
                                                 - 1./self.isothermal_compressibility_reuss))
        return np.einsum('ijkl, kl',
                         self._voigt_notation_to_compliance_tensor(a),
                         np.eye(3))

    @property
    def full_isentropic_compliance_tensor(self):
        return (self.full_isothermal_compliance_tensor
                - np.einsum('ij, kl->ijkl',
                            self.thermal_expansivity_tensor,
                            self.thermal_expansivity_tensor)
                * self.V * self.temperature / self.C_p)

    @property
    def isentropic_compliance_tensor(self):
        return self._contract_compliances(self.full_isentropic_compliance_tensor)

    @property
    def isentropic_stiffness_tensor(self):
        return np.linalg.inv(self.isentropic_compliance_tensor)
