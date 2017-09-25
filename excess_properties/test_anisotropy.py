from __future__ import absolute_import
import unittest
import inspect
import os
import sys
sys.path.insert(1, os.path.abspath('..'))
import numpy as np

import burnman
from burnman import anisotropy

Cijm = np.array([1., 2., 3., 4., 5., 6., 7., 8, 9, 10., 11., 12., 13.])
Cijn = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.1, 9.2, 10.3, 11.4, 12.5, 13.6])


m = anisotropy.MonoclinicMaterial(3000., Cijm)
print(m.stiffness_tensor)

n = anisotropy.MonoclinicMaterial(3000., Cijn)
#print(n.stiffness_tensor, n.compliance_tensor)

Sij = 1./(0.5/m.compliance_tensor + 0.5/n.compliance_tensor
Cij = np.linalg.inv(Sij)
p = anisotropy.AnisotropicMaterial(3000., Cij)


print(m.bulk_modulus_reuss, n.bulk_modulus_reuss, p.bulk_modulus_reuss)

'''
Sij = 0.5*m.compliance_tensor + 0.5*n.compliance_tensor
Cij = np.linalg.inv(Sij)
p = anisotropy.AnisotropicMaterial(3000., Cij)

print(Sij)
print(p.compliance_tensor)
print(m.bulk_modulus_reuss, n.bulk_modulus_reuss, p.bulk_modulus_reuss)
print(1./(0.5/m.bulk_modulus_reuss + 0.5/n.bulk_modulus_reuss))
'''
