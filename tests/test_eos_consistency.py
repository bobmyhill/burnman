from __future__ import absolute_import
import unittest
from util import BurnManTest

import burnman

from burnman.tools.eos import check_eos_consistency

class mypericlase4th(burnman.Mineral):

    """
    Stixrude & Lithgow-Bertelloni 2005 and references therein
    """

    def __init__(self):
        self.params = {
            'equation_of_state': 'slb4',
            'T_0': 300.,
            'P_0': 0.,
            'F_0': -569444.6,
            'V_0': 1.1244e-05,
            'K_0': 1.613836e+11,
            'Kprime_0': 3.84045,
            'Kprime_prime_0': -3.84045/1.613836e+11,
            'Debye_0': 767.0977,
            'grueneisen_0': 1.36127,
            'q_0': 1.7217,
            'G_0': 1.309e+11,
            'Gprime_0': 2.1438,
            'eta_s_0': 2.81765,
            'molar_mass': .0403,
            'n': 2}
        burnman.Mineral.__init__(self)

class EosConsistency(BurnManTest):

    def test_HP(self):
        P = 10.e9
        T = 3000.
        self.assertEqual(check_eos_consistency(
            burnman.minerals.HP_2011_ds62.per(), P, T, including_shear_properties=False), True)

    def test_SLB(self):
        P = 10.e9
        T = 3000.
        self.assertEqual(check_eos_consistency(burnman.minerals.SLB_2011.periclase(), P, T),
                         True)

    def test_modifier(self):
        P = 10.e9
        T = 3000.
        self.assertEqual(check_eos_consistency(
            burnman.minerals.Sundman_1991.bcc_iron(), P, T, including_shear_properties=False), True)

    def test_solution(self):
        P = 10.e9
        T = 3000.
        m = burnman.minerals.SLB_2011.garnet(
            molar_fractions=[0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(check_eos_consistency(m, P, T),
                         True)

    def test_slb4th(self):
        m = mypericlase4th()
        P = 10.e9
        T = 3000.
        self.assertEqual(check_eos_consistency(m, P, T, including_shear_properties=False),
                         True)


if __name__ == '__main__':
    unittest.main()
