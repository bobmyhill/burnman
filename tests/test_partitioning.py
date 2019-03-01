from __future__ import absolute_import
import unittest
from util import BurnManTest
import os
import sys
sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.processchemistry import convert_formula, calculate_potential_phase_amounts

class test(BurnManTest):

    def test_simple(self):
        bulk_composition_wt = {'Mg': 0.213, 'Fe': 0.0626,
                               'Si': 0.242, 'Ca': 0., 'Al': 0.}
        bulk_composition_mol = convert_formula(bulk_composition_wt,
                                               to_type='molar')
        
        norm = bulk_composition_mol['Mg'] + bulk_composition_mol['Fe']
        norm_bulk = {element: n/norm for (element, n) in bulk_composition_mol.items()}

        per = minerals.SLB_2011.ferropericlase()
        bdg = minerals.SLB_2011.mg_fe_bridgmanite()
        formulae = per.endmember_formulae
        formulae.extend(bdg.endmember_formulae)
        
        phase_amounts = calculate_potential_phase_amounts(bulk_composition_mol, formulae)
        f_per = sum(phase_amounts[0:2])/sum(phase_amounts)
        self.assertFloatEqual(f_per, 0.12828483)
        
        pressure = 23.83e9 # Pa
        temperature = 2000. # K
        (a, b) = burnman.calculate_nakajima_fp_pv_partition_coefficient(
            pressure, temperature, norm_bulk, 0.5)
        self.assertFloatEqual(a, 0.184533288)
        self.assertFloatEqual(b, 0.102937268)

        g2 = minerals.JH_2015.garnet()
        assemblage = burnman.Composite([g2])
        
        assemblage.set_potential_composition_from_bulk(g.formula,
                                                       unfitted_elements='O',
                                                       use_solution_guesses=False)
        self.assertArraysAlmostEqual(composition, assemblage.phases[0].molar_fractions)
        
if __name__ == '__main__':
    unittest.main()
