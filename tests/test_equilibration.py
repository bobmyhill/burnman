import unittest
from util import BurnManTest

import burnman
from burnman import equilibrate
from burnman.optimize.nonlinear_solvers import TerminationCode
from burnman.minerals import HP_2011_ds62, SLB_2011

import numpy as np


def make_ol_wad_assemblage():
    ol = SLB_2011.mg_fe_olivine()
    wad = SLB_2011.mg_fe_wadsleyite()

    assemblage = burnman.Composite([ol, wad], [0.7, 0.3])
    ol.set_composition([0.5, 0.5])
    wad.set_composition([0.6, 0.4])
    return assemblage


class equilibration(BurnManTest):
    def test_univariant_line(self):
        andalusite = HP_2011_ds62.andalusite()
        kyanite = HP_2011_ds62.ky()
        composition = kyanite.formula

        pressures = np.linspace(1.0e5, 1.0e9, 11)

        assemblage = burnman.Composite([andalusite, kyanite])
        equality_constraints = [
            ("P", pressures),
            ("phase_fraction", (andalusite, np.array([0.0]))),
        ]
        sols, prm = equilibrate(composition, assemblage, equality_constraints)

        Ts = [sol.assemblage.temperature for sol in sols]
        Ts_ref = [
            464.95783870074905,
            544.0828494722284,
            623.6421086627697,
            703.7607424193357,
            784.4975351197959,
            865.8789417005959,
            947.9152419887187,
            1030.608906228574,
            1113.95923718214,
            1197.9651099824446,
            1282.6266945380332,
        ]
        self.assertArraysAlmostEqual(Ts, Ts_ref)

    def test_invariant(self):
        sillimanite = HP_2011_ds62.sill()
        andalusite = HP_2011_ds62.andalusite()
        kyanite = HP_2011_ds62.ky()

        composition = sillimanite.formula
        assemblage = burnman.Composite([sillimanite, andalusite, kyanite])
        equality_constraints = [
            ("phase_fraction", (kyanite, np.array([0.0]))),
            ("phase_fraction", (sillimanite, np.array([0.0]))),
        ]

        sol, prm = equilibrate(composition, assemblage, equality_constraints)

        self.assertArraysAlmostEqual(
            sol.x, [4.30671426e08, 8.09342875e02, 0.0, 1.0, 0.0]
        )

    def test_univariant_line_fo(self):
        forsterite = SLB_2011.forsterite()
        periclase = SLB_2011.periclase()
        bridgmanite = SLB_2011.mg_bridgmanite()
        composition = forsterite.formula

        temperatures = np.linspace(300.0, 2000.0, 11)

        assemblage = burnman.Composite([forsterite, periclase, bridgmanite])
        equality_constraints = [
            ("T", temperatures),
            ("phase_fraction", (forsterite, np.array([0.0]))),
        ]
        sols, prm = equilibrate(composition, assemblage, equality_constraints)

        Ps = np.array([sol.assemblage.pressure for sol in sols])
        Ps_ref = [
            17405868549,
            17657896452,
            17941848050,
            18231743600,
            18517274303,
            18793435710,
            19057314295,
            19306923194,
            19540715743,
            19757356711,
            19955602937,
        ]
        self.assertArraysAlmostEqual(Ps, Ps_ref)

    def test_ol_wad_eqm(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        wad = assemblage.phases[1]
        assemblage.set_state(10.0e9, 1200.0)
        equality_constraints = [
            ("P", 10.0e9),
            (
                "phase_composition",
                (ol, (["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], 0.45)),
            ),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}

        sol, prm = equilibrate(composition, assemblage, equality_constraints)
        self.assertArraysAlmostEqual(
            [assemblage.temperature, ol.molar_fractions[1], wad.molar_fractions[1]],
            [1620.532183457096, 0.45, 0.6791743],
        )

    def test_binary_solution_convergence(self):
        mg_bdg = SLB_2011.mg_bridgmanite()
        fe_bdg = SLB_2011.fe_bridgmanite()
        mg_ppv = SLB_2011.mg_post_perovskite()
        fe_ppv = SLB_2011.fe_post_perovskite()
        bdg_endmembers = [[mg_bdg, "[Mg]"], [fe_bdg, "[Fe]"]]
        ppv_endmembers = [[mg_ppv, "[Mg]"], [fe_ppv, "[Fe]"]]

        bdg = burnman.Solution(
            "bdg",
            solution_model=burnman.classes.solutionmodel.IdealSolution(bdg_endmembers),
        )
        ppv = burnman.Solution(
            "ppv",
            solution_model=burnman.classes.solutionmodel.IdealSolution(ppv_endmembers),
        )

        composition = {"Mg": 0.9, "Fe": 0.1, "Si": 1.0, "O": 3.0}
        assemblage = burnman.Composite(
            phases=[bdg, ppv], fractions=[0.9, 0.1], name="MgSiO3-pv-ppv-assemblage"
        )
        bdg.set_composition([0.9, 0.1])
        ppv.set_composition([0.9, 0.1])

        temperatures = np.linspace(1000, 4000, 4)
        assemblage.set_state(120.0e9, temperatures[0])

        equality_constraints = [("T", temperatures), ("phase_fraction", (ppv, 0.0))]

        sol, prm = equilibrate(composition, assemblage, equality_constraints)
        self.assertFalse(any(not s.success for s in sol))

    def test_incorrect_tol(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        assemblage.set_state(10.0e9, 1200.0)
        equality_constraints = [
            ("P", 10.0e9),
            (
                "phase_composition",
                (ol, (["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], 0.45)),
            ),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}

        # These should raise errors
        with self.assertRaises(AssertionError):
            _ = equilibrate(
                composition, assemblage, equality_constraints, tol=[1.0e-3, 1.0e-3]
            )
        with self.assertRaises(AssertionError):
            _ = equilibrate(
                composition, assemblage, equality_constraints, tol=[[1.0e-3] * 6] * 2
            )

        # These should work
        _ = equilibrate(composition, assemblage, equality_constraints, tol=1.0e-3)
        _ = equilibrate(
            composition,
            assemblage,
            equality_constraints,
            tol=[1.0e-3] * 6,
        )
        _ = equilibrate(composition, assemblage, equality_constraints)

    def test_ol_wad_eqm_entropy(self):
        assemblage = make_ol_wad_assemblage()
        wad = assemblage.phases[1]
        equality_constraints = [
            ("P", 10.0e9),
            ("phase_fraction", (wad, 0.5)),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}

        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)

        P = assemblage.pressure
        T = assemblage.temperature
        S = assemblage.molar_entropy * assemblage.n_moles

        assemblage = make_ol_wad_assemblage()
        equality_constraints = [("P", P), ("S", S)]
        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(assemblage.temperature, T, places=5)

    def test_ol_wad_eqm_volume(self):
        assemblage = make_ol_wad_assemblage()
        wad = assemblage.phases[1]
        equality_constraints = [
            ("P", 10.0e9),
            ("phase_fraction", (wad, 0.5)),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}

        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)

        P = assemblage.pressure
        T = assemblage.temperature
        V = assemblage.molar_volume * assemblage.n_moles

        assemblage = make_ol_wad_assemblage()
        equality_constraints = [("P", P), ("V", V)]
        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(assemblage.temperature, T, places=5)

    def test_ol_wad_eqm_compositional_constraint(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        wad = assemblage.phases[1]
        assemblage.set_state(10.0e9, 1200.0)
        equality_constraints = [
            ("P", 10.0e9),
            ("X", [[0.0, 0.0, 0.0, 2.0, 0.0, 0.0], 0.9]),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}

        _, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertArraysAlmostEqual(
            [assemblage.temperature, ol.molar_fractions[1], wad.molar_fractions[1]],
            [1620.532183457096, 0.45, 0.6791743],
        )

    def test_convergence_with_singular_system(self):

        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        a = burnman.Composite(
            [SLB_2011.mg_fe_olivine(), SLB_2011.mg_fe_olivine()], [0.5, 0.5]
        )

        a.phases[0].set_composition([0.1, 0.9])
        a.phases[1].set_composition([0.9, 0.1])
        equality_constraints = [("P", 1.0e5), ("T", 1000.0)]
        sol, _ = equilibrate(composition, a, equality_constraints)
        self.assertTrue(sol.success)  # Expect successful convergence
        self.assertEqual(sol.code, TerminationCode.SINGULAR_SUCCESS)

    def test_ill_posed_problem(self):

        composition = {"Mg": 1.8, "Fe": 0.2, "Si": 1.0, "O": 4.0}
        a = burnman.Composite(
            [SLB_2011.mg_fe_olivine(), SLB_2011.mg_fe_olivine()], [0.5, 0.5]
        )

        a.phases[0].set_composition([0.4, 0.6])
        a.phases[1].set_composition([0.5, 0.5])
        equality_constraints = [("P", 1.0e5), ("T", 300.0)]
        sol, _ = equilibrate(composition, a, equality_constraints)
        self.assertEqual(sol.code, TerminationCode.CONSTRAINT_VIOLATION)

    def test_ill_conditioned_jacobian(self):

        composition = {"Mg": 1.8, "Fe": 0.2, "Si": 1.0, "O": 4.0}
        a = burnman.Composite(
            [SLB_2011.mg_fe_olivine(), SLB_2011.mg_fe_olivine()], [0.5, 0.5]
        )

        a.phases[0].set_composition([0.4, 0.6])
        a.phases[1].set_composition([0.4, 0.6])
        equality_constraints = [("P", 1.0e5), ("T", 300.0)]
        sol, _ = equilibrate(composition, a, equality_constraints)

        # Different testers give different results here
        if sol.code == TerminationCode.SINGULAR_SUCCESS:
            # In some cases the solver converges despite the singular Jacobian
            # In that case require that the two identical phases have the same composition
            self.assertArraysAlmostEqual(a.phases[0].molar_fractions, [0.9, 0.1])
            self.assertArraysAlmostEqual(a.phases[1].molar_fractions, [0.9, 0.1])
        else:
            # In other cases the solver fails due to the singular Jacobian
            self.assertEqual(sol.code, TerminationCode.SINGULAR_FAIL)


if __name__ == "__main__":
    unittest.main()
