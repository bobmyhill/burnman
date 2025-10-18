import unittest
from util import BurnManTest
import warnings

import burnman
from burnman import minerals
from burnman.eos import debye
from burnman.utils.chemistry import dictionarize_formula, formula_mass
import logging

logging.basicConfig(level=logging.CRITICAL)


class mypericlase(burnman.Mineral):
    """
    Stixrude & Lithgow-Bertelloni 2005 and references therein
    """

    def __init__(self):
        self.params = {
            "equation_of_state": "slb3",
            "T_0": 300.0,
            "P_0": 0.0,
            "V_0": 11.24e-6,
            "K_0": 161.0e9,
            "Kprime_0": 3.8,
            "G_0": 131.0e9,
            "Gprime_0": 2.1,
            "molar_mass": 0.0403,
            "n": 2,
            "Debye_0": 773.0,
            "grueneisen_0": 1.5,
            "q_0": 1.5,
            "eta_s_0": 2.8,
        }
        burnman.Mineral.__init__(self)


class Fe_Dewaele(burnman.Mineral):
    """
    Dewaele et al., 2006, Physical Review Letters
    """

    def __init__(self):
        self.params = {
            "equation_of_state": "vinet",
            "P_0": 0.0,
            "V_0": 6.75e-6,
            "K_0": 163.4e9,
            "Kprime_0": 5.38,
            "molar_mass": 0.055845,
            "n": 1,
        }


class Liquid_Fe_Anderson(burnman.Mineral):
    """
    Anderson & Ahrens, 1994 JGR
    """

    def __init__(self):
        self.params = {
            "equation_of_state": "bm4",
            "P_0": 0.0,
            "V_0": 7.95626e-6,
            "K_0": 109.7e9,
            "Kprime_0": 4.66,
            "Kprime_prime_0": -0.043e-9,
            "molar_mass": 0.055845,
        }


class outer_core_rkprime(burnman.Mineral):
    """
    Stacey and Davis, 2004 PEPI (Table 5)
    """

    def __init__(self):
        self.params = {
            "equation_of_state": "rkprime",
            "P_0": 0.0,
            "V_0": 0.055845 / 6562.54,
            "K_0": 124.553e9,
            "Kprime_0": 4.9599,
            "Kprime_inf": 3.0,
            "molar_mass": 0.055845,
        }


class periclase_morse(burnman.Mineral):
    """
    Periclase parameters from SLB dataset (which uses BM3)
    """

    def __init__(self):
        formula = "MgO"
        formula = dictionarize_formula(formula)
        self.params = {
            "name": "Periclase",
            "formula": formula,
            "equation_of_state": "morse",
            "P_0": 0.0,
            "V_0": 1.1244e-05,
            "K_0": 1.613836e11,
            "Kprime_0": 3.84045,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }


class alpha_quartz(burnman.Mineral):
    def __init__(self):
        formula = "SiO2"
        formula = dictionarize_formula(formula)
        self.params = {
            "name": "alpha quartz",
            "formula": formula,
            "equation_of_state": "slb3",
            "F_0": -2313.86317911,
            "V_0": 2.2761615699999998e-05,
            "K_0": 74803223700.0,
            "Kprime_0": 7.01334832,
            "Debye_0": 1069.02139,
            "grueneisen_0": -0.0615759576,
            "q_0": 10.940017,
            "G_0": 44856170000.0,
            "Gprime_0": 0.95315,
            "eta_s_0": 2.36469,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }

        self.property_modifiers = [
            ["einstein", {"Theta_0": 510.158911, "Cv_inf": 39.9366924855}],
            ["einstein", {"Theta_0": 801.41137578, "Cv_inf": -52.9869089}],
            ["einstein", {"Theta_0": 2109.4459429999997, "Cv_inf": 13.0502164145}],
        ]

        burnman.Mineral.__init__(self)


class eos(BurnManTest):
    def test_reference_values(self):
        rock = mypericlase()
        pressure = 0.0
        temperature = 300.0
        eoses = [
            burnman.eos.SLB2(),
            burnman.eos.SLB3(),
            burnman.eos.BM3Shear2(),
            burnman.eos.BM3(),
        ]

        for i in eoses:
            Volume_test = i.volume(pressure, temperature, rock.params)
            self.assertFloatEqual(Volume_test, rock.params["V_0"])
            Kt_test = i.isothermal_bulk_modulus_reuss(
                pressure, 300.0, rock.params["V_0"], rock.params
            )
            self.assertFloatEqual(Kt_test, rock.params["K_0"])

            G_test = i.shear_modulus(
                pressure, temperature, rock.params["V_0"], rock.params
            )
            self.assertFloatEqual(G_test, rock.params["G_0"])
            Density_test = i.density(rock.params["V_0"], rock.params)
            self.assertFloatEqual(
                Density_test, rock.params["molar_mass"] / rock.params["V_0"]
            )
            alpha_test = i.thermal_expansivity(
                pressure, temperature, rock.params["V_0"], rock.params
            )
            Cp_test = i.molar_heat_capacity_p(
                pressure, temperature, rock.params["V_0"], rock.params
            )

        eoses_thermal = [burnman.eos.SLB2(), burnman.eos.SLB3()]
        for i in eoses_thermal:
            rock.set_method(i)
            rock.set_state(pressure, temperature)
            Cp_test = rock.molar_heat_capacity_p
            self.assertFloatEqual(Cp_test, 37.076768469502042)
            Cv_test = rock.molar_heat_capacity_v
            self.assertFloatEqual(Cv_test, 36.577717628901553)
            alpha_test = rock.thermal_expansivity
            self.assertFloatEqual(alpha_test, 3.031905596878513e-05)
            Grun_test = rock.grueneisen_parameter
            self.assertFloatEqual(Grun_test, rock.params["grueneisen_0"])

    def test_reference_values_vinet(self):
        rock = Fe_Dewaele()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.eos.Vinet()

        Volume_test = eos.volume(pressure, temperature, rock.params)
        self.assertFloatEqual(Volume_test, rock.params["V_0"])
        Kt_test = eos.isothermal_bulk_modulus_reuss(
            pressure, 300.0, rock.params["V_0"], rock.params
        )
        self.assertFloatEqual(Kt_test, rock.params["K_0"])
        Density_test = eos.density(rock.params["V_0"], rock.params)
        self.assertFloatEqual(
            Density_test, rock.params["molar_mass"] / rock.params["V_0"]
        )

    def test_reference_values_bm4(self):
        rock = Liquid_Fe_Anderson()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.eos.BM4()

        Volume_test = eos.volume(pressure, temperature, rock.params)
        self.assertFloatEqual(Volume_test, rock.params["V_0"])
        Kt_test = eos.isothermal_bulk_modulus_reuss(
            pressure, 300.0, rock.params["V_0"], rock.params
        )
        self.assertFloatEqual(Kt_test, rock.params["K_0"])
        Density_test = eos.density(rock.params["V_0"], rock.params)
        self.assertFloatEqual(
            Density_test, rock.params["molar_mass"] / rock.params["V_0"]
        )

    def test_reference_values_morse(self):
        rock = periclase_morse()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.eos.Morse()
        Volume_test = eos.volume(pressure, temperature, rock.params)
        self.assertFloatEqual(Volume_test, rock.params["V_0"])
        Kt_test = eos.isothermal_bulk_modulus_reuss(
            pressure, 300.0, rock.params["V_0"], rock.params
        )
        self.assertFloatEqual(Kt_test, rock.params["K_0"])
        Density_test = eos.density(rock.params["V_0"], rock.params)
        self.assertFloatEqual(
            Density_test, rock.params["molar_mass"] / rock.params["V_0"]
        )

    def test_reference_values_rkprime(self):
        rock = outer_core_rkprime()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.eos.RKprime()
        Volume_test = eos.volume(pressure, temperature, rock.params)
        self.assertFloatEqual(Volume_test, rock.params["V_0"])
        Kt_test = eos.isothermal_bulk_modulus_reuss(
            pressure, 300.0, rock.params["V_0"], rock.params
        )
        self.assertFloatEqual(Kt_test, rock.params["K_0"])
        Density_test = eos.density(rock.params["V_0"], rock.params)
        self.assertFloatEqual(
            Density_test, rock.params["molar_mass"] / rock.params["V_0"]
        )

    def test_reference_values_aa(self):
        m = minerals.other.liquid_iron()
        pressure = m.params["P_0"]
        temperature = m.params["T_0"]
        rho0 = m.params["molar_mass"] / m.params["V_0"]
        m.set_state(pressure, temperature)
        self.assertFloatEqual(m.V, m.params["V_0"])
        Ks = m.isentropic_bulk_modulus_reuss
        self.assertFloatEqual(Ks, m.params["K_S"])
        self.assertFloatEqual(m.density, rho0)


class test_eos_validation(BurnManTest):
    def test_no_shear_error(self):
        # The validation should place nans in for the shear parameters
        # If any exceptions or warnings are raised, fail.
        class mymineralwithoutshear(burnman.Mineral):
            def __init__(self):
                self.params = {
                    "equation_of_state": "slb3",
                    "V_0": 11.24e-6,
                    "K_0": 161.0e9,
                    "Kprime_0": 3.8,
                    "molar_mass": 0.0403,
                    "n": 2,
                    "Debye_0": 773.0,
                    "grueneisen_0": 1.5,
                    "q_0": 1.5,
                    "eta_s_0": 2.8,
                }
                burnman.Mineral.__init__(self)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warning
            shearless = mymineralwithoutshear()
            if len(w) != 0:
                self.fail("Caught unexpected warning: " + str(w[-1]))
            try:
                x = shearless.params["G_0"]
                y = shearless.params["Gprime_0"]
                z = shearless.params["F_0"]
                z = shearless.params["eta_s_0"]
            except KeyError:
                self.fail("Parameter padding failed in validation")
                pass

    def test_dumb_parameter_values(self):
        class mymineralwithnegativekprime(burnman.Mineral):
            def __init__(self):
                self.params = {
                    "equation_of_state": "slb3",
                    "V_0": 11.24e-6,
                    "K_0": 161.0e9,
                    "Kprime_0": -4.0,
                    "molar_mass": 0.0403,
                    "n": 2,
                    "Debye_0": 773.0,
                    "grueneisen_0": 1.5,
                    "q_0": 1.5,
                    "eta_s_0": 2.8,
                }
                burnman.Mineral.__init__(self)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warning
            negative_Kprime = mymineralwithnegativekprime()
            if len(w) == 0:
                print(negative_Kprime.params)
                self.fail("Did not catch expected warning for negative K prime")

        class mymineralwithkingigapascals(burnman.Mineral):
            def __init__(self):
                self.params = {
                    "equation_of_state": "slb3",
                    "V_0": 11.24e-6,
                    "K_0": 161.0,
                    "Kprime_0": 3.8,
                    "molar_mass": 0.0403,
                    "n": 3.14159,
                    "Debye_0": 773.0,
                    "grueneisen_0": 1.5,
                    "q_0": 1.5,
                    "eta_s_0": 2.8,
                }
                burnman.Mineral.__init__(self)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warning
            low_K = mymineralwithkingigapascals()
            if len(w) == 0:
                self.fail("Did not catch expected warning K in GPa")

    def test_reference_energies(self):
        m = burnman.Mineral(
            params={
                "equation_of_state": "bm3",
                "F_0": 1000.0,
                "T_0": 100.0,
                "P_0": 1.0e10,
                "V_0": 7.95626e-6,
                "K_0": 109.7e9,
                "Kprime_0": 4.66,
                "Kprime_inf": 3.00,
                "Kprime_prime_0": -0.043e-9,
                "Kdprime_0": -4.66 / 100.0e9,
                "molar_mass": 0.055845,
            }
        )

        eoses = ["bm3", "bm4", "vinet", "mt", "morse", "rkprime"]

        energies = []
        for eos in eoses:
            m.params["equation_of_state"] = eos
            burnman.Mineral.__init__(m)
            m.set_state(m.params["P_0"], m.params["T_0"])
            energies.append(m.molar_helmholtz)

        self.assertArraysAlmostEqual(energies, [m.params["F_0"]] * len(energies))

    def test_energy_derivatives(self):
        m = burnman.Mineral(
            params={
                "equation_of_state": "bm3",
                "V_0": 7.95626e-6,
                "K_0": 109.7e9,
                "Kprime_0": 4.66,
                "Kprime_inf": 3.00,
                "Kprime_prime_0": -0.043e-9,
                "Kdprime_0": -4.66 / 100.0e9,
                "molar_mass": 0.055845,
            }
        )

        eoses = [
            "bm3",
            "bm4",
            "vinet",
            "mt",
            "morse",
            "rkprime",
            "macaw",
            "spock",
            "murnaghan",
        ]

        calculated = []
        derivative = []
        for eos in eoses:
            m.params["equation_of_state"] = eos
            burnman.Mineral.__init__(m)

            P_0 = 10.0e9
            dP = 10000.0
            pressures = [P_0 - 0.5 * dP, P_0, P_0 + 0.5 * dP]
            temperatures = [0.0, 0.0, 0.0]

            E, G, H, A, V, KT = m.evaluate(
                ["molar_internal_energy", "gibbs", "H", "helmholtz", "V", "K_T"],
                pressures,
                temperatures,
            )

            calculated.append(P_0)
            derivative.append(-(E[2] - E[0]) / (V[2] - V[0]))
            calculated.append(V[1])
            derivative.append((G[2] - G[0]) / dP)
            calculated.append(-V[1] / KT[1])
            derivative.append((V[2] - V[0]) / dP)

        self.assertArraysAlmostEqual(calculated, derivative)

    def test_debye(self):
        T = 500.0
        debye_T = 200.0
        n = 3.0

        dT = 0.001

        Cv = debye.molar_heat_capacity_v(T, debye_T, n)
        S = debye.entropy(T, debye_T, n)
        Cv2 = (
            debye.thermal_energy(T + dT / 2.0, debye_T, n)
            - debye.thermal_energy(T - dT / 2.0, debye_T, n)
        ) / dT
        S2 = (
            -(
                debye.helmholtz_energy(T + dT / 2.0, debye_T, n)
                - debye.helmholtz_energy(T - dT / 2.0, debye_T, n)
            )
            / dT
        )
        self.assertAlmostEqual(Cv, Cv2)
        self.assertAlmostEqual(S, S2)

    def test_pressure_finding_SLB(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = alpha_quartz()
            P = 1.0e5
            T = 1000.0
            m.set_state(P, T)
            V = m.V
            m.set_state_with_volume(V, T)
        self.assertAlmostEqual(m.pressure / 1.0e5, P / 1.0e5, places=3)

    def test_pressure_finding_HP(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = minerals.HP_2011_ds62.mt()
            P = 1.0e9
            T = 1000.0
            m.set_state(P, T)
            V = m.V
            m.set_state(P * 1.01, T)
            m.set_state_with_volume(V, T)
        self.assertAlmostEqual(m.pressure / 1.0e5, P / 1.0e5, places=3)


if __name__ == "__main__":
    unittest.main()
