import unittest
import os, sys

sys.path.insert(1, os.path.abspath('..'))
import warnings

import burnman
from burnman import minerals

from util import BurnManTest


class mypericlase(burnman.Mineral):
    """
    Stixrude & Lithgow-Bertelloni 2005 and references therein 
    """

    def __init__(self):
        self.params = {
            'equation_of_state': 'slb3',
            'V_0': 11.24e-6,
            'K_0': 161.0e9,
            'Kprime_0': 3.8,
            'G_0': 131.0e9,
            'Gprime_0': 2.1,
            'molar_mass': .0403,
            'n': 2,
            'Debye_0': 773.,
            'grueneisen_0': 1.5,
            'q_0': 1.5,
            'eta_s_0': 2.8}
        burnman.Mineral.__init__(self)

class eos(BurnManTest):
    def test_reference_values(self):
        rock = mypericlase()
        pressure = 0.
        temperature = 300.
        eoses = [burnman.eos.SLB2(), burnman.eos.SLB3(), burnman.eos.BM2(), burnman.eos.BM3()]

        for i in eoses:
            rock.method=i
            rock.set_state(pressure, temperature)
            Volume_test = i.volume(rock)
            self.assertFloatEqual(Volume_test, rock.params['V_0'])
            Kt_test = i.isothermal_bulk_modulus(rock)
            self.assertFloatEqual(Kt_test, rock.params['K_0'])
            # Compare K_T and K_S at 0 K:
            rock.set_state(pressure, 0.)
            Kt_test = i.isothermal_bulk_modulus(rock)
            K_test = i.adiabatic_bulk_modulus(rock)
            rock.set_state(pressure, temperature)
            self.assertFloatEqual(K_test, Kt_test)
            G_test = i.shear_modulus(rock)
            self.assertFloatEqual(G_test, rock.params['G_0'])
            Density_test = i.density(rock)
            self.assertFloatEqual(Density_test, rock.params['molar_mass'] / rock.params['V_0'])
            alpha_test = i.thermal_expansivity(rock)
            Cp_test = i.heat_capacity_p(rock)
            Cv_test = i.heat_capacity_v(rock)
            Grun_test = i.grueneisen_parameter(rock)

        eoses_thermal = [burnman.eos.SLB2(), burnman.eos.SLB3()]
        for i in eoses_thermal:
            rock.method=i
            rock.set_state(pressure, temperature)
            Cp_test = i.heat_capacity_p(rock)
            self.assertFloatEqual(Cp_test, 37.076768469502042)
            Cv_test = i.heat_capacity_v(rock)
            self.assertFloatEqual(Cv_test, 36.577717628901553)
            alpha_test = i.thermal_expansivity(rock)
            self.assertFloatEqual(alpha_test, 3.031905596878513e-05)
            Grun_test = i.grueneisen_parameter(rock)
            self.assertFloatEqual(Grun_test, rock.params['grueneisen_0'])


    



class test_eos_validation(BurnManTest):
    def test_no_shear_error(self):
        #The validation should place nans in for the shear parameters
        #If any exceptions or warnings are raised, fail.
        class mymineralwithoutshear(burnman.Mineral):
            def __init__(self):
                self.params = {
                    'equation_of_state': 'slb3',
                    'V_0': 11.24e-6,
                    'K_0': 161.0e9,
                    'Kprime_0': 3.8,
                    'molar_mass': .0403,
                    'n': 2,
                    'Debye_0': 773.,
                    'grueneisen_0': 1.5,
                    'q_0': 1.5,
                    'eta_s_0': 2.8}
                burnman.Mineral.__init__(self)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            #Trigger warning
            shearless = mymineralwithoutshear()
            if len(w) != 0:
                self.fail("Caught unexpected warning: "+str(w[-1]))
            try:
                x = shearless.params['G_0']
                y = shearless.params['Gprime_0']
                z = shearless.params['F_0']
                z = shearless.params['eta_s_0']
            except KeyError:
                self.fail('Parameter padding failed in validation')
                pass

    def test_dumb_parameter_values(self):

        class mymineralwithnegativekprime(burnman.Mineral):
            def __init__(self):
                self.params = {
                    'equation_of_state': 'slb3',
                    'V_0': 11.24e-6,
                    'K_0': 161.0e9,
                    'Kprime_0': -4.,
                    'molar_mass': .0403,
                    'n': 2,
                    'Debye_0': 773.,
                    'grueneisen_0': 1.5,
                    'q_0': 1.5,
                    'eta_s_0': 2.8}
                burnman.Mineral.__init__(self)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            #Trigger warning
            negative_Kprime = mymineralwithnegativekprime()
            if len(w) == 0:
                print negative_Kprime.params
                self.fail("Did not catch expected warning for negative K prime")
     
        class mymineralwithkingigapascals(burnman.Mineral):
            def __init__(self):
                self.params = {
                    'equation_of_state': 'slb3',
                    'V_0': 11.24e-6,
                    'K_0': 161.0,
                    'Kprime_0': 3.8,
                    'molar_mass': .0403,
                    'n': 3.14159,
                    'Debye_0': 773.,
                    'grueneisen_0': 1.5,
                    'q_0': 1.5,
                    'eta_s_0': 2.8}
                burnman.Mineral.__init__(self)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            #Trigger warning
            low_K = mymineralwithkingigapascals()
            if len(w) == 0:
                self.fail("Did not catch expected warning K in Gpa")


if __name__ == '__main__':
    unittest.main()
