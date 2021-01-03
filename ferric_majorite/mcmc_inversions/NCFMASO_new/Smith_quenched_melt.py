import numpy as np

elements = ['Fe', 'Ni', 'C', 'S']
phases = ['cohenite', 'FeNi alloy', 'sulfide']
#FeNiCS_at_fraction = np.array([[0.75, 0.0, 0.25, 0.0],
#                               [0.58, 0.37, 0.05, 0.0],
#                               [0.49, 0.0, 0.0, 0.51]])


FeNiCS_wt_fraction = np.array([[0.90, 0.03, 0.07, 0.0],
                               [0.60, 0.39, 0.01, 0.0],
                               [0.63, 0.0, 0.0, 0.37]])

volume_fractions = np.array([0.43, 0.37, 0.20]) # m^3/m^3
densities = np.array([7.4, 7.8, 4.6]) # kg/cm^3

wt_fractions = densities*volume_fractions
wt_fractions /= np.sum(wt_fractions)

FeNiCS_bulk_wt_fraction = FeNiCS_wt_fraction.T.dot(wt_fractions)

FeNiCS_molar_mass = np.array([55.845, 58.6934, 12.0107, 32.065])


FeNiCS_bulk_at_fraction = FeNiCS_bulk_wt_fraction/FeNiCS_molar_mass
FeNiCS_bulk_at_fraction /= np.sum(FeNiCS_bulk_at_fraction)


formula = {}
for i in range(4):
    formula[elements[i]] = FeNiCS_bulk_at_fraction[i]

print(formula)
