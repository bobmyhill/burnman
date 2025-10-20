import numpy as np
from scipy.optimize import linprog
from burnman.minerals import SLB_2024
from burnman import Composite
import matplotlib.pyplot as plt

# This script demonstrates Gibbs energy minimization
# for a multi-component system using BurnMan and the
# linear programming capabilities of SciPy.
# (either the Dual Simplex or Interior Point method).

# The basic idea is to split each solution phase into
# a set of endmembers, and then minimize the total Gibbs
# energy of the system subject to compositional constraints
# and positive endmember amounts.

# In Python, this technique is slow. This example is designed
# only for demonstration purposes and does not aim to be
# efficient. It illustrates the method used by Perple_X,
# which is written in Fortran, and is much faster both as
# a result of the compiled language and because it uses a
# number of clever techniques to speed up the calculations.

n_it = [0]


def minimize_gibbs(composite, bulk, x0=None):
    """
    Minimize total Gibbs energy of a set of BurnMan endmembers
    subject to compositional constraints and positive phase amounts.
    """
    print(n_it[0])
    n_it[0] += 1
    A_eq = composite.stoichiometric_array.T
    b_eq = np.array([bulk[el] if el in bulk else 0 for el in composite.elements])
    gibbs = np.array([e.gibbs for e in composite.phases])
    sol = linprog(
        c=gibbs, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), x0=x0, method="highs"
    )
    n_moles = sum(sol.x)
    composite.set_fractions(sol.x / n_moles)
    composite.n_moles = n_moles
    return sol


# Define your BurnMan endmembers at the chosen P–T
n_div = 101
n_div2 = 101
ol = SLB_2024.olivine()
mbrs = []
phases = []
for p in np.linspace(0.0, 0.5, n_div):
    ol.set_composition([1.0 - p, p])  # Mg2SiO4 - Fe2SiO4
    mbrs.append(ol.copy())
    phases.append(ol)

wad = SLB_2024.wadsleyite()
for p in np.linspace(0.0, 0.5, n_div):
    wad.set_composition([1.0 - p, p])  # Mg2SiO4 - Fe2SiO4
    mbrs.append(wad.copy())
    phases.append(wad)

rw = SLB_2024.ringwoodite()
for p in np.linspace(0.0, 0.5, n_div):
    rw.set_composition([1.0 - p, p])  # Mg2SiO4 - Fe2SiO4
    mbrs.append(rw.copy())
    phases.append(rw)

bdg = SLB_2024.bridgmanite()
for p in np.linspace(0.0, 0.5, n_div):
    bdg.set_composition([1.0 - p, p, 0.0, 0.0, 0.0, 0.0, 0.0])  # MgSiO3 - FeSiO3
    mbrs.append(bdg.copy())
    phases.append(bdg)

fper = SLB_2024.ferropericlase()
for i, p in enumerate(np.linspace(0.0, 0.5, n_div2)):
    for q in np.linspace(0.0, 0.5 - p, n_div2 - i):
        fper.set_composition([1.0 - p - q, p, q, 0.0, 0.0])  # MgO - FeO - FeLSO
        mbrs.append(fper.copy())
        phases.append(fper)


ppv = SLB_2024.post_perovskite()
for p in np.linspace(0.0, 0.5, n_div):
    ppv.set_composition([1.0 - p, p, 0.0, 0.0, 0.0])  # MgSiO3 - FeSiO3
    mbrs.append(ppv.copy())
    phases.append(ppv)

composite = Composite(mbrs)

composite_elements = composite.elements
stoichiometric_array = []
for phase in composite.phases:
    stoichiometric_array.append([phase.formula.get(el, 0) for el in composite_elements])
composite.stoichiometric_array = np.array(stoichiometric_array)


T = 1500.0  # Temperature in K

# Define bulk composition (in moles of elements)
bulk = {"Mg": 1.8, "Fe": 0.2, "Si": 1.0, "O": 4.0}

# Run minimization
pressures = np.linspace(0e9, 150e9, 151)
f = []
x0 = None
for i, P in enumerate(pressures):
    composite.set_state(P, T)
    sol = minimize_gibbs(composite, bulk, x0=x0)
    x0 = sol.x
    f.append(composite.molar_fractions.copy())

f = np.array(f)

fig = plt.figure()
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

print(composite.elements)
print(pressures)
phase_fractions = []
for phase in [ol, wad, rw, bdg, fper, ppv]:
    indices = [j for j, p in enumerate(phases) if phase == p]
    compositions = np.array(
        [composite.stoichiometric_array[j] for j, p in enumerate(phases) if phase == p]
    )
    molar_fractions = np.array(
        [
            (
                composite.phases[j].molar_fractions
                if hasattr(composite.phases[j], "molar_fractions")
                else [1.0]
            )
            for j, p in enumerate(phases)
            if phase == p
        ]
    )
    phase_fractions = np.sum(f[:, indices], axis=1)
    phase_compositions = np.dot(f[:, indices], compositions).T / phase_fractions
    molar_fractions = np.dot(f[:, indices], molar_fractions)

    print(phase.name)
    print(phase_compositions)
    print(molar_fractions)

    Mg_idx = composite.elements.index("Mg")
    Fe_idx = composite.elements.index("Fe")
    Mg_num = phase_compositions[Mg_idx] / (
        phase_compositions[Mg_idx] + phase_compositions[Fe_idx]
    )

    print()
    ax[0].plot(pressures, phase_fractions, label=phase.name)
    ax[1].plot(pressures, Mg_num, label=phase.name)
    ax[2].plot(pressures, molar_fractions[:, 0], label=phase.name)
    ax[3].plot(pressures, molar_fractions[:, 1], label=phase.name)

ax[0].set_xlabel("Pressure (Pa)")
ax[0].set_ylabel("Phase Fraction")
ax[0].legend()

ax[1].set_xlabel("Pressure (Pa)")
ax[1].set_ylabel("Mg Number")
ax[1].legend()

ax[2].set_xlabel("Pressure (Pa)")
ax[2].set_ylabel("Proportion of first endmember")
ax[2].legend()

ax[3].set_xlabel("Pressure (Pa)")
ax[3].set_ylabel("Proportion of second endmember")
ax[3].legend()

fig.set_layout_engine("tight")
fig.savefig("equilibrium_phase_fractions.pdf")
plt.show()
