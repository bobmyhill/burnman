import numpy as np
import matplotlib.pyplot as plt

from burnman import Composite, equilibrate
import mineral_models

rs = mineral_models.rocksalt()
wz = mineral_models.wurtzite()
rs.set_composition([0.999, 0.001])
wz.set_composition([0.001, 0.999])
rs.set_state(1.0e5, 300.0)
wz.set_state(1.0e5, 300.0)

composition = {"Mg": 0.2, "Zn": 0.8, "O": 1.0}
free_compositional_vectors = [{"Mg": 1.0, "Zn": -1.0}]
assemblage = Composite([rs, wz], [0.5, 0.5])

for pressure in [1.0e5, 3.0e9]:
    temperatures = np.linspace(900.0 + 273.15, 1700.0 + 273.15, 41)
    equality_constraints = [
        ("P", pressure),
        ("T", temperatures),
        ("phase_fraction", (rs, 0.0)),
    ]

    sols, prm = equilibrate(
        composition, assemblage, equality_constraints, free_compositional_vectors
    )

    # Interrogate the stable assemblages for phase compositions.
    x1s = np.array(
        [sol.assemblage.phases[0].molar_fractions[1] for sol in sols if sol.code == 0]
    )
    x2s = np.array(
        [sol.assemblage.phases[1].molar_fractions[1] for sol in sols if sol.code == 0]
    )
    Ts = np.array([sol.assemblage.temperature for sol in sols if sol.code == 0])

    plt.plot(x1s, Ts - 273.15, label=f"rs ({pressure/1.e9:.1f} GPa)")
    plt.plot(x2s, Ts - 273.15, label=f"wz ({pressure/1.e9:.1f} GPa)")

plt.text(0.5, 400.0, "miscibility gap", horizontalalignment="center")
plt.xlabel("Molar proportion of ZnO")
plt.ylabel("Temperature (C)")
plt.xlim(0.0, 1.0)
plt.ylim(900.0, 1700.0)
plt.legend()
plt.title("Figure 1")
plt.show()


for tC in [800.0, 1200.0]:
    pressures = np.linspace(0.0, 4.0e9, 41)
    equality_constraints = [
        ("P", pressures),
        ("T", tC + 273.15),
        ("phase_fraction", (rs, 0.0)),
    ]

    sols, prm = equilibrate(
        composition, assemblage, equality_constraints, free_compositional_vectors
    )

    # Interrogate the stable assemblages for phase compositions.
    x1s = np.array(
        [sol.assemblage.phases[0].molar_fractions[1] for sol in sols if sol.code == 0]
    )
    x2s = np.array(
        [sol.assemblage.phases[1].molar_fractions[1] for sol in sols if sol.code == 0]
    )
    Ps = np.array([sol.assemblage.pressure for sol in sols if sol.code == 0])

    plt.plot(x1s, Ps / 1.0e9, label=f"rs ({tC} C)")
    plt.plot(x2s, Ps / 1.0e9, label=f"wz ({tC} C)")

plt.text(0.5, 400.0, "miscibility gap", horizontalalignment="center")
plt.xlabel("Molar proportion of ZnO")
plt.ylabel("Pressure (GPa)")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 4.0)
plt.legend()
plt.title("Figure 3")
plt.show()
