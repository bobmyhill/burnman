# Benchmarks for the chemical potential functions
import burnman
from burnman import equilibrate
import numpy as np
import matplotlib.pyplot as plt
from burnman.optimize.nonlinear_solvers import TerminationCode

"""
Initialise solid solutions
"""
ol = burnman.minerals.SLB_2011.mg_fe_olivine()
wad = burnman.minerals.SLB_2011.mg_fe_wadsleyite()
rw = burnman.minerals.SLB_2011.mg_fe_ringwoodite()

#  Solver has trouble converging if all three phases have the similar compositions
#  Jacobian problem?
print(
    "Warning: Solver has trouble converging if all three phases have similar compositions"
)
ol.set_composition([0.90, 0.10])
wad.set_composition([0.90, 0.10])
rw.set_composition([0.80, 0.20])

"""
Temperature of phase diagram
"""
T = 1673.0  # K

"""
Find invariant point
"""
x_Fe_initial = 0.5
composition = {
    "Fe": x_Fe_initial * 2.0,
    "Mg": 2.0 * (1.0 - x_Fe_initial),
    "Si": 1.0,
    "O": 4.0,
}
assemblage = burnman.Composite([ol, wad, rw], [1.0, 0.0, 0.0])
equality_constraints = [
    ("T", T),
    ("phase_fraction", (ol, 0.0)),
    ("phase_fraction", (rw, 0.0)),
]
free_compositional_vectors = [{"Mg": 1.0, "Fe": -1.0}]

sol, prm = equilibrate(
    composition,
    assemblage,
    equality_constraints,
    free_compositional_vectors,
    store_iterates=True,
    verbose=False,
)

print(np.array(sol.iterates.x))
print()
if not sol.success:
    print(sol.text)
    raise Exception(
        "Could not find solution for the univariant using provided starting guesses."
    )

P_univariant = sol.assemblage.pressure
phase_names = [sol.assemblage.phases[i].name for i in range(3)]
x_fe_mbr = [sol.assemblage.phases[i].molar_fractions[1] for i in range(3)]

print(f"Univariant pressure at {T:.0f} K: {P_univariant/1.e9:.3f} GPa")
print("Fe2SiO4 concentrations at the univariant:")
for i in range(3):
    print(f"{phase_names[i]}: {x_fe_mbr[i]:.2f}")

output_codes = []
output = []

print(
    "Warning: Calculations struggle to converge at edges of binary (as problem is ill-posed)."
)

fig = plt.figure()
ax = [fig.add_subplot(1, 1, 1)]

eps = 1.0e-7
for m1, m2, x_fe_m1 in [
    [ol, wad, np.linspace(eps, x_fe_mbr[0], 20)],
    [ol, rw, np.linspace(1.0 - eps, x_fe_mbr[0], 20)],
    [wad, rw, np.linspace(eps, x_fe_mbr[1], 20)],
]:

    assemblage = burnman.Composite([m1, m2], [1.0, 0.0])

    # Reset the compositions of the two phases to have compositions
    # close to those at the univariant point
    m1.set_composition([1.0 - x_fe_mbr[1], x_fe_mbr[1]])
    m2.set_composition([1.0 - x_fe_mbr[1], x_fe_mbr[1]])

    # Also set the pressure and temperature
    assemblage.set_state(14.0e9, T)

    # Here our equality constraints are temperature,
    # the phase fraction of the second phase,
    # and we loop over the composition of the first phase.
    equality_constraints = [
        ("T", T),
        (
            "phase_composition",
            (m1, [["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], x_fe_m1]),
        ),
        ("phase_fraction", (m2, 0.0)),
    ]

    sols, prm = equilibrate(
        composition,
        assemblage,
        equality_constraints,
        free_compositional_vectors,
        store_iterates=True,
        verbose=False,
    )
    print(sols[0].n_it)
    plt.plot(
        x_Fe_initial - sols[0].iterates.x[:, -1] / 2,
        sols[0].iterates.x[:, 0] / 1.0e9,
        lw=2,
        color="gray",
        zorder=1,
    )
    plt.scatter(
        x_Fe_initial - sols[0].iterates.x[:, -1] / 2,
        sols[0].iterates.x[:, 0] / 1.0e9,
        s=5,
        c=range(len(sols[0].iterates.x)),
        cmap="rainbow_r",
        zorder=2,
    )

    # Process the solutions
    codes = [[sol.code, sol.text] for sol in sols]
    out = np.array(
        [
            [
                sol.assemblage.pressure,
                sol.assemblage.phases[0].molar_fractions[1],
                sol.assemblage.phases[1].molar_fractions[1],
            ]
            for sol in sols
        ]
    )

    output.append(out)
    output_codes.append(codes)

for i in range(3):
    print(f"Failed solutions for segment {i}:")
    for j in range(len(output_codes[i])):
        if output_codes[i][j][0] != TerminationCode.SUCCESS:
            print(
                f"  Index {j}, code {output_codes[i][j][0]}, message: {output_codes[i][j][1]}"
            )
            #  print(f"  Iterates: {sols[j].iterates}")
            print("")


"""
Plot the phase diagram
"""


color = "purple"
# Plot the line connecting the three phases
ax[0].plot(
    [x_fe_mbr[0], x_fe_mbr[2]],
    [P_univariant / 1.0e9, P_univariant / 1.0e9],
    color=color,
)

for i in range(3):
    if i == 0:
        ax[0].plot(
            output[i][:, 1], output[i][:, 0] / 1.0e9, color=color, label=f"{T} K"
        )
    else:
        ax[0].plot(output[i][:, 1], output[i][:, 0] / 1.0e9, color=color)

    ax[0].plot(output[i][:, 2], output[i][:, 0] / 1.0e9, color=color)
    ax[0].fill_betweenx(
        output[i][:, 0] / 1.0e9,
        output[i][:, 1],
        output[i][:, 2],
        color=color,
        alpha=0.2,
    )

ax[0].text(0.1, 6.0, "olivine", horizontalalignment="left")
ax[0].text(
    0.015,
    14.2,
    "wadsleyite",
    horizontalalignment="left",
    bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.2"),
)
ax[0].text(0.9, 15.0, "ringwoodite", horizontalalignment="right")

ax[0].set_xlim(0.0, 1.0)
ax[0].set_ylim(0.0, 20.0)
ax[0].set_xlabel("p(Fe$_2$SiO$_4$)")
ax[0].set_ylabel("Pressure (GPa)")
ax[0].legend()
plt.show()
