import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.image as mpimg

from FMSH_melt_model import ax_melt


Tm = 2550.0
Delta_Sf = 100.3 - 4.6

data = np.genfromtxt(
    "data/13GPa_fo-H2O.dat", dtype=[float, float, float, (np.unicode_, 16)]
)
phases = list(set([d[3] for d in data]))

experiments = {
    ph: np.array([[d[0], d[1], d[2]] for d in data if d[3] == ph]).T for ph in phases
}

# hyfo_img = mpimg.imread('data/hyfo_melting_Myhill_et_al_2017.png')
# plt.imshow(hyfo_img, extent=[0.0, 1.0, 1073.15, 2873.15], aspect='auto')

for phase, expts in experiments.items():
    plt.scatter(
        expts[2] / (expts[1] + expts[2]), expts[0] + 273.15, label=phase
    )  # on a 1-cation basis

temperatures = np.linspace(100.0, Tm, 101)
a_H2O = np.empty_like(temperatures)
X_H2O = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    a_H2O[i], X_H2O[i], _ = ax_melt(T, Tm, Delta_Sf)

plt.plot(X_H2O, temperatures, label="compositions")
plt.plot(a_H2O, temperatures, label="activities")
plt.legend()
plt.show()
