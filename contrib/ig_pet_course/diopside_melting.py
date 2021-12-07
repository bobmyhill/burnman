# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
Diopside melting
"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.interpolate import interp1d

import plotly.graph_objects as go


import burnman
from burnman import minerals


if __name__ == "__main__":

    diopside = burnman.minerals.HP_2011_ds62.sill()
    diopsideL = burnman.minerals.HP_2011_ds62.andalusite()
    diopsideL.property_modifiers.append(['linear', {'delta_E': 20*750,
                                                    'delta_S': 20.,
                                                    'delta_V': 0.}])

    composition = {'Al': 2., 'Si': 1., 'O': 5.}
    assemblage = burnman.Composite([diopside, diopsideL])
    pressures = np.linspace(1.e5, 1.e9, 101)
    equality_constraints = [['phase_fraction', (diopsideL, 0.)],
                            ['P', pressures]]

    sols = burnman.equilibrate(composition, assemblage, equality_constraints,
                               store_assemblage=True)

    T_fusion = np.array([sol.assemblage.temperature for sol in sols[0]])
    S_fusion = np.array([sol.assemblage.molar_entropy for sol in sols[0]])
    S_fusion2 = diopsideL.evaluate(['S'], pressures, T_fusion)[0]

    S_grid = np.linspace(200., 300., 101)
    T_grid = []
    for i, P in enumerate(pressures):
        temperatures = np.linspace(500., T_fusion[i], 101)
        entropies = diopside.evaluate(
            ['S'], P + 0.*temperatures, temperatures)[0]

        temperatures2 = np.linspace(T_fusion[i], 1000., 101)
        entropies2 = diopsideL.evaluate(
            ['S'], P + 0.*temperatures2, temperatures2)[0]

        S = list(entropies)
        S.extend(list(entropies2))
        S = np.array(S)

        T = list(temperatures)
        T.extend(list(temperatures2))
        T = np.array(T)
        T_interp = interp1d(S, T)

        T_grid.append(T_interp(S_grid))

    pp, SS = np.meshgrid(pressures, S_grid)
    print(pp)
    TT = np.array(T_grid)

    fig = go.Figure(data=[go.Surface(x=pp/1.e9, y=SS, z=TT,
                                     contours={"y": {"show": True,
                                                     "start": 100,
                                                     "end": 1000,
                                                     "size": 10,
                                                     "color": "white"},
                                               "z": {"show": True,
                                                     "start": 100,
                                                     "end": 1000,
                                                     "size": 10,
                                                     "color": "blue"}})])

    marker = dict(size=1, color=pressures/1.e9, colorscale='Reds')
    fig.add_scatter3d(x=pressures/1.e9, y=S_fusion, z=T_fusion,
                      mode='markers', marker=marker)
    fig.add_scatter3d(x=pressures/1.e9, y=S_fusion2, z=T_fusion,
                      mode='markers', marker=marker)

    fig.update_layout(title="Diopside melting",
                      xaxis_title="Pressure (GPa)",
                      yaxis_title="Temperature (K)",
                      legend_title="Entropy (J/K/mol)",
                      font=dict(family="Courier New, monospace",
                                size=18,
                                color="RebeccaPurple"
                                )
                      )

    fig.show()
