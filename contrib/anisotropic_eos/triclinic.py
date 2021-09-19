# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for
# the Earth and Planetary Sciences
# Copyright (C) 2012 - 2021 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""

triclinic
---------

"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import burnman_path  # adds the local burnman directory to the path
import burnman

from anisotropicmineral import AnisotropicMineral

from tools import print_table_for_mineral_constants
from tools import plot_projected_elastic_properties
from burnman import anisotropy
assert burnman_path  # silence pyflakes warning
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def draw_boxes(ax, v0, v1, colors):

    ps = []
    ccs = []
    for v in [v0, v1]:
        p = np.array([[0., 0., 0.],
                      v[0],
                      v[1],
                      v[2],
                      v[0]+v[1],
                      v[1]+v[2],
                      v[0]+v[2],
                      v[0]+v[1]+v[2]])
        cs = np.array([[p[0], p[1]],
                       [p[0], p[2]],
                       [p[0], p[3]],
                       [p[1], p[4]],
                       [p[1], p[6]],
                       [p[2], p[4]],
                       [p[2], p[5]],
                       [p[3], p[5]],
                       [p[3], p[6]],
                       [p[4], p[7]],
                       [p[5], p[7]],
                       [p[6], p[7]]])
        ps.append(p)
        ccs.append(cs)

    for i, cs in enumerate(ccs):
        for (p0, p1) in cs:
            ax.plot([p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]], color=colors[i])

    for i in range(1, 8):
        p0 = ps[0][i]
        p1 = ps[1][i]

        a = Arrow3D([p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]], mutation_scale=10,
                    lw=3, arrowstyle="-|>", color="#cccccc")
        ax.add_artist(a)

talc = burnman.minerals.HP_2011_ds62.ta()

f_order = 1
Pth_order = 1
constants = np.zeros((6, 6, f_order+1, Pth_order+1))

c = np.array([5.3, 9.2, 9.4])*np.cbrt(burnman.constants.Avogadro/1.e30/2.)
c *= 1.0675650257500944 # 1.000744960960125
talc_cell_parameters = np.array([c[0], c[1], c[2], 60, 70, 80])

talc_stiffness = [219.83e9,  59.66e9,  -4.82e9,  -0.82e9, -33.87e9, -1.04e9,
                  216.38e9, -3.67e9,   1.79e9, -16.51e9,  -0.62e9,
                  48.89e9,    4.12e9, -15.52e9,  -3.59e9,
                  26.54e9,    -3.6e9,  -6.41e9,
                  22.85e9,   -1.67e9,
                  78.29e9]
rho = 2.75e3
talc2 = anisotropy.TriclinicMaterial(rho, talc_stiffness)

S_N = talc2.isentropic_compliance_tensor
beta_N = np.sum(S_N[:3,:3])

constants[:,:,1,0] = S_N/beta_N
m = AnisotropicMineral(talc, talc_cell_parameters, constants)

m.set_state(1.e5, 298.15)
v0u = m.unrotated_cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)

m.set_state(1.e10, 298.15)
v1u = m.unrotated_cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)

m.set_state(3.e10, 298.15)
v2u = m.unrotated_cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)


m.set_state(1.e11, 300.)
v3u = m.unrotated_cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)



m.set_state(1.e5, 298.15)
v0 = m.cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)

m.set_state(1.e10, 298.15)
v1 = m.cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)

m.set_state(3.e10, 298.15)
v2 = m.cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)


m.set_state(1.e11, 300.)
v3 = m.cell_vectors/np.cbrt(burnman.constants.Avogadro/1.e30/2.)


print(v0)
print(v1)

fig = plt.figure()
ax = [fig.add_subplot(1, 1, 1, projection='3d')]

draw_boxes(ax[0], v0u, v1u, colors=['b', 'r'])
draw_boxes(ax[0], v1u, v1, colors=['r', 'g'])

ax[0].set_xlim(-5., 15.)
ax[0].set_ylim(-5., 15.)
ax[0].set_zlim(-5., 15.)
plt.show()
exit()
