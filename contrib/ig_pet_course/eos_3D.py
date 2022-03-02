import burnman
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

fo = burnman.minerals.SLB_2011.forsterite()

temperatures = np.linspace(300., 2000., 101)
pressures = np.linspace(0., 10.e9, 101)

G = np.empty((101, 101))



for i, T in enumerate(temperatures):
    G[i] = fo.evaluate(['gibbs'], pressures, T + pressures*0.)[0] / 1.e3

X, Y = np.meshgrid(pressures/1.e9, temperatures)
fig = go.Figure(data=[go.Surface(z=G, x=X, y=Y)])

fig.update_layout(scene=dict(xaxis_title='P (GPa)',
                             yaxis_title='Temperature (K)',
                             zaxis_title='Gibbs energy (kJ/mol)'),
                  width=700,
                  margin=dict(r=20, b=10, l=10, t=10))

fig.show()

exit()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.plot([0., 1.], [0., 1.], [0., 2.])



surf = ax.plot_surface(X, Y, G, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')



# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()