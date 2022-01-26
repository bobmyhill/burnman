import burnman
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

ol = burnman.minerals.SLB_2011.mg_fe_olivine()
ol.set_state(1.e5, 300.)

ol.endmembers[0][0].params['F_0'] -= ol.endmembers[0][0].gibbs + 1000.
ol.endmembers[1][0].params['F_0'] -= ol.endmembers[1][0].gibbs + 2000.
ol.set_state(1.e5, 301.)

n_fos = np.linspace(1.e-5, 1.2, 101)
n_fas = np.linspace(1.e-5, 1.2, 101)

G = np.empty((101, 101))



for i, n_fo in enumerate(n_fos):
    for j, n_fa in enumerate(n_fas):
        n = n_fo + n_fa
        x = n_fo / n
        ol.set_composition([x, 1.-x])
        G[i, j] = ol.gibbs * n
        if n > 1.05:
            G[i, j] = np.nan

X, Y = np.meshgrid(n_fos, n_fas)
fig = go.Figure(data=[go.Surface(z=G, x=X, y=Y,
                                 contours={"x": {"show": True, "start": 0.2, "end": 1., "size": 0.2, "color":"white"},
                                           "y": {"show": True, "start": 0.2, "end": 1., "size": 0.2, "color":"white"},
                                           "z": {"show": True, "start": 0., "end": 10000., "size": 1000., "color":"white"}})])

line_markers = [dict(color='#1212B5', width=10),
                dict(color='#3737B5', width=10),
                dict(color='#6565B5', width=10)]

line_marker_2 = dict(color='#B53737', width=10)

for i, x in enumerate([0.1, 0.3, 0.82]):
    ol.set_composition([x, 1.-x])
    fig.add_scatter3d(y=[0., x], x=[0., 1.-x], z=[0., ol.gibbs],
                      mode='lines', line=line_markers[i], name='') 

    fig.add_scatter3d(y=[1., 0.], x=[0., 1.], z=ol.partial_gibbs,
                      mode='lines', line=line_markers[i], name=f'x(fo) = {x}') 

xs = np.linspace(0., 1., 1001)
Gs = np.empty_like(xs)


for i, x in enumerate(xs):
    ol.set_composition([x, 1.-x])
    Gs[i] = ol.gibbs


line_marker = dict(color='#101010', width=10)
fig.add_scatter3d(y=xs, x=1.-xs, z=Gs+1.e-5, mode='lines', line=line_marker, name='') 
fig.add_scatter3d(x=[0., 0.], y=[1., 1.], z=[0., -6000.],
                  mode='lines', line=line_marker, name='')
fig.add_scatter3d(x=[1., 1.], y=[0., 0.], z=[0., -6000.],
                  mode='lines', line=line_marker, name='')

fig.update_layout(scene = dict(
                    xaxis_title='N(fo)',
                    yaxis_title='N(fa)',
                    zaxis_title='Gibbs energy (J)'),
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