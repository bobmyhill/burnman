import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def G_mix(x, w, lmbda):
    y = x/(x + lmbda*(1.-x))
    return w[0]*y*y*(1.-y) + w[1]*y*(1.-y)*(1.-y)




w=[1, 1]
lmbda=10.


xs=np.linspace(0., 1., 101)
ys=np.empty_like(xs)
for i, x in enumerate(xs):
    ys[i] = G_mix(x, w, lmbda)

plt.plot(xs, ys, linewidth=1)
plt.show()
