import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy import integrate

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

s = 0.25

x = np.linspace(2, 4., 1001)

for i in [1., 4., 16., 64.]:
    fn = lambda x: np.power(lognorm.pdf(4.-x, s), i)
    sum = integrate.quad(fn, x[0], x[-1])
    ax.plot(x, fn(x)/sum[0], label=f'{i:.0f}')

ax.set_ylim(0.,)
ax.set_xticklabels([])
ax.set_ylabel('Probability density')
ax.set_xlabel('Energy')
ax.legend()
fig.set_tight_layout(True)
fig.savefig('pdf_system_size.png')
plt.show()
