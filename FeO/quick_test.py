import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals





per = minerals.HP_2011_ds62.per()

P1 = 1.e5
T1 = 600.
T2 = 1200.

per.set_state(P1, T1)
print per.K_T

P2 = per.method.pressure(T2, per.V, per.params)
per.set_state(P2, T2)
print per.K_T


print per.method.
