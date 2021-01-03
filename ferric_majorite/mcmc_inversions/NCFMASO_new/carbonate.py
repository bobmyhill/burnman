import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import fsolve


# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))

import burnman


from burnman.minerals.HGP_2018_ds633 import mag, sid, diam

mag2 = burnman.CombinedMineral([mag(), diam()], [1., -1.])
sid2 = burnman.CombinedMineral([sid(), diam()], [1., -1.])

MgFeO3 = burnman.SolidSolution(name = 'magnesite-siderite - diamond',
                               solution_type = 'symmetric',
                               endmembers = [[mag2, '[Mg]O3'],
                                             [sid2, '[Fe]O3']],
                               energy_interaction = [[4.4e3]]) # W_E from Chai + Navrotsky, 1996
