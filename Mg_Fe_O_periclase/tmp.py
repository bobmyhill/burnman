
from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
from burnman.processchemistry import dictionarize_formula, formula_mass

X = 0.1
plt.plot([0.0], [1.0], label='Mg$_{{{0:.1f}}}$Fe$_{{{1:.1f}}}$O'.format(1. - X, X))
plt.legend()
plt.show()
