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

from burnman import anisotropy
assert burnman_path  # silence pyflakes warning
from matplotlib import pyplot as plt


talc2 = anisotropy.AnisotropicMaterial(rho, talc_stiffness)
