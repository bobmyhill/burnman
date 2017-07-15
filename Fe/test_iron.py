import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from scipy.optimize import fsolve
import matplotlib.image as mpimg

from fcc_iron import fcc_iron
from hcp_iron import hcp_iron
from liq_iron_modified_AA1994 import liq_iron


bcc = minerals.HP_2011_ds62.iron()
fcc = fcc_iron()
hcp = hcp_iron()
liq = liq_iron()



hcp.set_state(1.e5, 1809.)
print hcp.gibbs, hcp.V
hcp.set_state(1.e9, 2000.)
print hcp.gibbs, hcp.V
hcp.set_state(1.e9, 3000.)
print hcp.gibbs, hcp.V
hcp.set_state(1.e10, 3000.)
print hcp.gibbs, hcp.V

liq.set_state(1.e5, 1809.)
print liq.gibbs, liq.V
liq.set_state(1.e9, 2000.)
print liq.gibbs, liq.V
liq.set_state(1.e9, 3000.)
print liq.gibbs, liq.V
liq.set_state(1.e10, 3000.)
print liq.gibbs, liq.V
