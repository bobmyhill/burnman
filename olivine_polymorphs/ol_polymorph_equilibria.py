# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman.minerals import SLB_2011
from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

from equilibrium_functions import *
from models import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

from scipy import optimize


plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

'''
Initialise solid solutions
'''

#ol=SLB_2011.mg_fe_olivine()
#wd=SLB_2011.mg_fe_wadsleyite()
#rw=SLB_2011.mg_fe_ringwoodite()

ol=mg_fe_olivine()
wd=mg_fe_wadsleyite()
rw=mg_fe_ringwoodite()
#wd=mg_fe_wadsleyite_asymmetric()
#rw=mg_fe_ringwoodite_asymmetric()

fig1 = mpimg.imread('figures/ol_polymorphs_1400C.png')
plt.imshow(fig1, extent=[0,1,6.0,20.], aspect='auto')


# MG-RINGWOODITE is a problem. High quality data suggests that the metastable transition of forsterite to Mg-ringwoodite should be at higher pressure than predicted by the fo-wad and wad-ringwoodite curves. Dan Frost gets around this by assuming a much smaller volume for ringwoodite, but this is inconsistent with PVT observations and thermal expansivity measurements at ambient pressure.

# Possible solutions: 
# Mg-Si disordering in rw at HT? Hazen et al., 1993
# Mg-Fe ordering stabilising rw at ~ 50%  
# Fe-Si disorder at the Fe rich end
# Some aspects of all three?


# Data from Dan Frost's experiments
ol_wad_data = []
ol_rw_data = []
wad_rw_data = []
for line in open('ol_polymorph_equilibria.dat'):
    content=line.strip().split()
    if content[0] != '%':
        # ol-wad
        if content[7] != '-' and content[9] != '-':
            ol_wad_data.append([float(content[2]), 0.5, float(content[3]), float(content[7]), float(content[8]), float(content[9]), float(content[10])])
        # ol-rw
        if content[7] != '-' and content[11] != '-':
            ol_rw_data.append([float(content[2]), 0.5, float(content[3]), float(content[7]), float(content[8]), float(content[11]), float(content[12])])
        # wd-rw
        if content[9] != '-' and content[11] != '-':
            wad_rw_data.append([float(content[2]), 0.5, float(content[3]), float(content[9]), float(content[10]), float(content[11]), float(content[12])])

ol_wad_data = zip(*ol_wad_data)
ol_rw_data = zip(*ol_rw_data)
wad_rw_data = zip(*wad_rw_data)



# Temperature of phase diagram
T=1673. # K

# Find invariant point
invariant=optimize.fsolve(eqm_P_xMgABC(ol, wd, rw), [15.e9, 0.2, 0.3, 0.4], args=(T))
print invariant

# Initialise arrays
XMgA_ol_wad=np.linspace(invariant[1], 0.9999, 15)
XMgA_ol_rw=np.linspace(0.0001, invariant[1], 15)
XMgA_wad_rw=np.linspace(invariant[2], 0.9999, 15)

P_ol_wad=np.empty_like(XMgA_ol_wad)
XMgB_ol_wad=np.empty_like(XMgA_ol_wad)

P_ol_rw=np.empty_like(XMgA_ol_wad)
XMgB_ol_rw=np.empty_like(XMgA_ol_wad)

P_wad_rw=np.empty_like(XMgA_ol_wad)
XMgB_wad_rw=np.empty_like(XMgA_ol_wad)


# Find transition pressures
for idx, XMgA in enumerate(XMgA_ol_wad):
    XMgB_guess=1.0-((1.0-XMgA_ol_wad[idx])*0.8)
    P_ol_wad[idx], XMgB_ol_wad[idx] = optimize.fsolve(eqm_P_xMgB(ol, wd), [5.e9, XMgB_guess], args=(T, XMgA_ol_wad[idx]))
    XMgB_guess=1.0-((1.0-XMgA_ol_rw[idx])*0.8)
    P_ol_rw[idx], XMgB_ol_rw[idx] = optimize.fsolve(eqm_P_xMgB(ol, rw), [5.e9, XMgB_guess], args=(T, XMgA_ol_rw[idx]))
    XMgB_guess=1.0-((1.0-XMgA_wad_rw[idx])*0.8)
    P_wad_rw[idx], XMgB_wad_rw[idx] = optimize.fsolve(eqm_P_xMgB(wd, rw), [5.e9, XMgB_guess], args=(T, XMgA_wad_rw[idx]))


# Plot data
plt.plot( 1.0-np.array([invariant[1], invariant[2], invariant[3]]), np.array([invariant[0], invariant[0], invariant[0]])/1.e9, color='black', linewidth=3, label='invariant')

plt.plot( 1.0-XMgA_ol_wad, P_ol_wad/1.e9, 'r-', linewidth=3, label='wad-out (ol, wad)')
plt.plot( 1.0-XMgB_ol_wad, P_ol_wad/1.e9, 'g-', linewidth=3, label='ol-out (ol, wad)')

plt.plot( 1.0-XMgA_ol_rw, P_ol_rw/1.e9, 'r-',  linewidth=3, label='rw-out (ol, rw)')
plt.plot( 1.0-XMgB_ol_rw, P_ol_rw/1.e9, 'b-',  linewidth=3, label='ol-out (ol, rw)')

plt.plot( 1.0-XMgA_wad_rw, P_wad_rw/1.e9, 'g-',  linewidth=3, label='rw-out (wad, rw)')
plt.plot( 1.0-XMgB_wad_rw, P_wad_rw/1.e9, 'b-',  linewidth=3, label='wad-out (wad, rw)')

plt.errorbar( ol_wad_data[3], ol_wad_data[0], xerr=[ol_wad_data[4], ol_wad_data[4]], yerr=[ol_wad_data[1], ol_wad_data[1]], fmt='--o', color='red', linestyle='none', label='ol')
plt.errorbar( ol_wad_data[5], ol_wad_data[0], xerr=[ol_wad_data[6], ol_wad_data[6]], yerr=[ol_wad_data[1], ol_wad_data[1]], fmt='--o', color='green', linestyle='none', label='wad')
plt.errorbar( ol_rw_data[3], ol_rw_data[0], xerr=[ol_rw_data[4], ol_rw_data[4]], yerr=[ol_rw_data[1], ol_rw_data[1]], fmt='--o', color='red', linestyle='none', label='ol')
plt.errorbar( ol_rw_data[5], ol_rw_data[0], xerr=[ol_rw_data[6], ol_rw_data[6]], yerr=[ol_rw_data[1], ol_rw_data[1]], fmt='--o', color='blue', linestyle='none', label='rw')
plt.errorbar( wad_rw_data[3], wad_rw_data[0], xerr=[wad_rw_data[4], wad_rw_data[4]], yerr=[wad_rw_data[1], wad_rw_data[1]], fmt='--o', color='green', linestyle='none', label='wad')
plt.errorbar( wad_rw_data[5], wad_rw_data[0], xerr=[wad_rw_data[6], wad_rw_data[6]], yerr=[wad_rw_data[1], wad_rw_data[1]], fmt='--o', color='blue', linestyle='none', label='rw')


plt.title('Mg2SiO4-Fe2SiO4 phase diagram')
plt.xlabel("X_Fe")
plt.ylabel("Pressure (GPa)")
plt.legend(loc='center right')
plt.show()

