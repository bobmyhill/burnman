# Benchmarks for the solid solution class
import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

import burnman
from burnman import minerals
from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import *
from burnman.chemicalpotentials import *
from burnman import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import optimize
atomic_masses=read_masses()

def gibbs_murnaghan(pressure, params):
    G0, V0, K0, Kprime0 = params
    exponent=(Kprime0-1.0)/Kprime0
    return G0 + V0*(K0/(Kprime0 - 1.0)*(np.power((1.+Kprime0/K0*pressure),exponent)-1.)) 

def pressure(VoverV0, dT, K0, a, b, dPdT):
    f = (np.power(VoverV0, -2./3.)-1.)/2.
    return 3.*K0*f*np.power((1.+2.*f), 2.5)*(1. + a*f +b*f*f) +dPdT*dT

'''
Let's check the data for the fo-mwd-mrw transition ( et al).
'''

'''
fo=minerals.HP_2011_ds62.fo()
mwd=minerals.HP_2011_ds62.mwd()
mrw=minerals.HP_2011_ds62.mrw()
fa=minerals.HP_2011_ds62.fa()
fwd=minerals.HP_2011_ds62.fwd()
frw=minerals.HP_2011_ds62.frw()
'''


fo=minerals.HHPH_2013.fo()
mwd=minerals.HHPH_2013.mwd()
mrw=minerals.HHPH_2013.mrw()
fa=minerals.HHPH_2013.fa()
fwd=minerals.HHPH_2013.fwd()
frw=minerals.HHPH_2013.frw()


fo=minerals.SLB_2011.forsterite()
mwd=minerals.SLB_2011.mgwa()
mrw=minerals.SLB_2011.mgri()
fa=minerals.SLB_2011.fayalite()
fwd=minerals.SLB_2011.fewa()
frw=minerals.SLB_2011.feri()


mrw.set_state(1.e5, 1400.)
print mrw.V/mrw.params['V_0']

'''
Initialise solid solutions
'''

class mg_fe_olivine(SolidSolution):
    def __init__(self):
        # Name
        self.name='olivine'
        self.endmembers = [[fo, '[Mg]2SiO4'],[fa, '[Fe]2SiO4']]
        self.enthalpy_interaction=[[4.000e3]]
        self.volume_interaction=[[1.e-7]]
        self.type='symmetric'
        burnman.SolidSolution.__init__(self)

class mg_fe_wadsleyite(SolidSolution):
    def __init__(self):
        # Name
        self.name='wadsleyite'
        self.endmembers = [[mwd, '[Mg]2SiO4'],[fwd, '[Fe]2SiO4']]
        self.enthalpy_interaction=[[15.000e3]]
        self.volume_interaction=[[0.e-7]]
        self.type='symmetric'
        burnman.SolidSolution.__init__(self)

class mg_fe_ringwoodite(SolidSolution):
    def __init__(self):
        # Name
        self.name='ringwoodite'
        self.endmembers = [[mrw, '[Mg]2SiO4'],[frw, '[Fe]2SiO4']]
        self.enthalpy_interaction=[[8.3200e3]]
        self.volume_interaction=[[0.e-7]]
        self.type='symmetric'
        burnman.SolidSolution.__init__(self)

ol=mg_fe_olivine()
wd=mg_fe_wadsleyite()
rw=mg_fe_ringwoodite()


rw.set_composition([0.91, 0.09])
rw.set_state(20.68e9, 1273.)
print rw.V
rw.set_state(21.24e9, 1273.)
print rw.V
print 489.22*constants.Avogadro/1.e30/8.
exit

# MG-RINGWOODITE is a problem. High quality data suggests that the metastable transition of forsterite to Mg-ringwoodite should be at higher pressure than predicted by the fo-wad and wad-ringwoodite curves. Frost gets around this by assuming a much smaller volume for ringwoodite, but this is inconsistent with PVT observations and thermal expansivity measurements at ambient pressure.

# Possible solutions: 
# Mg-Si disordering in rw at HT? Hazen et al., 1993
# Mg-Fe ordering stabilising rw at ~ 50%  

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


'''
#frw.params['K_0'] = frw.params['K_0'] + 0.e9
frw.params['H_0'] = frw.params['H_0'] - 0.4e3 # increment to H_0 for frw to fit data of Yagi et al., 1987
#mrw.params['K_0'] = mrw.params['K_0'] + 20.e9
#mrw.params['H_0'] = mrw.params['H_0'] - 4.e3

#fwd.params['K_0'] = fwd.params['K_0'] - 10.e9
fwd.params['H_0'] = fwd.params['H_0'] + 2.5e3
fwd.params['a_0'] = fwd.params['a_0'] #- 0.4e-6
fwd.params['V_0'] = fwd.params['V_0'] - 0.4e-6
'''

P=1.e5
T=1673.
phases=[[fo, 4.6053], [fa, 4.8494], [mwd, 4.2206], [fwd, 4.4779], [mrw, 4.1484], [frw, 4.3813]]
for phase, volume in phases:
    phase.set_state(P, T)
    print phase.name, phase.V, volume*1.e-5, phase.V-volume*1.e-5


def eqm_pressure(min1, min2):
    def eqm(arg, T):
        P=arg[0]
        min1.set_state(P,T)
        min2.set_state(P,T)
        return [min1.gibbs - min2.gibbs]
    return eqm

temperatures=np.linspace(800.,2000.,21)
fo_mwd_pressures=np.empty_like(temperatures)
mwd_mrw_pressures=np.empty_like(temperatures)

for idx, T in enumerate(temperatures):
    fo_mwd_pressures[idx]=optimize.fsolve(eqm_pressure(fo, mwd), 10.e9, args=(T))[0]
    mwd_mrw_pressures[idx]=optimize.fsolve(eqm_pressure(mwd, mrw), 20.e9, args=(T))[0]

plt.plot( temperatures, fo_mwd_pressures/1.e9, 'r-', linewidth=1., label='fo mwd')
plt.plot( temperatures, mwd_mrw_pressures/1.e9, 'b-', linewidth=1., label='mwd mrw')
plt.title('Mg2SiO4 phase diagram')
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (GPa)")
plt.legend(loc='lower right')
plt.show()


# EOS for mwd comes from
# Thermal expansion: Ye et al. (2009), Trots et al. (2012)
# In excellent agreement with Holland and Powell data

# EOS for mrw comes from
#Li et al., 2003; Higo et al., 2006

'''
Let's check the data for the fa-frw transition (Yagi et al, 1987).
'''

fa_out_data=[]
frw_out_data=[]
for line in open('Yagi_ASSA_1987_fa_frw.dat'):
    content=line.strip().split()
    if content[0] != '%':
        if content[6] == 'fa->':
            fa_out_data.append([float(content[1])+273.15, pressure(float(content[2]), float(content[1])-25., 238.8, 1.796, -5.0, 0.0286)/10., float(content[4]), float(content[5])])
        else:
            frw_out_data.append([float(content[1])+273.15, pressure(float(content[2]), float(content[1])-25., 238.8, 1.796, -5.0, 0.0286)/10., float(content[4]), float(content[5])])

# The second column here is pressure calculated using the NaCl equation of state from Birch, 1986
T_Yagi_fa_out, P_Yagi_fa_out, P_Yagi_Decker_fa_out, Perr_Decker_Yagi_fa_out = zip(*fa_out_data)
T_Yagi_frw_out, P_Yagi_frw_out, P_Yagi_Decker_frw_out, Perr_Decker_Yagi_frw_out = zip(*frw_out_data)

fa_frw_pressures=np.empty_like(temperatures)
fwd_frw_pressures=np.empty_like(temperatures)
fa_fwd_pressures=np.empty_like(temperatures)
for idx, T in enumerate(temperatures):
    fa_frw_pressures[idx]=optimize.fsolve(eqm_pressure(fa, frw), 5.e9, args=(T))[0]
    fa_fwd_pressures[idx]=optimize.fsolve(eqm_pressure(fa, fwd), 5.e9, args=(T))[0]
    fwd_frw_pressures[idx]=optimize.fsolve(eqm_pressure(fwd, frw), 5.e9, args=(T))[0]

plt.plot( T_Yagi_fa_out, P_Yagi_fa_out, marker='.', linestyle='None', label='fa->')
plt.plot( T_Yagi_frw_out, P_Yagi_frw_out, marker='*', linestyle='None', label='frw->')
plt.plot( temperatures, fa_frw_pressures/1.e9, 'r-', linewidth=1., label='fa frw')
plt.plot( temperatures, fwd_frw_pressures/1.e9, 'r--', linewidth=1., label='fwd frw')
plt.plot( temperatures, fa_fwd_pressures/1.e9, 'r--', linewidth=1., label='fa fwd')
plt.title('Fe2SiO4 phase diagram')
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (GPa)")
plt.legend(loc='lower right')
plt.show()



def eqm_P_xMgB(A, B):
    def eqm(arg, T, xMgA):
        P=arg[0]
        xMgB=arg[1]

        A.set_composition([xMgA, 1.0-xMgA])
        A.set_state(P,T)

        B.set_composition([xMgB, 1.0-xMgB])
        B.set_state(P,T)

        diff_mu_Mg2SiO4=A.partial_gibbs[0] - B.partial_gibbs[0]
        diff_mu_Fe2SiO4=A.partial_gibbs[1] - B.partial_gibbs[1]
        return [diff_mu_Mg2SiO4, diff_mu_Fe2SiO4]
    return eqm

def eqm_P_xMgABC(A, B, C):
    def eqm(arg, T):
        P=arg[0]
        xMgA=arg[1]
        xMgB=arg[2]
        xMgC=arg[3]

        A.set_composition([xMgA, 1.0-xMgA])
        A.set_state(P,T)

        B.set_composition([xMgB, 1.0-xMgB])
        B.set_state(P,T)

        C.set_composition([xMgC, 1.0-xMgC])
        C.set_state(P,T)

        diff_mu_Mg2SiO4_0=A.partial_gibbs[0] - B.partial_gibbs[0]
        diff_mu_Fe2SiO4_0=A.partial_gibbs[1] - B.partial_gibbs[1]
        diff_mu_Mg2SiO4_1=A.partial_gibbs[0] - C.partial_gibbs[0]
        diff_mu_Fe2SiO4_1=A.partial_gibbs[1] - C.partial_gibbs[1]


        return [diff_mu_Mg2SiO4_0, diff_mu_Fe2SiO4_0, diff_mu_Mg2SiO4_1, diff_mu_Fe2SiO4_1]
    return eqm

T=1673. # K

invariant=optimize.fsolve(eqm_P_xMgABC(ol, wd, rw), [15.e9, 0.2, 0.3, 0.4], args=(T))
print invariant

XMgA_ol_wad=np.linspace(invariant[1], 0.9999, 21)
XMgA_ol_rw=np.linspace(0.0001, invariant[1], 21)
XMgA_wad_rw=np.linspace(invariant[2], 0.9999, 21)

P_ol_wad=np.empty_like(XMgA_ol_wad)
XMgB_ol_wad=np.empty_like(XMgA_ol_wad)

P_ol_rw=np.empty_like(XMgA_ol_wad)
XMgB_ol_rw=np.empty_like(XMgA_ol_wad)

P_wad_rw=np.empty_like(XMgA_ol_wad)
XMgB_wad_rw=np.empty_like(XMgA_ol_wad)


for idx, XMgA in enumerate(XMgA_ol_wad):
    XMgB_guess=1.0-((1.0-XMgA_ol_wad[idx])*0.8)
    P_ol_wad[idx], XMgB_ol_wad[idx] = optimize.fsolve(eqm_P_xMgB(ol, wd), [5.e9, XMgB_guess], args=(T, XMgA_ol_wad[idx]))
    XMgB_guess=1.0-((1.0-XMgA_ol_rw[idx])*0.8)
    P_ol_rw[idx], XMgB_ol_rw[idx] = optimize.fsolve(eqm_P_xMgB(ol, rw), [5.e9, XMgB_guess], args=(T, XMgA_ol_rw[idx]))
    XMgB_guess=1.0-((1.0-XMgA_wad_rw[idx])*0.8)
    P_wad_rw[idx], XMgB_wad_rw[idx] = optimize.fsolve(eqm_P_xMgB(wd, rw), [5.e9, XMgB_guess], args=(T, XMgA_wad_rw[idx]))

plt.plot( 1.0-np.array([invariant[1], invariant[2], invariant[3]]), np.array([invariant[0], invariant[0], invariant[0]])/1.e9, color='black', linewidth=1, label='invariant')

plt.plot( 1.0-XMgA_ol_wad, P_ol_wad/1.e9, 'r-', linewidth=1, label='wad-out (ol, wad)')
plt.plot( 1.0-XMgB_ol_wad, P_ol_wad/1.e9, 'g-', linewidth=1, label='ol-out (ol, wad)')

plt.plot( 1.0-XMgA_ol_rw, P_ol_rw/1.e9, 'r-',  linewidth=1, label='rw-out (ol, rw)')
plt.plot( 1.0-XMgB_ol_rw, P_ol_rw/1.e9, 'b-',  linewidth=1, label='ol-out (ol, rw)')

plt.plot( 1.0-XMgA_wad_rw, P_wad_rw/1.e9, 'g-',  linewidth=1, label='rw-out (wad, rw)')
plt.plot( 1.0-XMgB_wad_rw, P_wad_rw/1.e9, 'b-',  linewidth=1, label='wad-out (wad, rw)')

plt.errorbar( ol_wad_data[3], ol_wad_data[0], xerr=[ol_wad_data[4], ol_wad_data[4]], yerr=[ol_wad_data[1], ol_wad_data[1]], fmt='--o', color='red', linestyle='none', label='ol')
plt.errorbar( ol_wad_data[5], ol_wad_data[0], xerr=[ol_wad_data[6], ol_wad_data[6]], yerr=[ol_wad_data[1], ol_wad_data[1]], fmt='--o', color='green', linestyle='none', label='wad')
plt.errorbar( ol_rw_data[3], ol_rw_data[0], xerr=[ol_rw_data[4], ol_rw_data[4]], yerr=[ol_rw_data[1], ol_rw_data[1]], fmt='--o', color='red', linestyle='none', label='ol')
plt.errorbar( ol_rw_data[5], ol_rw_data[0], xerr=[ol_rw_data[6], ol_rw_data[6]], yerr=[ol_rw_data[1], ol_rw_data[1]], fmt='--o', color='blue', linestyle='none', label='rw')
plt.errorbar( wad_rw_data[3], wad_rw_data[0], xerr=[wad_rw_data[4], wad_rw_data[4]], yerr=[wad_rw_data[1], wad_rw_data[1]], fmt='--o', color='green', linestyle='none', label='wad')
plt.errorbar( wad_rw_data[5], wad_rw_data[0], xerr=[wad_rw_data[6], wad_rw_data[6]], yerr=[wad_rw_data[1], wad_rw_data[1]], fmt='--o', color='blue', linestyle='none', label='rw')


plt.title('Mg2SiO4-Fe2SiO4 phase diagram')
plt.xlabel("X_Mg")
plt.ylabel("Pressure (GPa)")
plt.legend(loc='upper right')
plt.show()

#G0, V0, K0, Kprime0
# 1e4 bar = 1e9 Pa
# 1e0 bar = 1e5 Pa

T=1673.
pressures=np.linspace(1.e5, 18.e9, 101)
diff=np.zeros(shape=(6, 101))
for i, P in enumerate(pressures):
    fo.set_state(P, T)
    fa.set_state(P,T)
    mwd.set_state(P, T)
    fwd.set_state(P, T)
    mrw.set_state(P, T)
    frw.set_state(P, T)

    diff[0][i]=fo.gibbs - gibbs_murnaghan(P/1.e5, [-2555725, 4.6053, 957000., 4.6])
    diff[1][i]=fa.gibbs - gibbs_murnaghan(P/1.e5, [-1976015, 4.8494, 998484., 4.])
    diff[2][i]=mwd.gibbs - gibbs_murnaghan(P/1.e5, [-2514481, 4.2206, 1462544., 4.21])
    diff[3][i]=fwd.gibbs - gibbs_murnaghan(P/1.e5, [-1955055, 4.4779, 1399958., 4.])
    diff[4][i]=mrw.gibbs - gibbs_murnaghan(P/1.e5, [-2500910, 4.1484, 1453028., 4.4])
    diff[5][i]=frw.gibbs -  gibbs_murnaghan(P/1.e5, [-1950160, 4.3813, 1607810., 5.])


phases=[fo, fa, mwd, fwd, mrw, frw]
for i, phase in enumerate(phases):
    plt.plot( pressures/1.e9, diff[i], linewidth=1, label=phase.name)

plt.title('Mg2SiO4-Fe2SiO4 endmember free energy differences')
plt.xlabel("Pressure (GPa)")
plt.ylabel("Gibbs difference")
plt.legend(loc='upper right')
plt.show()

Z=8.
nA=6.02214e23
voltoa=1.e30

V00=539.26*(nA/Z/voltoa)
V25=546.59*(nA/Z/voltoa)
V100=V00+4.*(V25-V00)
print mwd.params['V_0'], V00
print fwd.params['V_0'], V100 #(Hazen et al., 2000; extrapolated from fwd25)
