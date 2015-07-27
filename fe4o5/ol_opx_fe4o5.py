import os, sys, numpy as np, matplotlib.pyplot as plt
#hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1,os.path.abspath('..'))

# Benchmarks for the solid solution class
import burnman
from burnman import minerals
from burnman.chemicalpotentials import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.optimize as optimize

from mineral_models import *
from equilibrium_functions import eqm_pressure

ol = olivine()
wad = wadsleyite()
rw=ringwoodite()

fm45 = MgFeFe2O5()

hen = minerals.HP_2011_ds62.hen()
hfs = minerals.HP_2011_ds62.fs()
hfm=ordered_fm_hpx()

# EMOD minerals
mag = minerals.HP_2011_ds62.mag()
fo = minerals.HP_2011_ds62.fo()
mwd = minerals.HP_2011_ds62.mwd()
mrw = minerals.HP_2011_ds62.mrw()
diam = minerals.HP_2011_ds62.diam()
EMOD = [hen, mag, fo, diam]
EMWD = [hen, mag, mwd, diam]
EMRD = [hen, mag, mrw, diam]

# QFM minerals
q= minerals.HP_2011_ds62.q()
fa = minerals.HP_2011_ds62.fa()
mt = minerals.HP_2011_ds62.mt()
QFM = [q, fa, mt]

hem = minerals.HP_2011_ds62.hem()

O2=minerals.HP_2011_fluids.O2()



P=0.1e9
T=1273.

hen.set_state(P,T)
hfs.set_state(P,T)
hfm.set_state(P,T)

print hen.gibbs, hfs.gibbs, hfm.gibbs, 0.5*(hen.gibbs + hfs.gibbs) - hfm.gibbs


opx=orthopyroxene()

################################
# Check order parameter finder #
################################

def opx_composition(arg, X_Mg, P, T):
    Q=arg[0]
    opx.set_composition([X_Mg-0.5*Q, 1-X_Mg-0.5*Q, Q])
    opx.set_state(P,T)
    return 0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2] 


P=1.e9
T=1473.15
X_Mgs=np.linspace(0.0,1.0,101)
G=np.empty_like(X_Mgs)
Q=np.empty_like(X_Mgs)
for idx, X_Mg in enumerate(X_Mgs):
    optimize.fsolve(opx_composition, [0.2], args=(X_Mg, P, T))
    G[idx]=opx.gibbs
    Q[idx]=opx.molar_fractions[2]

plt.plot( X_Mgs, G, '-', linewidth=3., label='Gibbs opx')

plt.title('')
plt.xlabel("X_Mg opx")
plt.ylabel("G")
plt.legend(loc='lower right')
plt.show()


#################################
# Find ol-opx-Fe4O5 equilibrium #
#################################

# Constraints
# ol+opx composition
# 1.4 (Mg+Fe):Si (i.e. 4 Mg2SiO4 + 3 Mg2Si2O6)


# Unknowns
# opx order parameter
# opx composition
# Fe4O5 composition

# Reactions
# opx (order parameter)
# Fe-Mg exchange (ol-opx)
# Fe-Mg exchange (ol-Fe4O5)

def ol_polymorph_opx_fm45_equilibrium(args, X_Mg_olopx, ol_polymorph, P, T):
    Q, X_Mg_opx, X_Mg_Fe4O5 = args

    # X_Mg_olopx = 
    # (p_ol*Mg_ol + p_opx*Mg_opx) / (p_ol + p_opx)
    X_Mg_ol = (X_Mg_olopx * (p_ol_polymorphs + p_opx) - p_opx*X_Mg_opx)/p_ol_polymorphs
    
    ol_polymorph.set_composition([X_Mg_ol, 1.-X_Mg_ol])
    opx.set_composition([X_Mg_opx-0.5*Q, 1-X_Mg_opx-0.5*Q, Q])
    fm45.set_composition([X_Mg_Fe4O5, 1.-X_Mg_Fe4O5])

    ol_polymorph.set_state(P, T)
    opx.set_state(P, T)
    fm45.set_state(P, T)

    equations = [ 0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2],
                  ( ol_polymorph.partial_gibbs[0] + opx.partial_gibbs[1] ) - ( ol_polymorph.partial_gibbs[1] + opx.partial_gibbs[0] ), 
                  ( ol_polymorph.partial_gibbs[0] + 2.*fm45.partial_gibbs[1] ) - ( ol_polymorph.partial_gibbs[1] + 2.*fm45.partial_gibbs[0] ) ]
    
    return equations


def ol_polymorphs_opx_fm45_equilibrium(args, X_Mg_olwadopx, polymorph_1, polymorph_2, p_polymorph_1, T):
    Q, P, X_Mg_polymorph_1, X_Mg_polymorph_2, X_Mg_opx, X_Mg_Fe4O5 = args
    p_polymorph_2 = p_ol_polymorphs - p_polymorph_1

    # X_Mg_olwadopx = 
    # (p_ol*Mg_ol + p_wad*Mg_wad + p_opx*Mg_opx) / (p_ol + p_wad + p_opx)
    X_Mg_ol_polymorphs = (X_Mg_olwadopx * (p_ol_polymorphs + p_opx) - p_opx*X_Mg_opx)/p_ol_polymorphs

    polymorph_1.set_composition([X_Mg_polymorph_1, 1.-X_Mg_polymorph_1])
    polymorph_2.set_composition([X_Mg_polymorph_2, 1.-X_Mg_polymorph_2])
    opx.set_composition([X_Mg_opx-0.5*Q, 1-X_Mg_opx-0.5*Q, Q])
    fm45.set_composition([X_Mg_Fe4O5, 1.-X_Mg_Fe4O5])

    polymorph_1.set_state(P, T)
    polymorph_2.set_state(P, T)
    opx.set_state(P, T)
    fm45.set_state(P, T)

    equations = [ p_polymorph_1 * X_Mg_polymorph_1 + p_polymorph_2 * X_Mg_polymorph_2 - p_ol_polymorphs * X_Mg_ol_polymorphs,
                  0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2],
                  polymorph_1.partial_gibbs[0] - polymorph_2.partial_gibbs[0],
                  polymorph_1.partial_gibbs[1] - polymorph_2.partial_gibbs[1],
                  ( polymorph_2.partial_gibbs[0] + opx.partial_gibbs[1] ) - ( polymorph_2.partial_gibbs[1] + opx.partial_gibbs[0] ), 
                  ( polymorph_2.partial_gibbs[0] + 2.*fm45.partial_gibbs[1] ) - ( polymorph_2.partial_gibbs[1] + 2.*fm45.partial_gibbs[0] ) ]
    
    return equations


# Find the ol-wad-rw invariant
def ol_wad_rw_invariant(args, T):
    P, X_ol, X_wad, X_rw, X_opx, Q = args

    ol.set_composition([X_ol, 1.-X_ol])
    wad.set_composition([X_wad, 1.-X_wad])
    rw.set_composition([X_rw, 1.-X_rw])
    opx.set_composition([X_opx-0.5*Q, 1-X_opx-0.5*Q, Q])

    ol.set_state(P, T)
    wad.set_state(P, T)
    rw.set_state(P, T)
    opx.set_state(P, T)

    return [0.5*(opx.partial_gibbs[0] + opx.partial_gibbs[1]) - opx.partial_gibbs[2],
            ( ol.partial_gibbs[0] + opx.partial_gibbs[1] ) - ( ol.partial_gibbs[1] + opx.partial_gibbs[0] ),
            ol.partial_gibbs[0] - wad.partial_gibbs[0],
            wad.partial_gibbs[0] - rw.partial_gibbs[0],
            ol.partial_gibbs[1] - wad.partial_gibbs[1],
            wad.partial_gibbs[1] - rw.partial_gibbs[1]]


# Compositions
p_ol_polymorphs = 4./7.
p_opx = 3./7.

T = 1673.15
O2.set_state(1.e5,T)
X_Mg_olopx = 0.9

invariant = optimize.fsolve(ol_wad_rw_invariant, [13.e9, 0.8, 0.7, 0.5, 0.8, 0.1], args=(T))
P_inv, X_ol_inv, X_wad_inv, X_rw_inv, X_opx_inv, Q_opx_inv =  invariant

X_olopx_invariant = p_ol_polymorphs * X_ol_inv + p_opx * X_opx_inv
X_wadopx_invariant =  p_ol_polymorphs * X_wad_inv + p_opx * X_opx_inv
X_rwopx_invariant =  p_ol_polymorphs * X_rw_inv + p_opx * X_opx_inv

# First, plot ol-wad phase boundary
X_Mgs=np.linspace(X_olopx_invariant, 0.99, 21)
pressures_ol_wad_opx_fm45=np.empty_like(X_Mgs)
fO2_ol_wad_opx_fm45=np.empty_like(X_Mgs)
sol = [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7]
for i, X_Mg_olopx in enumerate(X_Mgs):
    print X_Mg_olopx
    sol = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, sol, args=(X_Mg_olopx, ol, wad, p_ol_polymorphs, T))
    pressures_ol_wad_opx_fm45[i] = sol[1]
    sol[1] += 1.e9
    fO2_ol_wad_opx_fm45[i] =  np.log10(fugacity(O2, [ol, opx, fm45]))
plt.plot(pressures_ol_wad_opx_fm45/1.e9, fO2_ol_wad_opx_fm45, label='ol-wad-opx-Fe4O5')

# Do the same for the wad-rw phase boundary
X_Mgs=np.linspace(X_wadopx_invariant, 0.99, 21)
pressures_wad_rw_opx_fm45=np.empty_like(X_Mgs)
fO2_wad_rw_opx_fm45=np.empty_like(X_Mgs)
sol = [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7]
for i, X_Mg_olopx in enumerate(X_Mgs):
    print X_Mg_olopx
    sol = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, sol, args=(X_Mg_olopx, wad, rw, p_ol_polymorphs, T))
    pressures_wad_rw_opx_fm45[i] = sol[1]
    sol[1] += 1.e9
    fO2_wad_rw_opx_fm45[i] =  np.log10(fugacity(O2, [wad, opx, fm45]))
plt.plot(pressures_wad_rw_opx_fm45/1.e9, fO2_wad_rw_opx_fm45, label='wad-rw-opx-Fe4O5')

# And for the ol-rw boundary
X_Mgs=np.linspace(0.4, X_olopx_invariant, 60)
pressures_ol_rw_opx_fm45=np.empty_like(X_Mgs)
fO2_ol_rw_opx_fm45=np.empty_like(X_Mgs)
sol = [0.14, 13.e9, 0.66, 0.33, 0.81, 0.20]
for i, X_Mg_olopx in enumerate(X_Mgs):
    print X_Mg_olopx
    sol = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, sol, args=(X_Mg_olopx, ol, rw, p_ol_polymorphs, T))
    pressures_ol_rw_opx_fm45[i] = sol[1]
    sol[1] += 1.e9
    fO2_ol_rw_opx_fm45[i] =  np.log10(fugacity(O2, [rw, opx, fm45]))
plt.plot(pressures_ol_rw_opx_fm45/1.e9, fO2_ol_rw_opx_fm45, label='ol-rw-opx-Fe4O5')


# Now plot contours
ol_contours=[]
wad_contours=[]
rw_contours=[]
pressure_range=[8.e9, 18.e9]

X_Mgs=np.linspace(0.6, 0.95, 8)
for i, X_Mg_olopx in enumerate(X_Mgs):
    print X_Mg_olopx

    if i%2 == 0:
        label_type=''
    else:
        label_type='-'

    # OLIVINE
    ol_wad_pressure = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7], \
                                          args=(X_Mg_olopx, ol, wad, p_ol_polymorphs, T))[1]
    ol_rw_pressure = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7], \
                                         args=(X_Mg_olopx, ol, rw, p_ol_polymorphs, T))[1]

    max_pressure = min(ol_wad_pressure, ol_rw_pressure)
    if max_pressure > pressure_range[0]:
        ol_pressures = np.linspace(pressure_range[0], max_pressure, 11)
        fO2_ol_opx_fm45 = np.empty_like(ol_pressures)
        for i, P in enumerate(ol_pressures):    
            sol = optimize.fsolve(ol_polymorph_opx_fm45_equilibrium, [0.0, 0.94, 0.7], args=(X_Mg_olopx, ol, P, T))
            fO2_ol_opx_fm45[i] =  np.log10(fugacity(O2, [ol, opx, fm45]))
        ol_contours.append([ol_pressures, fO2_ol_opx_fm45, str(X_Mg_olopx), '-W0.5,grey,'+label_type])
        plt.plot(ol_pressures/1.e9, fO2_ol_opx_fm45, label='ol-opx-Fe4O5')

    # WADSLEYITE
    wad_ol_pressure = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7], \
                                          args=(X_Mg_olopx, wad, ol, p_ol_polymorphs, T))[1]
    wad_rw_pressure = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7], \
                                         args=(X_Mg_olopx, wad, rw, p_ol_polymorphs, T))[1]

    if wad_ol_pressure < wad_rw_pressure:
        if wad_ol_pressure < pressure_range[0]:
            wad_ol_pressure = pressure_range[0]
        if wad_rw_pressure > pressure_range[1]:
            wad_rw_pressure = pressure_range[1]
        wad_pressures = np.linspace(wad_ol_pressure, wad_rw_pressure, 11)
        fO2_wad_opx_fm45 = np.empty_like(wad_pressures)    
        for i, P in enumerate(wad_pressures): 
            sol = optimize.fsolve(ol_polymorph_opx_fm45_equilibrium, [0.0, 0.94, 0.7], args=(X_Mg_olopx, wad, P, T))
            fO2_wad_opx_fm45[i] =  np.log10(fugacity(O2, [wad, opx, fm45]))
        wad_contours.append([wad_pressures, fO2_wad_opx_fm45, str(X_Mg_olopx), '-W0.5,grey,'+label_type])
        plt.plot(wad_pressures/1.e9, fO2_wad_opx_fm45, label='wad-opx-Fe4O5')


    # RINGWOODITE
    rw_ol_pressure = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7], \
                                          args=(X_Mg_olopx, rw, ol, p_ol_polymorphs, T))[1]
    rw_wad_pressure = optimize.fsolve(ol_polymorphs_opx_fm45_equilibrium, [0.0, 13.e9, 0.87, 0.88, 0.94, 0.7], \
                                         args=(X_Mg_olopx, rw, wad, p_ol_polymorphs, T))[1]

    min_pressure = max(rw_ol_pressure, rw_wad_pressure)
    if min_pressure < pressure_range[1]:
        rw_pressures = np.linspace(min_pressure, pressure_range[1], 11)
        fO2_rw_opx_fm45 = np.empty_like(rw_pressures)
        for i, P in enumerate(rw_pressures): 
            sol = optimize.fsolve(ol_polymorph_opx_fm45_equilibrium, [0.0, 0.94, 0.7], args=(X_Mg_olopx, rw, P, T))
            fO2_rw_opx_fm45[i] =  np.log10(fugacity(O2, [rw, opx, fm45]))
        rw_contours.append([rw_pressures, fO2_rw_opx_fm45, str(X_Mg_olopx), '-W0.5,grey,'+label_type])
        plt.plot(rw_pressures/1.e9, fO2_rw_opx_fm45, label='rw-opx-Fe4O5')
    
buffer_pressures = np.linspace(pressure_range[0], pressure_range[1], 21)
fO2_EMOD = np.empty_like(buffer_pressures)
fO2_QFM = np.empty_like(buffer_pressures)

OW_pressure = optimize.fsolve(eqm_pressure, 6.e9, args=(T, [fo, mwd], [1.,-1.]))[0]
WR_pressure = optimize.fsolve(eqm_pressure, 6.e9, args=(T, [mwd, mrw], [1.,-1.]))[0]

for i, P in enumerate(buffer_pressures):
    for mineral in QFM:
        mineral.set_state(P, T)
    fO2_QFM[i] = np.log10(fugacity(O2, QFM))
    
    if P < OW_pressure: # olivine
        for mineral in EMOD:
            mineral.set_state(P, T)
        fO2_EMOD[i] = np.log10(fugacity(O2, EMOD))
    elif P < WR_pressure:
        for mineral in EMWD: # wadsleyite
            mineral.set_state(P, T)
        fO2_EMOD[i] = np.log10(fugacity(O2, EMWD))
    else:
        for mineral in EMRD: # ringwoodite
            mineral.set_state(P, T)
        fO2_EMOD[i] = np.log10(fugacity(O2, EMRD))


plt.plot(buffer_pressures/1.e9, fO2_EMOD, label='EMOD')
plt.plot(buffer_pressures/1.e9, fO2_QFM, label='QFM')

plt.legend(loc='lower left')
plt.show()


# Export to file
f = open('ol_opx_fe4o5.dat', 'w')
f.write('>> -W1,black\n')
for i, P in enumerate(pressures_ol_wad_opx_fm45):
    f.write(str(P/1.e9)+' '+str(fO2_ol_wad_opx_fm45[i])+'\n')
f.write('>> -W1,black\n')
for i, P in enumerate(pressures_ol_rw_opx_fm45):
    f.write(str(P/1.e9)+' '+str(fO2_ol_rw_opx_fm45[i])+'\n')
f.write('>> -W1,black\n')
for i, P in enumerate(pressures_wad_rw_opx_fm45):
    f.write(str(P/1.e9)+' '+str(fO2_wad_rw_opx_fm45[i])+'\n')
f.write('>> -W1,red\n')
for i, P in enumerate(buffer_pressures):
    f.write(str(P/1.e9)+' '+str(fO2_EMOD[i])+'\n')
f.write('>> -W1,blue\n')
for i, P in enumerate(buffer_pressures):
    f.write(str(P/1.e9)+' '+str(fO2_QFM[i])+'\n')

for contour in ol_contours:
    pressures, fO2s, label, marker = contour
    f.write('>> '+marker+'\n')
    for i, P in enumerate(pressures):
        f.write(str(P/1.e9)+' '+str(fO2s[i])+' '+label+'\n')

for contour in wad_contours:
    pressures, fO2s, label, marker = contour
    f.write('>> '+marker+'\n')
    for i, P in enumerate(pressures):
        f.write(str(P/1.e9)+' '+str(fO2s[i])+' '+label+'\n')

for contour in rw_contours:
    pressures, fO2s, label, marker = contour
    f.write('>> '+marker+'\n')
    for i, P in enumerate(pressures):
        f.write(str(P/1.e9)+' '+str(fO2s[i])+' '+label+'\n')

f.write('\n')
f.close()
print 'ol_opx_fe4o5.dat (over)written'

###########################
# EQUILIBRIUM WITH GARNET #
###########################
garnet = CFMASO_garnet()

def eqm_gt_composition(args, P, T, p_gr_andr, mu_kho_sk_assemblage):
    p_py, p_andr = args

    p_gr = p_gr_andr - p_andr
    p_alm = 1. - p_py - p_gr - p_andr

    garnet.set_composition([p_py, p_alm, p_gr, p_andr])
    garnet.set_state(P, T)
    
    mu_kho_sk_gt = chemical_potentials([garnet],
                                [burnman.processchemistry.dictionarize_formula('Mg3Fe2Si3O12'),
                                 burnman.processchemistry.dictionarize_formula('Fe3Fe2Si3O12')])

    return [ mu_kho_sk_gt[0] - mu_kho_sk_assemblage[0],
             mu_kho_sk_gt[1] - mu_kho_sk_assemblage[1]]


# Set up problem
p_gr_andr = 0.0 # Ca concentration in garnet
X_Mg_olopx = 0.9
T = 1673.15
pressures = np.linspace(8.e9, 13.e9, 101)
Fe3sumFe = np.empty_like(pressures)
p_py = np.empty_like(pressures)
print 'Pressures (GPa)   Fe3+/sum(Fe)   [molar fractions]'
for i, P in enumerate(pressures):
    sol = optimize.fsolve(ol_polymorph_opx_fm45_equilibrium, [0.0, 0.94, 0.7], args=(X_Mg_olopx, ol, P, T))
    mu_ol_opx_fm45 = chemical_potentials([ol,opx,fm45],
                                         [burnman.processchemistry.dictionarize_formula('Mg3Fe2Si3O12'),
                                          burnman.processchemistry.dictionarize_formula('Fe3Fe2Si3O12')])

    optimize.fsolve(eqm_gt_composition, [0.75, 0.1], args=(P, T, p_gr_andr, mu_ol_opx_fm45))

    Fe3sumFe[i] = 2.*garnet.molar_fractions[3] / (3.*garnet.molar_fractions[1] + 2.*garnet.molar_fractions[3])
    p_py[i] = garnet.molar_fractions[0]

    garnet.set_composition([1., 0., -1., 1.])
    garnet.set_state(P, T)
    hem.set_state(P, T)
    hen.set_state(P, T)

    print garnet.gibbs, hem.gibbs + hen.gibbs*1.5, garnet.gibbs -  (hem.gibbs + hen.gibbs*1.5)

# Export to file
f = open('gt_compositions.dat', 'w')
f.write('>> -W1,black\n')
for i, P in enumerate(pressures):
    f.write(str(P/1.e9)+' '+str(Fe3sumFe[i])+' '+str(p_py[i])+'\n')

f.write('\n')
f.close()
print 'gt_compositions.dat (over)written'
