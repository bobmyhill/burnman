import os, sys
sys.path.insert(1,os.path.abspath('..'))

import numpy as np
import burnman

coe = burnman.minerals.HP_2011_ds62.coe()
stv = burnman.minerals.HP_2011_ds62.stv()
stv_SLB = burnman.minerals.SLB_2011.stishovite()
gr = burnman.minerals.HP_2011_ds62.gr()

# Better fit to Ito et al. (1974) with these parameters
#stv.params['a_0'] = 1.4e-5
#stv.params['V_0'] = 1.4006e-5

coe.set_state(0.001e8, 2000.+273.15)
stv.set_state(0.001e8, 2000.+273.15)
print coe.gibbs - -1162.09e3, 'CHECK ZERO'
print stv.gibbs - -1094.73e3, 'CHECK ZERO'

coe.set_state(100.e8, 2000.+273.15)
stv.set_state(100.e8, 2000.+273.15)
print coe.gibbs - -959.30e3, 'CHECK ZERO'
print stv.gibbs - -949.73e3, 'CHECK ZERO'

print 'Checking volume'
dP = 100.
stv.set_state(1.e5, 2000.+273.15)
G0 = stv.gibbs
V0 = stv.V
stv.set_state(1.e5+dP, 2000.+273.15)
G1 = stv.gibbs
print V0, (G1-G0)/dP

print 'Checking thermal expansion'
dT = 0.1
stv.set_state(1.e5, 298.15)
V0 = stv.V
stv.set_state(1.e5, 298.15+dT)
V1 = stv.V
print 1/V0*(V1-V0)/dT, stv.params['a_0']


f = open('burnman_stv_volumes.dat', 'w')
temperatures = np.linspace(50., 900., 101)
stv_volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    stv.set_state(1.e5, T)
    stv_volumes[i] = stv.V*1.e5
    f.write(str(T)+' '+str(stv_volumes[i])+'\n')

f.close()

f = open('burnman_stv_volumes_298K.dat', 'w')
pressures = np.linspace(1.e5, 230.e9, 101)
stv_volumes = np.empty_like(pressures)
for i, P in enumerate(pressures):
    stv.set_state(P, 298.)
    stv_volumes[i] = stv.V*1.e5
    f.write(str(P/1.e9)+' '+str(stv_volumes[i])+'\n')

f.close()

f = open('burnman_gr_volumes.dat', 'w')
temperatures = [300., 600., 800., 1000.]
pressures = np.linspace(1.e5, 4.e9, 101)
gr_volumes = np.empty_like(pressures)
for i, T in enumerate(temperatures):
    f.write('>> -W1,black \n')
    for j, P in enumerate(pressures):
        gr.set_state(P, T)
        gr_volumes[j] = gr.V*1.e5
        f.write(str(P/1.e9*10.)+' '+str(gr_volumes[j])+'\n')

f.close()
