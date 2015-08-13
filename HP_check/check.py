import os, sys
sys.path.insert(1,os.path.abspath('..'))

import numpy as np
import burnman

stv = burnman.minerals.HP_2011_ds62.stv()
gr = burnman.minerals.HP_2011_ds62.gr()

f = open('burnman_stv_volumes.dat', 'w')
temperatures = np.linspace(50., 900., 101)
stv_volumes = np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    stv.set_state(1.e5, T)
    stv_volumes[i] = stv.V*1.e5
    f.write(str(T)+' '+str(stv_volumes[i])+'\n')

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
