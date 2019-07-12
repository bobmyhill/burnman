import numpy as np

from input_dataset import *

# Matsuzaka fper-rw-stv equilibria
with open('data/Matsuzaka_et_al_2000_rw_wus_stv.dat', 'r') as f:
    ds = [line.split() for line in f if line.split() != [] and line[0] != '#']

Matsuzaka_2000_assemblages = []
for (run_id, P, T, t, ph1, Mg_ring, Mgerr_ring, ph2, Mg_mw, Mgerr_mw) in ds:

    if float(t) > 179.:
        assemblage = burnman.Composite([child_solutions['ring'],
                                        solutions['mw'],
                                        endmembers['stv']])

        assemblage.experiment_id = run_id
        assemblage.nominal_state = np.array([float(P)*1.e9, float(T)])
        assemblage.state_covariances = np.array([[0.5e9*0.5e9, 0.],[0., 10.*10]])

        ring_fractions = np.array([float(Mg_ring)/100., 1. - (float(Mg_ring)/100.)])
        ring_sig = float(Mgerr_ring)*float(Mgerr_ring)
        ring_cov = np.array([[ring_sig, -ring_sig],
                             [-ring_sig, ring_sig]])
        mw_fractions = np.array([float(Mg_mw)/100., 1. - (float(Mg_mw)/100.)])
        mw_sig = float(Mgerr_mw)*float(Mgerr_mw)
        mw_cov = np.array([[mw_sig, -mw_sig],
                           [-mw_sig, mw_sig]])

        assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
        assemblage.stored_compositions[0] = (ring_fractions, ring_cov)
        assemblage.stored_compositions[1] = (mw_fractions, mw_cov)

        Matsuzaka_2000_assemblages.append(assemblage)
