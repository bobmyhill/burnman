import numpy as np

from input_dataset import burnman

from burnman.eos_fitting import fit_PTV_data

data = np.loadtxt('data/Dymshits_et_al_2014_Na2MgSi5O12_EoS.dat')

data[:,2] *= 1.e9
data[:,3] *= 1.e9
data[:,9] *= 1.e-6
data[:,10] *= 1.e-6

namaj = burnman.minerals.HHPH_2013.nagt()
print(namaj.params)

fit_PTV_data(mineral=namaj, fit_params=['V_0', 'K_0', 'Kprime_0', 'a_0'],
             data=data[:,[2,4,9]],
             data_covariances=[],
             param_tolerance=1.e-7,
             max_lm_iterations=50, verbose=True)
print(namaj.params)


# now do nagt (half way between py and namaj)

py = burnman.minerals.HP_2011_ds62.py()
V_py = py.evaluate(['V'], data[:,2], data[:,4])[0]

data[:,9] = (data[:,9] + V_py)/2.
data[:,10] /= np.sqrt(2.)

nagt = burnman.minerals.HHPH_2013.nagt()
fit_PTV_data(mineral=nagt, fit_params=['V_0', 'K_0', 'Kprime_0', 'a_0'],
             data=data[:,[2,4,9]],
             data_covariances=[],
             param_tolerance=1.e-7,
             max_lm_iterations=50, verbose=True)
print(nagt.params)


nagt.params['Kprime_0'] = 4.0
nagt.params['Kdprime_0'] = -4./178.e9
fit_PTV_data(mineral=nagt, fit_params=['V_0', 'K_0', 'a_0'],
             data=data[:,[2,4,9]],
             data_covariances=[],
             param_tolerance=1.e-7,
             max_lm_iterations=50, verbose=True)
print(nagt.params)
