import numpy as np

def fit_PVT_data(mineral):
    def fit_data(PT, V_0, K_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        Kprime_0 = 4.0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/K_0
        mineral.params['a_0'] = a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)

        return volumes
    return fit_data

def fit_PVT_data_Ka(mineral):
    def fit_data(PT, K_0, a_0):
        mineral.params['K_0'] = K_0
        Kprime_0 = mineral.params['Kprime_0'] 
        mineral.params['Kdprime_0'] = -Kprime_0/K_0
        mineral.params['a_0'] = a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)

        return volumes
    return fit_data

def fita0(mineral):
    def fit_data(PT, a_0):
        mineral.params['a_0']=a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)
            
        return volumes
    return fit_data

def fit_PVT_data_full_noV(mineral):
    def fit_data(PT, K_0, Kprime_0, a_0):
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/K_0
        mineral.params['a_0'] = a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)

        return volumes
    return fit_data

def fit_PVT_data_full_noVK(mineral):
    def fit_data(PT, Kprime_0, a_0):
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/mineral.params['K_0']
        mineral.params['a_0'] = a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)

        return volumes
    return fit_data

def fit_PVT_data_full(mineral):
    def fit_data(PT, V_0, K_0, Kprime_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -Kprime_0/K_0
        mineral.params['a_0'] = a_0

        volumes=[]
        for P, T in zip(*PT):
            mineral.set_state(P, T)
            volumes.append(mineral.V)

        return volumes
    return fit_data

def fitCp(mineral):
    def fit(temperatures, a, b, c, d):
        mineral.params['Cp']=[a, b, c, d]
        Cp=np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            mineral.set_state(1.e5, T)
            Cp[i]=mineral.C_p
        return Cp
    return fit

def fitalpha(mineral):
    def fit(temperatures, alpha, T_einst):
        mineral.params['a_0']=alpha
        mineral.params['T_einstein']=T_einst
        mineral.set_state(1.e5, 293.)
        L_293=np.power(mineral.V, 1./3.)
        alphastar=np.empty_like(temperatures)
        for i, T in enumerate(temperatures):
            deltaT=0.1
            mineral.set_state(1.e5, T-deltaT)
            L_T0=np.power(mineral.V, 1./3.)
            mineral.set_state(1.e5, T+deltaT)
            L_T1=np.power(mineral.V, 1./3.)
            DeltaL=L_T1 - L_T0
            DeltaT=2.*deltaT
            alphastar[i]=(1./L_293)*(DeltaL/DeltaT)
        return alphastar
    return fit

def fitK0(mineral, standard_mineral):
    def fit(pressures, K_0):
        mineral.params['K_0']=K_0
        fractional_volumes=[]
        for i, P in enumerate(pressures):
            mineral.set_state(P, 300.)
            fractional_volumes.append(mineral.V/standard_mineral.params['V_0'])
        return np.array(fractional_volumes)
    return fit

def fitK0_V0(mineral, standard_mineral):
    def fit(pressures, K_0, V_0):
        mineral.params['K_0']=K_0
        mineral.params['V_0']=V_0
        fractional_volumes=[]
        for i, P in enumerate(pressures):
            mineral.set_state(P, 300.)
            fractional_volumes.append(mineral.V/standard_mineral.params['V_0'])
        return np.array(fractional_volumes)
    return fit

def fitK_p0(mineral, standard_mineral):
    def fit(pressures, K_0, Kprime_0):
        mineral.params['K_0']=K_0
        mineral.params['Kprime_0']=Kprime_0
        mineral.params['Kdprime_0']=-Kprime_0/K_0
        fractional_volumes=[]
        for i, P in enumerate(pressures):
            mineral.set_state(P, 300.)
            fractional_volumes.append(mineral.V/standard_mineral.params['V_0'])
        return np.array(fractional_volumes)
    return fit

def fit_EOS(mineral, standard_mineral):
    def fit(pressures, V_0, K_0, Kprime_0):
        mineral.params['V_0']=V_0
        mineral.params['K_0']=K_0
        mineral.params['Kprime_0']=Kprime_0
        fractional_volumes=[]
        for i, P in enumerate(pressures):
            mineral.set_state(P, 300.)
            fractional_volumes.append(mineral.V/standard_mineral.params['V_0'])
        return np.array(fractional_volumes)
    return fit

def fitV0(mineral, standard_mineral):
    def fit(pressures, V_0):
        mineral.params['V_0']=V_0
        fractional_volumes=[]
        for i, P in enumerate(pressures):
            mineral.set_state(P, 300.)
            fractional_volumes.append(mineral.V/standard_mineral.params['V_0'])
        return np.array(fractional_volumes)
    return fit

def eqm_pressure(minerals, multiplicities):
    def eqm(P, T):
        gibbs = 0.
        for i, mineral in enumerate(minerals):
            mineral.set_state(P[0], T)
            gibbs = gibbs + mineral.gibbs*multiplicities[i]
        return gibbs
    return eqm

def eqm_temperature(minerals, multiplicities):
    def eqm(T, P):
        gibbs = 0.
        for i, mineral in enumerate(minerals):
            mineral.set_state(P, T[0])
            gibbs = gibbs + mineral.gibbs*multiplicities[i]
        return gibbs
    return eqm
