# Initial guess.
def fitV_full(mineral):
    def fit(data, V_0, K_0, Kprime_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        mineral.params['a_0'] = a_0
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV(mineral):
    def fit(data, V_0, K_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        mineral.params['a_0'] = a_0
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_wout_a0(mineral):
    def fit(data, V_0, K_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_w_Kprime_wout_a0(mineral):
    def fit(data, V_0, K_0, Kprime_0):
        mineral.params['V_0'] = V_0
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_a0(mineral):
    def fit(data, V_0, a_0):
        mineral.params['V_0'] = V_0
        mineral.params['a_0'] = a_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for i, datum in enumerate(data):
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_w_Kprime(mineral):
    def fit(data, V_0, K_0, Kprime_0, a_0):
        mineral.params['V_0'] = V_0 
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        mineral.params['a_0'] = a_0
        
        vols=[]
        for i, datum in enumerate(data):            
            mineral.set_state(datum[0], datum[1])
            vols.append(mineral.V)
        return vols
    return fit

def fitV_T0(mineral):
    def fit(pressures, V_0, K_0, Kprime_0):
        mineral.params['V_0'] = V_0 
        mineral.params['K_0'] = K_0
        mineral.params['Kprime_0'] = Kprime_0
        mineral.params['Kdprime_0'] = -mineral.params['Kprime_0']/mineral.params['K_0']
        
        vols=[]
        for pressure in pressures:
            mineral.set_state(pressure, 298.15)
            vols.append(mineral.V)
        return vols
    return fit
