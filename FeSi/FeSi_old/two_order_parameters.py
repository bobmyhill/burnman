'''
def eqm_order_2(order_parameters, deltaHs, Ws, X, T):
    Q1, Q2 = order_parameters
    deltaH0, deltaH1 = deltaHs
    W=Ws

# Proportions
    p=[1.-X-0.5*Q1-0.75*Q2,Q2, Q1, X-0.5*Q1-0.25*Q2]
    
# Non ideal activities
    RTlng=[0., 0., 0., 0.]
    RTlng[0] = (1-p[0])*(p[1]*W[0][1] + p[2]*W[0][2] + p[3]*W[0][3]) - p[1]*(p[2]*W[1][2] + p[3]*W[1][3]) - p[2]*(p[3]*W[2][3])
    RTlng[1] = (1-p[1])*(p[2]*W[1][2] + p[3]*W[1][3] + p[0]*W[0][1]) - p[2]*(p[3]*W[2][3] + p[0]*W[0][2]) - p[3]*(p[0]*W[0][3])
    RTlng[2] = (1-p[2])*(p[3]*W[2][3] + p[0]*W[0][2] + p[1]*W[1][2]) - p[3]*(p[0]*W[0][3] + p[1]*W[1][3]) - p[0]*(p[1]*W[0][1])
    RTlng[3] = (1-p[3])*(p[0]*W[0][3] + p[1]*W[1][3] + p[2]*W[2][3]) - p[0]*(p[1]*W[0][1] + p[2]*W[0][2]) - p[1]*(p[2]*W[1][2])

    XAFe=1. - X + 0.5*Q1 + 0.25*Q2
    XBFe=1. - X - 0.5*Q1 + 0.25*Q2
    XCFe=1. - X - 0.5*Q1 - 0.75*Q2

    XASi=1.-XAFe
    XBSi=1.-XBFe
    XCSi=1.-XCFe

    lnKd0=(1./8.)*np.log((XAFe*XAFe*XBSi*XCSi)/(XASi*XASi*XBFe*XCFe))
    
    lnKd1=(1./16.)*np.log((XAFe*XAFe*XBFe*XCSi*XCSi*XCSi)/(XASi*XASi*XBSi*XCFe*XCFe*XCFe))

    # mu_i = G_i + RTlna
    
    # 0 = Fe0.5Si0.5 - 0.5 Fe - 0.5 Si
    # deltaH = H_reaction
    eqn0=deltaH0 + (RTlng[2] - 0.5*RTlng[0] - 0.5*RTlng[3]) + R*T*lnKd0
    eqn1=deltaH1 + (RTlng[1] - 0.75*RTlng[0] - 0.25*RTlng[3]) + R*T*lnKd1

    return [eqn0, eqn1]

# Fe, Fe3Si, FeSi, Si
Ws=[[0., 0., 16000., 26000.],[0., 0., 0., 0.],[0., 0., 0., 16000.]]
deltaHs=[-6000., -6000.]
T=2200.
X=0.3


# Test
m=0.5
n=0.5
DeltaH=-6000.
W=[26000., 16000., 16000.]

order=optimize.fsolve(eqm_order, [0.599], args=(X, T, m, n, DeltaH, W))
print eqm_order(order, X, T, m, n, DeltaH, W)
print order


order=optimize.fsolve(eqm_order_2, [0.999, 0.199], args=(deltaHs, Ws, X, T))
print eqm_order_2(order, deltaHs, Ws, X, T)
print order
'''
