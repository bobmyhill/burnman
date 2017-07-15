import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = 8.31446

def update_solution(npr, s, T):
    s.npr = np.array(npr)
    
    s.X = s.n/np.sum(s.n)
    s.Xpr= s.npr/np.sum(s.npr)
    
    
    s.Z = [1./( (1./s.Zpr[0][0]*(2.*s.npr[0][0]/(2.*s.npr[0][0] + s.npr[0][1]))) +
                (1./s.Zpr[0][1]*(s.npr[0][1]/(2.*s.npr[0][0] + s.npr[0][1])))) ,
           1./( (1./s.Zpr[1][1]*(2.*s.npr[1][1]/(2.*s.npr[1][1] + s.npr[0][1]))) +
                (1./s.Zpr[1][0]*(s.npr[0][1]/(2.*s.npr[1][1] + s.npr[0][1])))) ]
    
    s.Y = s.Z*s.n/np.sum(s.Z*s.n)

    s.S = -R*(np.sum(s.n*np.log(s.X)) +
                (s.npr[0][0]*np.log(s.Xpr[0][0]/(s.Y[0]*s.Y[0])) +
                 s.npr[1][1]*np.log(s.Xpr[1][1]/(s.Y[1]*s.Y[1])) +
                 s.npr[0][1]*np.log(s.Xpr[0][1]/(2.*s.Y[0]*s.Y[1]))))
    
    
    s.G = (s.npr[0][1]/2.)*s.delta_g(s.Xpr)[0][1] - T*s.S
    s.H = s.G + T*s.S
    
    # return #[s.n[0] - (2.*s.npr[0][0]/s.Zpr[0][0] + s.npr[0][1]/s.Zpr[0][1]), # eq. 21
    # s.n[1] - (2.*s.npr[1][1]/s.Zpr[1][1] + s.npr[0][1]/s.Zpr[1][0]), # eq. 22


def gibbs_at_order(n01, s, T):
    c = [[((s.n[0] - n01[0]/s.Zpr[0][1])*s.Zpr[0][0]/2.), n01[0]],
         [0., ((s.n[1] - n01[0]/s.Zpr[1][0])*s.Zpr[1][1]/2.)]]
    update_solution(c, s, T)
    return s.G

class KCl_MgCl2():
    def __init__(self):
        self.Zpr = np.array([[6., 3.],
                             [6., 6.]])
        self.delta_g = lambda Xpr : np.array([[0., -17497. - 1026.*Xpr[0][0] - 14801.*Xpr[1][1]],
                                              [0., 0.]])


class fig1_binary3():
    def __init__(self):
        self.Zpr = np.array([[2., 2.],
                             [2., 2.]])
        self.delta_g = lambda Xpr : np.array([[0., -42.e3],
                                              [0., 0.]])







        

# plot stuff

s = KCl_MgCl2()
#s = fig1_binary3()
Xs = np.linspace(0.001, 0.999, 91)

plt.subplot(132)
fig1 = mpimg.imread('figures/deltaH_KCl_MgCl2.png')
plt.imshow(fig1, extent=[0, 1, -20000, 0], aspect='auto')

for T in np.linspace(1073., 1473., 3):

    Gs = np.zeros_like(Xs)
    Hs = np.zeros_like(Xs)
    Ss = np.zeros_like(Xs)
    success = []
    for i, X in enumerate(Xs):
        s.n = np.array([1. - X, X])

        # find maximum number of pairs (maximum ordering)
        max_n01 = np.min([s.n[0]*s.Zpr[0][1], s.n[1]*s.Zpr[1][0]]) # see eqn. 21, 22
        m = minimize(gibbs_at_order, [max_n01*0.9999], args=(s, T),
                     method='SLSQP',
                     bounds=[(0., max_n01)])
        success.append(m.success)
        if m.success == True:
            Gs[i] = s.G
            Hs[i] = s.H
            Ss[i] = s.S


    mask = [i for (i, TF) in enumerate(success) if TF == True]
    plt.subplot(131)
    plt.plot(Xs[mask], Gs[mask])
    plt.subplot(132)
    plt.plot(Xs[mask], Hs[mask])
    plt.subplot(133)
    plt.plot(Xs[mask], Ss[mask])
plt.show()



