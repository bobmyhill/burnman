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
