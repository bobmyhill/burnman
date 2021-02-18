import numpy as np


def solve_quadratic(a, b, c, sgn):
    return (-b + sgn*np.sqrt(b*b - 4.*a*c))/(2.*a)

def partition(p_Fe_bulk, f_liq, KD):
    """
    p_Fe_bulk is the molar ratio Fe/(Mg+Fe) in the bulk
    f_liq is the fraction of combined Mg and Fe in the liq (relative to the bulk)
    KD is the partition coefficient (p_Feliq*p_Mgsol)/(p_Fesol*p_Mgliq)
    """
    a = -f_liq*(KD - 1.)
    b = (KD - 1.)*p_Fe_bulk - (1. - f_liq)*KD - f_liq
    c = p_Fe_bulk

    p_Fe_liq = solve_quadratic(a, b, c, -1.)
    p_Fe_sol = (p_Fe_bulk - f_liq*p_Fe_liq)/(1. - f_liq)

    return p_Fe_liq, p_Fe_sol


print(partition(0.3, 0.3, 0.3))
exit()
