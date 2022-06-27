# Anisotropic solution

def psi_func(f, Pth, X, params):
    Psi = 0.
    dPsidf = 0.
    dPsidPth = 0.
    dPsidX = 0.
    return (Psi, dPsidf, dPsidPth, dPsidX)


# Initialised objects
# Scalar solution model
# Psi function
# Reference endmember
# Endmembers to X vector (each component should sum to zero)
# Components of the X vector that are freely varying on seismic timescales
