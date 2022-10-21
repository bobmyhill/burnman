import numpy as np

# https://serc.carleton.edu/NAGTWorkshops/mineralogy/mineral_physics/tensors.html
# see also Nye, 1959
p11 = 4.79501845e-01
p12 = -6.91663267e-02
p13 = -4.62784420e-02
p14 = -1.64139386e-01
p33 = 3.64442730e-01
p44 = 7.31250186e-01
p66 = 2.0 * (p11 - p12)
alpha1_params = {
    "a": np.array(
        [
            [p11, p12, p13, p14, 0.0, 0.0],
            [p12, p11, p13, -p14, 0.0, 0.0],
            [p13, p13, p33, 0.0, 0.0, 0.0],
            [p14, -p14, 0.0, p44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, p44, 2.0 * p14],
            [0.0, 0.0, 0.0, 0.0, 2.0 * p14, p66],
        ]
    ),
    "b_1": np.zeros((6, 6)),
    "c_1": np.ones((6, 6)),
    "b_2": np.zeros((6, 6)),
    "c_2": np.ones((6, 6)),
    "d": np.zeros((6, 6)),
}

alpha2_params = {
    "a": np.array(
        [
            [p11, p12, p13, -p14, 0.0, 0.0],
            [p12, p11, p13, p14, 0.0, 0.0],
            [p13, p13, p33, 0.0, 0.0, 0.0],
            [-p14, p14, 0.0, p44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, p44, -2.0 * p14],
            [0.0, 0.0, 0.0, 0.0, -2.0 * p14, p66],
        ]
    ),
    "b_1": np.zeros((6, 6)),
    "c_1": np.ones((6, 6)),
    "b_2": np.zeros((6, 6)),
    "c_2": np.ones((6, 6)),
    "d": np.zeros((6, 6)),
}


def psi_func_mbr(f, Pth, params):
    dPsidf = params["a"] + params["b_1"] * params["c_1"] * np.exp(params["c_1"] * f)
    Psi = (
        0.0
        + params["a"] * f
        + params["b_1"] * (np.exp(params["c_1"] * f) - 1.0)
        + params["d"] * Pth / 1.0e9
    )
    dPsidPth = params["d"] / 1.0e9
    return (Psi, dPsidf, dPsidPth)
