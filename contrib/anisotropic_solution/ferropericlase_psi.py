import numpy as np


def psi_func(f, Pth, params):
    dPsidf = (params['a'] + params['b_1']*params['c_1'] *
              np.exp(params['c_1']*f) +
              params['b_2']*params['c_2']*np.exp(params['c_2']*f))
    dPsidf = (dPsidf + Pth/1.e9*(params['b_3']*params['c_3'] *
                                 np.exp(params['c_3']*f) +
                                 params['b_4']*params['c_4'] *
                                 np.exp(params['c_4']*f)))

    Psi = (0. + params['a']*f + params['b_1']*np.exp(params['c_1']*f) +
           params['b_2']*np.exp(params['c_2']*f))
    Psi = (Psi + Pth/1.e9*(params['b_3']*np.exp(params['c_3']*f) +
                           params['b_4']*np.exp(params['c_4']*f)))

    dPsidPth = (params['b_3']*np.exp(params['c_3']*f) +
                params['b_4']*np.exp(params['c_4']*f))/1.e9
    return (Psi, dPsidf, dPsidPth)


def per_anisotropic_parameters():
    x = np.array([4.78290975e-01,
                  4.37591672e-02, -7.74683571e-01,
                  -1.93952207e-01,  5.82744809e+00,
                  1.11294499e+00,  7.99641884e-01,
                  -5.26577925e-01,  5.82314077e+00,
                  -2.38281540e+00,  2.05834924e-03,
                  1.17385211e-02,  1.72275539e+00,
                  1.88626859e+00,  1.84158921e-04,
                  -3.87909545e-03,  2.27384829e+01,
                  4.13816434e+00])

    anisotropic_parameters = {'a': np.zeros((6, 6)),
                              'b_1': np.zeros((6, 6)),
                              'c_1': np.ones((6, 6)),
                              'b_2': np.zeros((6, 6)),
                              'c_2': np.ones((6, 6)),
                              'b_3': np.zeros((6, 6)),
                              'c_3': np.ones((6, 6)),
                              'b_4': np.zeros((6, 6)),
                              'c_4': np.ones((6, 6))}

    i = 0
    for p in ['a', 'b_1', 'c_1', 'b_2', 'c_2', 'b_3', 'c_3', 'b_4', 'c_4']:
        if p == 'a':
            anisotropic_parameters[p][:3, :3] = (1. - 3.*x[i])/6.
        elif p[0] == 'b' or p == 'd':
            anisotropic_parameters[p][:3, :3] = -3.*x[i]/6.
        else:
            anisotropic_parameters[p][:3, :3] = x[i]

        anisotropic_parameters[p][0, 0] = x[i]
        anisotropic_parameters[p][1, 1] = x[i]
        anisotropic_parameters[p][2, 2] = x[i]
        i = i + 1
        anisotropic_parameters[p][3, 3] = x[i]
        anisotropic_parameters[p][4, 4] = x[i]
        anisotropic_parameters[p][5, 5] = x[i]
        i = i + 1
    assert len(x) == i

    return anisotropic_parameters
