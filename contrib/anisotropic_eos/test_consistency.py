def check_eos_consistency(m, P=1.e9, T=300., tol=1.e-4, verbose=False,
                          including_shear_properties=True):
    """
    Compute numerical derivatives of the gibbs free energy of a mineral
    under given conditions, and check these values against those provided
    analytically by the equation of state

    Parameters
    ----------
    m : mineral
        The mineral for which the equation of state
        is to be checked for consistency
    P : float
        The pressure at which to check consistency
    T : float
        The temperature at which to check consistency
    tol : float
        The fractional tolerance for each of the checks
    verbose : boolean
        Decide whether to print information about each
        check
    including_shear_properties : boolean
        Decide whether to check shear information,
        which is pointless for liquids and equations of state
        without shear modulus parameterizations

    Returns
    -------
    consistency: boolean
        If all checks pass, returns True

    """
    dT = 1.
    dP = 1000.


    m.set_state(P, T)
    G0 = m.gibbs
    S0 = m.S
    V0 = m.V

    # cell parameters and volume

    # alpha_P -> cell vectors at two different temperatures (constant V)
    # alpha_V -> cell vectors at two different temperatures (constant P)
    # beta_T = S_Tijkl delta_kl -> cell vectors at two different pressures (constant T)
    # beta_S = S_Nijkl delta_kl -> cell vectors at two different pressures (constant S)

    # C_Nijkl -> S_Nijkl ^(-1)
    # C_Tijkl -> S_Tijkl ^(-1)


    expr = ['G = F + PV', 'G = H - TS', 'G = E - TS + PV']
    eq = [[m.gibbs, (m.helmholtz + P*m.V)],
          [m.gibbs, (m.H - T*m.S)],
          [m.gibbs, (m.molar_internal_energy - T*m.S + P*m.V)]]

    m.set_state(P, T + dT)
    G1 = m.gibbs
    S1 = m.S
    V1 = m.V

    m.set_state(P + dP, T)
    G2 = m.gibbs
    V2 = m.V

    # T derivatives
    m.set_state(P, T + 0.5*dT)
    expr.extend(['S = -dG/dT', 'alpha = 1/V dV/dT', 'C_p = T dS/dT'])
    eq.extend([[m.S, -(G1 - G0)/dT],
               [m.alpha, (V1 - V0)/dT/m.V],
               [m.molar_heat_capacity_p, (T + 0.5*dT)*(S1 - S0)/dT]])

    # P derivatives
    m.set_state(P + 0.5*dP, T)
    expr.extend(['V = dG/dP', 'K_T = -V dP/dV'])
    eq.extend([[m.V, (G2 - G0)/dP],
               [m.K_T, -0.5*(V2 + V0)*dP/(V2 - V0)]])

    expr.extend(['C_v = Cp - alpha^2*K_T*V*T', 'K_S = K_T*Cp/Cv', 'gr = alpha*K_T*V/Cv'])
    eq.extend([[m.molar_heat_capacity_v, m.molar_heat_capacity_p - m.alpha*m.alpha*m.K_T*m.V*T],
               [m.K_S, m.K_T*m.molar_heat_capacity_p/m.molar_heat_capacity_v],
               [m.gr, m.alpha*m.K_T*m.V/m.molar_heat_capacity_v]])

    expr.append('Vphi = np.sqrt(K_S/rho)')
    eq.append([m.bulk_sound_velocity, np.sqrt(m.K_S/m.rho)])

    if including_shear_properties:
        expr.extend(['Vp = np.sqrt((K_S + 4G/3)/rho)', 'Vs = np.sqrt(G_S/rho)'])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eq.extend([[m.p_wave_velocity, np.sqrt((m.K_S + 4.*m.G/3.)/m.rho)],
                       [m.shear_wave_velocity, np.sqrt(m.G/m.rho)]])
            if len(w) == 1:
                print(w[0].message)
                print('\nYou can suppress this message by setting the '
                      'parameter\nincluding_shear_properties to False '
                      'when calling check_eos_consistency.\n')
        note = ''
    else:
        note = ' (not including shear properties)'
