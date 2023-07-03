ordered_properties = [
    ("name", "name"),
    ("formula", "formula"),
    ("n", "n"),
    ("Pref", "$P\\textsubscript{ref} (Pa)$"),
    ("molar_mass", "mass (kg/mol)"),
    ("T_0", "$T\\textsubscript{ref}$ (K)"),
    ("T_einstein", "$T\\textsubscript{ein}$ (K)"),
    ("H_Pref", "$\\mathcal{H}(P\\textsubscript{ref}, T\\textsubscript{ref})$ (J/mol)"),
    ("S_Pref", "$S(P\\textsubscript{ref}, T\\textsubscript{ref})$ (J/K/mol)"),
    ("V_0", "$V_0$ (m$^3$/mol)"),
    ("K_0", "$K_0$ (Pa)"),
    ("Kprime_0", "$K'_0$"),
    ("Kdprime_0", "$K''_0$ (/Pa)"),
    ("a_0", "$a_0$ (/K)"),
    ("Cp_Pref", ["J/K/mol", "J/K$^2$/mol", "JK/mol", "JK$^{-1/2}$/mol"]),
]

possible_elements = ["Fe", "Mg", "Si", "H", "O"]

"""
from model_parameters import (
    mg_wadsleyite_params,
    fe_wadsleyite_params,
    majorite_params,
    water_params,
)
mins = [mg_wadsleyite_params, fe_wadsleyite_params, majorite_params, water_params]
print('\\hline')
for (prp, prpname) in ordered_properties:
    if prp != 'Cp_Pref' and prp != 'formula':
        print(f'{prpname}', end='')
        for m in mins:
            try:
                print(f' & {m[prp]:.6g}', end='')
            except:
                print(f' & {m[prp]}', end='')
        print(' \\\\', end='\n')
    elif prp == 'Cp_Pref':
        for i in range(4):
            print(f'$C_{i}$ ({prpname[i]})', end='')
            for m in mins:
                print(f' & {m[prp][i]:.4g}', end='')
            print(' \\\\', end='\n')
    else:
        print(f'{prpname}', end='')
        for m in mins:
            print(' & \ce{', end='')
            for el in possible_elements:
                if el in m[prp]:
                    nel = m[prp][el]
                    if nel % 1 < 1.e-5:
                        if nel > 1:
                            print(f'{el}{int(nel)}', end='')
                        else:
                            print(f'{el}', end='')
                    else:
                        print(f'{el}_{{{nel}}}', end='')
            print(f'}}', end='')
        print('\\\\', end='\n')
    if prp == 'name':
        print('\\hline')
"""
