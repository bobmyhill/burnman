import numpy as np
import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../../burnman'):
    sys.path.insert(1, os.path.abspath('../../..'))


from burnman.solutionbases import transform_solution_to_new_basis


def make_child_solutions(solutions):
    child_solutions = {'mg_fe_bdg': transform_solution_to_new_basis(solutions['bdg'],
                                                                    np.array([[1., 0., 0., 0., 0.],
                                                                              [0., 1., 0., 0., 0.]]),
                                                                    solution_name='mg-fe bridgmanite'),
                       'mg_al_bdg': transform_solution_to_new_basis(solutions['bdg'],
                                                                    np.array([[1., 0., 0., 0., 0.],
                                                                              [0., 0., 1., 0., 0.]]),
                                                                    solution_name='mg-al bridgmanite'),
                       'al_mg_cor': transform_solution_to_new_basis(solutions['cor'],
                                                                    np.array([[1., 0., 0., 0., 0.],
                                                                              [0., 0., 0., 1., 0.]]),
                                                                    solution_name='al-mg corundum'),
                       'py_alm_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                    np.array([[1., 0., 0., 0., 0., 0.],
                                                                              [0., 1., 0., 0., 0., 0.]]),
                                                                    solution_name='py-alm garnet'),

                       'py_gr_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                   np.array([[1., 0., 0., 0., 0., 0.],
                                                                             [0., 0., 1., 0., 0., 0.]]),
                                                                   solution_name='py-gr garnet'),

                       'py_gr_dmaj_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                        np.array([[1., 0., 0., 0., 0., 0.],
                                                                                  [0., 0., 1., 0., 0., 0.],
                                                                                  [0., 0., 0., 0., 1., 0.]]),
                                                                        solution_name='py-gr-dmaj garnet'),
                       'py_gr_nmaj_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                        np.array([[1., 0., 0., 0., 0., 0.],
                                                                                  [0., 0., 1., 0., 0., 0.],
                                                                                  [0., 0., 0., 0., 0., 1.]]),
                                                                        solution_name='py-gr-nmaj garnet'),
                       'NCMAS_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                        np.array([[1., 0., 0., 0., 0., 0.],
                                                                                  [0., 0., 1., 0., 0., 0.],
                                                                                  [0., 0., 0., 0., 1., 0.],
                                                                                  [0., 0., 0., 0., 0., 1.]]),
                                                                        solution_name='NCMAS garnet'),
                       'ca-mg_dmaj_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                        np.array([[-1., 0., 1., 0., 1., 0.],
                                                                                  [0., 0., 0., 0., 1., 0.]]),
                                                                        solution_name='ca-mg-dmaj garnet'),

                       'py_alm_gr_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                       np.array([[1., 0., 0., 0., 0., 0.],
                                                                                 [0., 1., 0., 0., 0., 0.],
                                                                                 [0., 0., 1., 0., 0., 0.]]),
                                                                       solution_name='py-alm-gr garnet'),

                       'alm_sk_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                    np.array([[0., 1., 0., 0., 0., 0.],
                                                                              [0., 1., -1., 1., 0., 0.]]),
                                                                    solution_name='alm-sk garnet'),
                       'lp_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                np.array([[1., 0., 0., 0., 0., 0.],
                                                                          [0., 1., 0., 0., 0., 0.],
                                                                          [0., 0., 1., 0., 0., 0.],
                                                                          [0., 0., 0., 1., 0., 0.]]),
                                                                solution_name='py-alm-gr-andr garnet'),
                       'lp_FMASO_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                      np.array([[1., 0., 0., 0., 0., 0.],
                                                                                [0., 1., 0., 0., 0., 0.],
                                                                                [1., 0., -1., 1., 0., 0.]]),
                                                                      solution_name='py-alm-kho garnet'),
                       'FMASO_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                   np.array([[1., 0., 0., 0., 0., 0.],
                                                                             [0., 1., 0., 0., 0., 0.],
                                                                             [1., 0., -1., 1., 0., 0.],
                                                                             [0., 0., 0., 0., 1., 0.]]),
                                                                   solution_name='py-alm-kho-dmaj garnet'),
                       'FMAS_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                  np.array([[1., 0., 0., 0., 0., 0.],
                                                                            [0., 1., 0., 0., 0., 0.],
                                                                            [0., 0., 0., 0., 1., 0.]]),
                                                                  solution_name='py-alm-dmaj garnet'),
                       'xna_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                 np.array([[1., 0., 0., 0., 0., 0.],
                                                                           [0., 1., 0., 0., 0., 0.],
                                                                           [0., 0., 1., 0., 0., 0.],
                                                                           [0., 0., 0., 1., 0., 0.],
                                                                           [0., 0., 0., 0., 1., 0.]]),
                                                                 solution_name='py-alm-gr-andr-dmaj garnet'),
                       'xmj_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                 np.array([[1., 0., 0., 0., 0., 0.],
                                                                           [0., 1., 0., 0., 0., 0.],
                                                                           [0., 0., 1., 0., 0., 0.],
                                                                           [0., 0., 0., 1., 0., 0.],
                                                                           [0., 0., 0., 0., 0., 1.]]),
                                                                 solution_name='py-alm-gr-andr-nagt garnet'),

                       'sk_gt': transform_solution_to_new_basis(solutions['gt'], np.array([[0., 1., -1., 1., 0., 0.]]),
                                                                    solution_name='skiagite'),

                       'py_dmaj_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                     np.array([[1., 0., 0., 0., 0., 0.],
                                                                               [0., 0., 0., 0., 1., 0.]]),
                                                                     solution_name='py-dmaj garnet'),
                       'py_nmaj_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                     np.array([[1., 0., 0., 0., 0., 0.],
                                                                               [0., 0., 0., 0., 0., 1.]]),
                                                                     solution_name='py-nmaj garnet'),
                       'py_dmaj_nmaj_gt': transform_solution_to_new_basis(solutions['gt'],
                                                                          np.array([[1., 0., 0., 0., 0., 0.],
                                                                                    [0., 0., 0., 0., 1., 0.],
                                                                                    [0., 0., 0., 0., 0., 1.]]),
                                                                          solution_name='py-dmaj-nmaj garnet'),
                       'sp_herc': transform_solution_to_new_basis(solutions['sp'],
                                                                  np.array([[1., 0., 0., 0., 0.],
                                                                            [0., 1., 0., 0., 0.]]),
                                                                  solution_name='spinel-hercynite'),

                       'ring': transform_solution_to_new_basis(solutions['sp'],
                                                               np.array([[0., 0., 0., 1., 0.],
                                                                         [0., 0., 0., 0., 1.]]),
                                                               solution_name='ringwoodite'),

                       'herc_mt_frw': transform_solution_to_new_basis(solutions['sp'],
                                                                      np.array([[0., 1., 0., 0., 0.],
                                                                                [0., 0., 1., 0., 0.],
                                                                                [0., 0., 0., 0., 1.]]),
                                                                      solution_name='herc-mt-frw spinel'),
                       'mt_frw': transform_solution_to_new_basis(solutions['sp'],
                                                                 np.array([[0., 0., 1., 0., 0.],
                                                                           [0., 0., 0., 0., 1.]]),
                                                                 solution_name='mt-frw spinel'),

                       'mg_fe_opx': transform_solution_to_new_basis(solutions['opx'],
                                                                    np.array([[1., 0., 0., 0., 0.],
                                                                              [0., 1., 0., 0., 0.],
                                                                              [0., 0., 0., 0., 1.]]),
                                                                    solution_name='Mg-Fe orthopyroxene with order-disorder'),

                       'oen_mgts': transform_solution_to_new_basis(solutions['opx'],
                                                                   np.array([[1., 0., 0., 0., 0.],
                                                                             [0., 0., 1., 0., 0.]]),
                                                                   solution_name='MAS orthopyroxene'),

                       'oen_mgts_odi': transform_solution_to_new_basis(solutions['opx'],
                                                                       np.array([[1., 0., 0., 0., 0.],
                                                                                 [0., 0., 1., 0., 0.],
                                                                                 [0., 0., 0., 1., 0.]]),
                                                                       solution_name='CMAS orthopyroxene'),

                       'oen_odi': transform_solution_to_new_basis(solutions['opx'],
                                                                  np.array([[1., 0., 0., 0., 0.],
                                                                            [0., 0., 0., 1., 0.]]),
                                                                  solution_name='CMS orthopyroxene'),

                       'ofs_fets': transform_solution_to_new_basis(solutions['opx'],
                                                                   np.array([[0., 1., 0., 0., 0.],  # ofs
                                                                             [-1., 0., 1., 0., 1.]]),  # fets = - oen + mgts + ofm
                                                                   solution_name='FAS orthopyroxene'),

                       'mg_fe_hpx': transform_solution_to_new_basis(solutions['hpx'],
                                                                    np.array([[1., 0., 0., 0., 0.],
                                                                              [0., 1., 0., 0., 0.],
                                                                              [0., 0., 0., 0., 1.]]),
                                                                    solution_name='Mg-Fe HP clinopyroxene with order-disorder'),

                       'cfm_hpx': transform_solution_to_new_basis(solutions['hpx'],
                                                                  np.array([[1., 0., 0., 0., 0.],
                                                                            [0., 1., 0., 0., 0.],
                                                                            [0., 0., 0., 1., 0.],
                                                                            [0., 0., 0., 0., 1.]]),
                                                                  solution_name='CFM HP clinopyroxene with order-disorder'),

                       'di_cen': transform_solution_to_new_basis(solutions['cpx'],
                                                                 np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                                           [0., 0., 1., 0., 0., 0., 0.]]),
                                                                 solution_name='CMS clinopyroxene'),

                       'di_hed': transform_solution_to_new_basis(solutions['cpx'],
                                                                 np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                                           [0., 1., 0., 0., 0., 0., 0.]]),
                                                                 solution_name='di-hed clinopyroxene'),

                       'di_cen_cats': transform_solution_to_new_basis(solutions['cpx'],
                                                                      np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                                                [0., 0., 1., 0., 0., 0., 0.],
                                                                                [0., 0., 0., 0., 1., 0., 0.]]),
                                                                      solution_name='CMAS clinopyroxene'),

                       'di_jd': transform_solution_to_new_basis(solutions['cpx'],
                                                                np.array([[1., 0., 0., 0., 0., 0., 0.],
                                                                          [0., 0., 0., 0., 0., 1., 0.]]),
                                                                solution_name='di-jd clinopyroxene'),

                       'cen_jd': transform_solution_to_new_basis(solutions['cpx'],
                                                                 np.array([[0., 0., 1., 0., 0., 0., 0.],
                                                                           [0., 0., 0., 0., 0., 1., 0.]]),
                                                                 solution_name='cen-jd clinopyroxene')}
    return child_solutions
