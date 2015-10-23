# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU GPL v2 or later.


import inspect
import slb
import dks_liquid
import dks_solid
import mie_grueneisen as mg
import mie_grueneisen_debye as mgd
import vinet_anderson_grueneisen as v_ag
import birch_murnaghan as bm
import modified_tait as mt
import hp 
import cork
from equation_of_state import EquationOfState


def create(method):
    """
    Creates an instance of an EquationOfState from a string,
    a class EquationOfState, or an instance of EquationOfState.
    """
    if isinstance(method, basestring):
        if method == "slb2":
            return slb.SLB2()
        elif method == "mg":
            return mg.MG()
        elif method == "mgd2":
            return mgd.MGD2()
        elif method == "mgd3":
            return mgd.MGD3()
        elif method == "slb3":
            return slb.SLB3()
        elif method == "dks_l":
            return dks_liquid.DKS_L()
        elif method == "dks_s":
            return dks_solid.DKS_S()
        elif method == "bm2":
            return bm.BM2()
        elif method == "bm3":
            return bm.BM3()
        elif method == "v_ag":
            return v_ag.V_AG()
        elif method == "mt":
            return mt.MT()
        elif method == "hp_tmt":
            return hp.HP_TMT()
        elif method == "cork":
            return cork.CORK()
        else:
            raise Exception("unsupported material method " + method)
    elif isinstance(method, EquationOfState):
        return method
    elif inspect.isclass(method) and issubclass(method, EquationOfState):
        return method()
    else:
        raise Exception("unsupported material method " + method.__class__.__name__)
