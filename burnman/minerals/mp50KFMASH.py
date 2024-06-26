# This file is part of BurnMan - a thermoelastic
# and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2024 by the BurnMan team, released under the GNU
# GPL v2 or later.

"""
mp50KFMASH
^^^^^^^^^^

HPx-eos solutions using endmembers from
dataset HP_2011_ds62.
The values in this document are all in S.I. units,
unlike those in the original THERMOCALC file.
This file is autogenerated using process_HPX_eos.py
"""

from numpy import array, nan
from . import HP_2011_ds62
from ..classes.mineral import Mineral
from ..classes.solution import Solution
from ..classes.solutionmodel import SymmetricRegularSolution
from ..classes.solutionmodel import AsymmetricRegularSolution
from ..classes.combinedmineral import CombinedMineral


annm = CombinedMineral([HP_2011_ds62.ann()], [1.0], [-3000.0, -0.0, 0.0], "annm")
obi = CombinedMineral(
    [HP_2011_ds62.ann(), HP_2011_ds62.phl()],
    [0.3333333333333333, 0.6666666666666666],
    [-3000.0, -0.0, 0.0],
    "obi",
)
fm = CombinedMineral(
    [HP_2011_ds62.en(), HP_2011_ds62.fs()], [0.5, 0.5], [-6600.0, -0.0, 0.0], "fm"
)
fspm = CombinedMineral([HP_2011_ds62.fspr()], [1.0], [-2000.0, -0.0, 0.0], "fspm")
spro = CombinedMineral(
    [HP_2011_ds62.fspr(), HP_2011_ds62.spr4()],
    [0.75, 0.25],
    [-3500.0, -0.0, 0.0],
    "spro",
)
mstm = CombinedMineral([HP_2011_ds62.mst()], [1.0], [0.0, -0.0, 0.0], "mstm")
ochl1 = CombinedMineral(
    [HP_2011_ds62.afchl(), HP_2011_ds62.clin(), HP_2011_ds62.daph()],
    [1.0, -1.0, 1.0],
    [3000.0, -0.0, 0.0],
    "ochl1",
)
ochl4 = CombinedMineral(
    [HP_2011_ds62.afchl(), HP_2011_ds62.clin(), HP_2011_ds62.daph()],
    [1.0, -0.2, 0.2],
    [2400.0, -0.0, 0.0],
    "ochl4",
)


class g(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "g"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.py(), "[Mgx]3"],
                [HP_2011_ds62.alm(), "[Fex]3"],
            ],
            energy_interaction=[[2500.0]],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class mu(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "mu"
        self.solution_model = AsymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.mu(), "[Alma][Sit1/2Alt1/2]2"],
                [HP_2011_ds62.cel(), "[Mgma][Sit]2"],
                [HP_2011_ds62.fcel(), "[Fema][Sit]2"],
            ],
            alphas=[0.63, 0.63, 0.63],
            energy_interaction=[[0.0, 0.0], [0.0]],
            volume_interaction=[
                [2.0000000000000003e-06, 2.0000000000000003e-06],
                [0.0],
            ],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class bi(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "bi"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.phl(), "[Mgm][Mgm]2[Sit1/2Alt1/2]2"],
                [annm, "[Fem][Fem]2[Sit1/2Alt1/2]2"],
                [obi, "[Fem][Mgm]2[Sit1/2Alt1/2]2"],
                [HP_2011_ds62.east(), "[Alm][Mgm]2[Alt]2"],
            ],
            energy_interaction=[
                [12000.0, 4000.0, 10000.0],
                [8000.0, 15000.0],
                [7000.0],
            ],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class opx(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "opx"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.en(), "[Mgm][Mgm][Sit]1/2"],
                [HP_2011_ds62.fs(), "[Fem][Fem][Sit]1/2"],
                [fm, "[Mgm][Fem][Sit]1/2"],
                [HP_2011_ds62.mgts(), "[Alm][Mgm][Alt1/2Sit1/2]1/2"],
            ],
            energy_interaction=[
                [7000.0, 4000.0, 13000.0],
                [4000.0, 13000.0],
                [17000.0],
            ],
            volume_interaction=[[0.0, 0.0, -1.5e-06], [0.0, -1.5e-06], [-1.5e-06]],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class sa(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "sa"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.spr4(), "[Mgm][Mgm]3[Sit]"],
                [HP_2011_ds62.spr5(), "[Alm][Mgm]3[Alt]"],
                [fspm, "[Fem][Fem]3[Sit]"],
                [spro, "[Mgm][Fem]3[Sit]"],
            ],
            energy_interaction=[
                [10000.0, 16000.0, 12000.0],
                [19000.0, 22000.0],
                [4000.0],
            ],
            volume_interaction=[
                [-2.0000000000000002e-07, 0.0, 0.0],
                [-2.0000000000000002e-07, -2.0000000000000002e-07],
                [0.0],
            ],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class cd(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "cd"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.crd(), "[Mgx]2[Vh]"],
                [HP_2011_ds62.fcrd(), "[Fex]2[Vh]"],
                [HP_2011_ds62.hcrd(), "[Mgx]2[Hoh]"],
            ],
            energy_interaction=[[8000.0, 0.0], [9000.0]],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class st(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "st"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [mstm, "[Mgx]4"],
                [HP_2011_ds62.fst(), "[Fex]4"],
            ],
            energy_interaction=[[16000.0]],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class chl(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "chl"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.clin(), "[Mgm][Mgm]4[Alm][Sit1/2Alt1/2]2"],
                [HP_2011_ds62.afchl(), "[Mgm][Mgm]4[Mgm][Sit]2"],
                [HP_2011_ds62.ames(), "[Alm][Mgm]4[Alm][Alt]2"],
                [HP_2011_ds62.daph(), "[Fem][Fem]4[Alm][Sit1/2Alt1/2]2"],
                [ochl1, "[Mgm][Fem]4[Fem][Sit]2"],
                [ochl4, "[Fem][Mgm]4[Mgm][Sit]2"],
            ],
            energy_interaction=[
                [17000.0, 17000.0, 20000.0, 30000.0, 21000.0],
                [16000.0, 37000.0, 20000.0, 4000.0],
                [30000.0, 29000.0, 13000.0],
                [18000.0, 33000.0],
                [24000.0],
            ],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class ctd(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "ctd"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.mctd(), "[Mgmb]"],
                [HP_2011_ds62.fctd(), "[Femb]"],
            ],
            energy_interaction=[[4000.0]],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)


class sp1(Solution):
    def __init__(self, molar_fractions=None):
        self.name = "sp1"
        self.solution_model = SymmetricRegularSolution(
            endmembers=[
                [HP_2011_ds62.herc(), "[Fea]"],
                [HP_2011_ds62.sp(), "[Mga]"],
            ],
            energy_interaction=[[0.0]],
        )
        Solution.__init__(self, molar_fractions=molar_fractions)
