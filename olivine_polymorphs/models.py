import burnman
from burnman.minerals import SLB_2011, HHPH_2013, HP_2011_ds62
from burnman.solidsolution import SolidSolution

'''
fo=HP_2011_ds62.fo()
mwd=HP_2011_ds62.mwd()
mrw=HP_2011_ds62.mrw()
fa=HP_2011_ds62.fa()
fwd=HP_2011_ds62.fwd()
frw=HP_2011_ds62.frw()
'''
'''
fo=HHPH_2013.fo()
mwd=HHPH_2013.mwd()
mrw=HHPH_2013.mrw()
fa=HHPH_2013.fa()
fwd=HHPH_2013.fwd()
frw=HHPH_2013.frw()
'''

fo=SLB_2011.forsterite()
mwd=SLB_2011.mgwa()
mrw=SLB_2011.mgri()
fa=SLB_2011.fayalite()
fwd=SLB_2011.fewa()
frw=SLB_2011.feri()

mwd.params['F_0'] += 0.8e3
mrw.params['F_0'] += 0.8e3
fwd.params['F_0'] += 1.5e3
frw.params['F_0'] -= 1.5e3

class mg_fe_olivine(SolidSolution):
    def __init__(self):
        # Name
        self.name='olivine'
        self.endmembers = [[fo, '[Mg]2SiO4'],[fa, '[Fe]2SiO4']]
        self.energy_interaction=[[4.000e3]]
        self.volume_interaction=[[1.e-7]]
        self.solution_type='symmetric'
        burnman.SolidSolution.__init__(self)

class mg_fe_wadsleyite(SolidSolution):
    def __init__(self):
        # Name
        self.name='wadsleyite'
        self.endmembers = [[mwd, '[Mg]2SiO4'],[fwd, '[Fe]2SiO4']]
        self.energy_interaction=[[15.000e3]]
        self.volume_interaction=[[0.e-7]]
        self.alphas=[1.0, 1.0]
        self.solution_type='asymmetric'
        burnman.SolidSolution.__init__(self)

class mg_fe_ringwoodite(SolidSolution):
    def __init__(self):
        # Name
        self.name='ringwoodite'
        self.endmembers = [[mrw, '[Mg]2SiO4'],[frw, '[Fe]2SiO4']]
        self.energy_interaction=[[8.320e3]]
        self.alphas=[1.0, 1.0]
        self.volume_interaction=[[0.e-7]]
        self.solution_type='asymmetric'
        burnman.SolidSolution.__init__(self)


class mg_fe_ringwoodite_asymmetric(SolidSolution):
    def __init__(self):
        # Name
        self.name='ringwoodite'
        self.endmembers = [[mrw, '[Mg]2SiO4'],[frw, '[Fe]2SiO4']]
        self.energy_interaction=[[7.e3]]
        self.alphas=[2.0, 1.0]
        self.volume_interaction=[[0.e-7]]
        self.solution_type='asymmetric'
        burnman.SolidSolution.__init__(self)

class mg_fe_wadsleyite_asymmetric(SolidSolution):
    def __init__(self):
        # Name
        self.name='wadsleyite'
        self.endmembers = [[mwd, '[Mg]2SiO4'],[fwd, '[Fe]2SiO4']]
        self.energy_interaction=[[13.50e3]]
        self.volume_interaction=[[0.e-7]]
        self.alphas=[1.0, 1.0]
        self.solution_type='asymmetric'
        burnman.SolidSolution.__init__(self)
