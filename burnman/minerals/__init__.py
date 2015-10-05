# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
Mineral database

  - :mod:`~burnman.minerals.SLB_2005`
  - :mod:`~burnman.minerals.SLB_2011_ZSB_2013`
  - :mod:`~burnman.minerals.SLB_2011`
  - :mod:`~burnman.minerals.Murakami_etal_2012`
  - :mod:`~burnman.minerals.Murakami_2013`
  - :mod:`~burnman.minerals.Matas_etal_2007`
  - :mod:`~burnman.minerals.HP_2011_ds62`
  - :mod:`~burnman.minerals.HP_2011_fluids`
  - :mod:`~burnman.minerals.HHPH_2013`
  - :mod:`~burnman.minerals.other`
"""

# Stixrude and Lithgow-Bertelloni
import SLB_2011
import SLB_2011_ZSB_2013
import SLB_2005

# de Koker and Stixrude
import DKS_2013_liquids
import DKS_2013_solids
import DKS_2008_fo

# Murakami and coworkers
import Murakami_etal_2012
import Murakami_2013

# Matas and coworkers
import Matas_etal_2007

# Komabayashi 
import Komabayashi_2014

# Holland, Powell and coworkers
import HP_2011_ds62
import HP_2011_fluids
import HHPH_2013

# Other
import other

import Fe_Si_O
import Myhill_calibration_iron
import Myhill_silicate_liquid
import DKS_2013_liquids
import Komabayashi_2014
