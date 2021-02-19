R = 8.31446  # gas constant

liq_sp = {'H_poly': -94000.,
          'S_poly': -75.,
          'r': 4.}  # 4 oxygens per Mg2SiO4

ol = {'E': -16982.,
      'S': 4.6,
      'V': 1.89e-6,
      'hyE': 74000.,
      'hyS': 15.,
      'hyV': -2.e-6,
      'feE': 0.,  # NEED TO CHANGE
      'feS': 0.,  # NEED TO CHANGE
      'feV': 0.}  # NEED TO CHANGE

wad = {'E': 0.,
       'S': 0.,
       'V': 0.,
       'hyE': 34500.,
       'hyS': 15.,
       'hyV': 0.,
       'feE': 0.,  # NEED TO CHANGE
       'feS': 0.,  # NEED TO CHANGE
       'feV': 0.}  # NEED TO CHANGE

ring = {'E': 7060.,
        'S': -4.8,
        'V': -8.12e-7,
        'hyE': 32500.,
        'hyS': 15.,
        'hyV': 0.,
        'feE': 0.,  # NEED TO CHANGE
        'feS': 0.,  # NEED TO CHANGE
        'feV': 0.}  # NEED TO CHANGE

lm = {'E': 77858.,
      'S': -0.645,
      'V': -3.53e-6,
      'hyE': 100000.,  # NEED TO CHANGE
      'hyS': -12,  # NEED TO CHANGE
      'hyV': 0.,  # NEED TO CHANGE
      'feE': 0.,  # NEED TO CHANGE
      'feS': 0.,  # NEED TO CHANGE
      'feV': 0.}  # NEED TO CHANGE

melt = {'E': 175553.,
        'S': 100.3,
        'V': -3.277e-6,
        'a': 2.60339693e-06,
        'b': 2.64753089e-11,
        'c': 1.18703511e+00}


olwad = {'Delta_E': ol['E'] - wad['E'],
         'Delta_S': ol['S'] - wad['S'],
         'Delta_V': ol['V'] - wad['V'],
         'halfPint0': 0.976e9,  # estimated from SLB2011 Figure
         'dhalfPintdT': -4.05e5} # estimated from SLB2011 Figure

wadring = {'Delta_E': wad['E'] - ring['E'],
           'Delta_S': wad['S'] - ring['S'],
           'Delta_V': wad['V'] - ring['V'],
           'halfPint0': 2.286e9,   # estimated from SLB2011 Figure
           'dhalfPintdT': -8.85e5} # estimated from SLB2011 Figure

ringlm = {'Delta_E': ring['E'] - lm['E'],
          'Delta_S': ring['S'] - lm['S'],
          'Delta_V': ring['V'] - lm['V'],
          'halfPint0': 0.1e9,  # Something small, could increase this to simulate garnet decomposition.
          'dhalfPintdT': 0.}  # 0.
