import numpy as np

R = 8.31446  # gas constant


Tm_fraction_pure_H2O_melt = 0.31
W = -91000.

ol = {'E': -16982.,
      'S': 4.6,
      'V': 1.89e-6,
      'hyE': -1272564.8,
      'hyS': 208,
      'hyV': 2.8e-05,
      'feE': 0.,  # NEED TO CHANGE
      'feS': 0.,  # NEED TO CHANGE
      'feV': 0.}  # NEED TO CHANGE

wad = {'E': 0.,
       'S': 0.,
       'V': 0.,
       'hyE': -1306930.55,
       'hyS': 203,
       'hyV': 2.8e-05,
       'feE': 0.,  # NEED TO CHANGE
       'feS': 0.,  # NEED TO CHANGE
       'feV': 0.}  # NEED TO CHANGE

ring = {'E': 7060.,
        'S': -4.8,
        'V': -8.12e-7,
        'hyE': -1304564.8,
        'hyS': 208,
        'hyV': 2.8e-05,
        'feE': 0.,  # NEED TO CHANGE
        'feS': 0.,  # NEED TO CHANGE
        'feV': 0.}  # NEED TO CHANGE

lm = {'E': 77858.,
      'S': -0.645,
      'V': -3.53e-6,
      'hyE': -1272564.8,  # NEED TO CHANGE
      'hyS': 208,  # NEED TO CHANGE
      'hyV': 2.8e-05,  # NEED TO CHANGE
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
         'halfPint0': 1.e9,  # NEED TO CHANGE
         'dhalfPintdT': 0.}  # NEED TO CHANGE

wadring = {'Delta_E': wad['E'] - ring['E'],
           'Delta_S': wad['S'] - ring['S'],
           'Delta_V': wad['V'] - ring['V'],
           'halfPint0': 1.e9,  # NEED TO CHANGE
           'dhalfPintdT': 0.}  # NEED TO CHANGE

ringlm = {'Delta_E': ring['E'] - lm['E'],
          'Delta_S': ring['S'] - lm['S'],
          'Delta_V': ring['V'] - lm['V'],
          'halfPint0': 1.e9,  # NEED TO CHANGE
          'dhalfPintdT': 0.}  # NEED TO CHANGE
