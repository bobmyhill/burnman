import numpy as np


def sdr2cmt(moment, s, d, r):
    strike = s / 180. * np.pi
    dip = d / 180. * np.pi
    rake = r / 180. * np.pi
    
    mt = np.array([np.sin(2. * dip) * np.sin(rake),
                   -(np.sin(dip) * np.cos(rake) * np.sin(2. * strike)
                     + np.sin(2. * dip) * np.sin(rake) * np.power(np.sin(strike), 2.)),
                   np.sin(dip) * np.cos(rake) * np.sin(2. * strike)
                   - np.sin(2. * dip) * np.sin(rake) * np.power(np.cos(strike), 2.),
                   -(np.cos(dip) * np.cos(rake) * np.cos(strike)
                     + np.cos(2. * dip) * np.sin(rake) * np.sin(strike)),
                   np.cos(dip) * np.cos(rake) * np.sin(strike)
                   - np.cos(2. * dip) * np.sin(rake) * np.cos(strike),
                   -(np.sin(dip) * np.cos(rake) * np.cos(2. * strike)
                     + 0.5 * np.sin(2. * dip) * np.sin(rake) * np.sin(2. * strike))])
    
    return moment*mt

'''
print sdr2cmt(1.98e25, 206., 18., 78.)
print sdr2cmt(1.98e25, 39., 73., 94.)

print sdr2cmt(1.58e26, 233., 38., -96.)
print sdr2cmt(1.58e26, 61., 53., -85.)


print sdr2cmt(2.04e27, 254., 73., -10.)
print sdr2cmt(2.04e27, 347., 80., -162.)

print sdr2cmt(1.18e25, 357., 6., 140.)
print sdr2cmt(1.18e25, 128., 86., 85.)
'''


print sdr2cmt(100., 5., 45., -90.)
