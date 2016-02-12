import numpy as np

K_0 = 100.e9
Kprime_0 = 4.
Kdprime_0 = -Kprime_0/K_0

print 2*K_0*Kdprime_0
print Kprime_0*Kprime_0

sqrt = np.sqrt(Kprime_0*Kprime_0 - 2.*K_0*Kdprime_0)

plus = Kprime_0 + sqrt
minus = Kprime_0 - sqrt

print plus/minus

p = -20.83333333333e9
p = -22.5e9
print np.power((np.abs( (Kdprime_0*p + minus ) / (Kdprime_0*p + plus )) \
                    * np.abs( plus / minus )), -1./sqrt)

a = (1.+Kprime_0) / ( 1. + Kprime_0 + K_0*Kdprime_0 )
b = ( (Kprime_0/K_0) - Kdprime_0/(Kprime_0+1.))
c = ( 1. + Kprime_0 + K_0*Kdprime_0 ) / ( Kprime_0*Kprime_0 + Kprime_0 - K_0*Kdprime_0)
print 1. - a*(1. - np.power((1.+b*p), -c))
