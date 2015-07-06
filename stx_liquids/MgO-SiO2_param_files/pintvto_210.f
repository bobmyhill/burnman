c
c  Nico de Koker, Sept 2007
c
c  Evaluate the integral fo P(V,To) from V1 to V2
c
c
c23456789112345678921234567893123456789412345678951234567896123456789712
c23456789112345678921234567893123456789412345678951234567896123456789712


      function pintvto_210(vol1,vol2)


c23456789112345678921234567893123456789412345678951234567896123456789712

      implicit none

      include 'passfcl200.h'

c23456789112345678921234567893123456789412345678951234567896123456789712

      real pintvto_210
      real vol1,vol2

      real p1, p2, fstr

      integer i

c23456789112345678921234567893123456789412345678951234567896123456789712


      fstr = (1./fsn201(1)) * ( (Vref200(1)/vol1)**(fsn201(1)/3.) - 1. )

      p1 = 0.
      do i = 2,npfit200(1)+1
        p1 = p1 + (1./real(i))*pfit200(1,i-1)*fstr**(i)
      enddo

      fstr = (1./fsn201(1))*( (Vref200(1)/vol2)**(fsn201(1)/3.) - 1. )
      p2 = 0.
      do i = 2,npfit200(1)+1
        p2 = p2 + (1./real(i))*pfit200(1,i-1)*fstr**(i)
      enddo

      pintvto_210 = -(p2 - p1)


c23456789112345678921234567893123456789412345678951234567896123456789712

      return
      end

c23456789112345678921234567893123456789412345678951234567896123456789712
c23456789112345678921234567893123456789412345678951234567896123456789712