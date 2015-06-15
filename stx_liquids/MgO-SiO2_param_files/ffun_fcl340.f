      bf(1)  = 1.     ! 0 0

      bf(2)  = fstr    ! 1 0  volume functional (finite strain)
      bf(3)  = theta   ! 0 1  temperature functional

      bf(4)  = (1./2.)*(fstr**2.)   ! 2 0
      bf(5)  = (fstr)*(theta)       ! 1 1
      bf(6)  = (1./2.)*(theta**2.)  ! 0 2

      bf(7)  = (1./6.)*(fstr**3.)           ! 3 0
      bf(8)  = (1./2.)*(fstr**2.)*(theta)   ! 2 1
      bf(9)  = (1./2.)*(fstr)*(theta**2.)   ! 1 2
      bf(10) = (1./6.)*(theta**3.)          ! 0 3

      bf(11) = (1./24.)*(fstr**4.)               ! 4 0
      bf(12) = (1./ 6.)*(fstr**3.)*(theta)       ! 3 1
      bf(13) = (1./ 4.)*(fstr**2.)*(theta**2.)   ! 2 2
      bf(14) = (1./ 6.)*(fstr)*(theta**3.)       ! 1 3
      bf(15) = (1./24.)*(theta**4.)              ! 0 4

      bf(16) = (1./120.)*(fstr**5.)               ! 5 0
      bf(17) = (1./ 24.)*(fstr**4.)*(theta)       ! 4 1
      bf(18) = (1./ 12.)*(fstr**3.)*(theta**2.)   ! 3 2
      bf(19) = (1./ 12.)*(fstr**2.)*(theta**3.)   ! 2 3
      bf(20) = (1./ 24.)*(fstr)*(theta**4.)       ! 1 4
      bf(21) = (1./120.)*(theta**5.)              ! 0 5

      bf(22) = (1./720.)*(fstr**6.)               ! 6 0
      bf(23) = (1./120.)*(fstr**5.)*(theta)       ! 5 1
      bf(24) = (1./ 48.)*(fstr**4.)*(theta**2.)   ! 4 2
      bf(25) = (1./ 36.)*(fstr**3.)*(theta**3.)   ! 3 3
      bf(26) = (1./ 48.)*(fstr**2.)*(theta**4.)   ! 2 4
      bf(27) = (1./120.)*(fstr)*(theta**5.)       ! 1 5
      bf(28) = (1./720.)*(theta**6.)              ! 0 6

      bf(29) = (1./5040.)*(fstr**7.)               ! 7 0
      bf(30) = (1./ 720.)*(fstr**6.)*(theta)       ! 6 1
      bf(31) = (1./ 240.)*(fstr**5.)*(theta**2.)   ! 5 2
      bf(32) = (1./ 144.)*(fstr**4.)*(theta**3.)   ! 4 3
      bf(33) = (1./ 144.)*(fstr**3.)*(theta**4.)   ! 3 4
      bf(34) = (1./ 240.)*(fstr**2.)*(theta**5.)   ! 2 5
      bf(35) = (1./ 720.)*(fstr)*(theta**6.)       ! 1 6
      bf(36) = (1./5040.)*(theta**7.)              ! 0 7


      ffun_fcl340 = 0.
      do i = 1,nepfit300(1)
        ffun_fcl340 = ffun_fcl340 + aepfit300(1,i)*bf(i)
      enddo
