      read(inpfid,*)
      read(inpfid,*)
      read(inpfid,*) aperf
      read(inpfid,*) formmass
      read(inpfid,*) nspec
      read(inpfid,*) (nspeci(i), i=1,nspec)
      read(inpfid,*) (zspeci(i), i=1,nspec)
      read(inpfid,*) (atmass(i), i=1,nspec)
      read(inpfid,*) figfm, fclfm, felfm

      if (fclfm.eq.220) then
      read(inpfid,*)
      read(inpfid,*) Vref200(1), Tref200(1), Eref200(1), Sref200(1)
      read(inpfid,*) fsn201(1)
      read(inpfid,*) npfit200(1), (pfit200(1,i),i=1,npfit200(1))
      read(inpfid,*) (akfit200(1,i),i=1,2)
      read(inpfid,*) cvfit200(1,1)

      elseif (fclfm.eq.340) then

      read(inpfid,*)
      read(inpfid,*) Vref300(1), Tref300(1), Eref300(1), Sref300(1)
      read(inpfid,*) vffm300(1)
      read(inpfid,*) fsn301(1)
      read(inpfid,*) tffm300(1)
      read(inpfid,*) thn301(1)
      read(inpfid,*) nepfit300(1), (aepfit300(1,i),i=1,nepfit300(1))

      endif

      if (felfm.eq.200) then
        read(inpfid,*)
        read(inpfid,*) 
     &        elbeto(1), elbetpow(1), elTeo(1), elTepow(1), elVo(1)
      endif

      if (figfm.eq.1) then
        read(inpfid,*)
        read(inpfid,*) figtfm, figrfm, figefm
        if (figefm.ne.0) then
          do i=1,nspec
            read(inpfid,*) naes(i)
            read(inpfid,*) (jel(i,j),j=1,naes(i))
            read(inpfid,*) (eel(i,j),j=1,naes(i))
          enddo
        endif
      endif
