#!/bin/bash

base=wad_thermal_expansion

psbasemap -JX15/15 -R0/1800/3.9/4.2 -B200:"Temperature (K)":/0.05:"Volume (kJ/kbar)":SWen -K > ${base}.ps
awk '{print $1, $2*nA/voltoa/Z, $3*nA/voltoa/Z}' nA="6.02214e23" voltoa="1e25" Z="8" Trots_wad_expansion_TVsig.dat | awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' | psxy -J -R -O -K -EY0 -Sc0.1c -Gred >> ${base}.ps

./fit_expansion_Trots_wad.py  | grep -v "\[" | head -2
./fit_expansion_Trots_wad.py | grep -v "\[" | grep -v ":" | awk '{print $1, $2}' | sort -k1 -n | psxy -J -R -O -Wdefault,red >> ${base}.ps

gv ${base}.ps &
