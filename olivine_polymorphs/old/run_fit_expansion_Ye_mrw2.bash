#!/bin/bash

base=mrw2_thermal_expansion

psbasemap -JX15/15 -R0/1800/3.9/4.2 -B200:"Temperature (K)":/0.05:"Volume (kJ/kbar)":SWen -K > ${base}.ps
awk '{print $1, $2*nA/voltoa/Z, $3*nA/voltoa/Z}' nA="6.02214e23" voltoa="1e25" Z="8" Ye_mrw2_expansion_TVsig.dat | awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' | psxy -J -R -O -K -EY0 -Sc0.1c -Gblack >> ${base}.ps

awk '$1<650 {print $1, $2*nA/voltoa/Z, $3*nA/voltoa/Z}' nA="6.02214e23" voltoa="1e25" Z="8" Ye_mrw2_expansion_TVsig.dat | awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' | psxy -J -R -O -K -EY0 -Sc0.1c -Gred >> ${base}.ps

mv Ye_mrw2_expansion_TVsig.dat save.tmp
awk '$1<650' save.tmp > Ye_mrw2_expansion_TVsig.dat
./fit_expansion_Ye_mrw2.py  | grep -v "\[" | head -2
./fit_expansion_Ye_mrw2.py | grep -v "\[" | grep -v ":" | awk '{print $1, $2}' | sort -k1 -n | psxy -J -R -O -Wdefault,red >> ${base}.ps

mv save.tmp Ye_mrw2_expansion_TVsig.dat
gv ${base}.ps &
