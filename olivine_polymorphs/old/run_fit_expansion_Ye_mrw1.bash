#!/bin/bash

base=mrw_thermal_expansion

psbasemap -JX15/15 -R0/1800/3.9/4.2 -B200:"Temperature (K)":/0.05:"Volume (kJ/kbar)":SWen -K > ${base}.ps
awk '{print $1, $2*nA/voltoa/Z, $3*nA/voltoa/Z}' nA="6.02214e23" voltoa="1e25" Z="8" Ye_mrw1_expansion_TVsig.dat | awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' | psxy -J -R -O -K -EY0 -Sc0.1c -Gblack >> ${base}.ps
cp Ye_mrw1_expansion_TVsig.dat saved.tmp
awk '$1<700' saved.tmp > Ye_mrw1_expansion_TVsig.dat
awk '{print $1, $2*nA/voltoa/Z, $3*nA/voltoa/Z}' nA="6.02214e23" voltoa="1e25" Z="8" Ye_mrw1_expansion_TVsig.dat | awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' | psxy -J -R -O -K -EY0 -Sc0.1c -Gred >> ${base}.ps

./fit_expansion_Ye_mrw1.py  | grep -v "\[" | head -3
./fit_expansion_Ye_mrw1.py | grep -v "\[" | grep -v ":" | awk '{print $1, $2}'  | psxy -J -R -O -Wdefault,red >> ${base}.ps

mv saved.tmp Ye_mrw1_expansion_TVsig.dat
gv ${base}.ps &
