#!/bin/bash

base=fo_thermal_expansion

psbasemap -JX15/15 -R0/1800/4.3/4.7 -B200:"Temperature (K)":/0.05:"Volume (kJ/kbar)":SWen -K > ${base}.ps
awk '{print $1, $2*nA/voltoa/Z, $3*nA/voltoa/Z}' nA="6.02214e23" voltoa="1e25" Z="4" Ye_fo_expansion_TVsig.dat | awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' | psxy -J -R -O -K -EY0 -Sc0.1c -Gred >> ${base}.ps

./fit_expansion_Ye_fo.py  | grep -v "\[" | head -2
./fit_expansion_Ye_fo.py | grep -v "\[" | grep -v ":" | awk '{print $1, $2}' | sort -k1 -n | psxy -J -R -O -Wdefault,red >> ${base}.ps

gv ${base}.ps &
