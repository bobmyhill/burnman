#!/bin/bash

base=frw_thermal_expansion

psbasemap -JX15/15 -R0/2000/4.15/4.45 -B200:"Temperature (K)":/0.05:"Volume (kJ/kbar)":SWen -K > ${base}.ps

V8=`fit_expansion_Mao_frw.py | grep "V8:" | tail -1 | awk '{print $2}'`
awk '{print $1, V8+$2*V8, $3*V8}' V8=${V8} Mao_et_al_alpha_Fe2SiO4.dat | awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' | psxy -J -R -O -K -EY0 -Sc0.1c -Gred >> ${base}.ps

./fit_expansion_Mao_frw.py  | grep -v "\[" | head -8
./fit_expansion_Mao_frw.py | grep -v "\[" | grep -v ":" | awk '{print $1, $2}'  | psxy -J -R -O -Wdefault,red >> ${base}.ps

gv ${base}.ps &
