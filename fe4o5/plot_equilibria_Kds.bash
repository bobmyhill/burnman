#!/bin/bash

base="equilibria_Kds"

psbasemap -JX10/10 -R0/0.5/0/0.25 -B0.1f0.05:"X@-Mg@- Fe@-4@-O@-5@-":/0.05f0.01:"Fe/Mg Kd Fe@-2@-SiO@-4@-/Fe@-4@-O@-5@-":SWen -Sc0.1c -P -K > ${base}.ps 


awk '$3=="ol" {print $6/2, ((1-$10)/$10)/($8/$6)}' phasePTX.dat | psxy -J -R -O -K -Sc0.2c -W1,black >> ${base}.ps 

awk '$3=="wad" {print $6/2, ((1-$10)/$10)/($8/$6)}' phasePTX.dat | psxy -J -R -O -K -S+0.2c -W1,blue >> ${base}.ps 

awk '$3=="rw" {print $6/2, ((1-$10)/$10)/($8/$6)}' phasePTX.dat | psxy -J -R -O -K -Sx0.2c -W1,red >> ${base}.ps 

echo 0.035 0.148 10.5 | pstext -J -R -O -K -F+f10,4,red+jLM >> ${base}.ps
echo 0.118 0.165 10.5 | pstext -J -R -O -K -F+f10,4,red+jLM >> ${base}.ps
echo 0.092 0.108 10.5 | pstext -J -R -O -K -F+f10,4,red+jLM >> ${base}.ps
echo 0.092 0.100 14   | pstext -J -R -O -K -F+f10,4,red+jLM >> ${base}.ps

echo "0.25 0.075 12,14,16" | pstext -J -R -O -K -F+f10,4,blue+jLM >> ${base}.ps
echo 0.19 0.055 14   | pstext -J -R -O -K -F+f10,4,blue+jLM >> ${base}.ps

echo 0.10 0.045 9 | pstext -J -R -O -K -F+f10,4,black+jLM >> ${base}.ps
echo 0.225 0.035 10.5   | pstext -J -R -O -K -F+f10,4,black+jLM >> ${base}.ps

awk '{print $1, $2}' Kds_10GPa.dat | psxy -J -R -O -K -W1,black >> ${base}.ps 
awk '{print $1, $3}' Kds_10GPa.dat | psxy -J -R -O -K -W1,blue >> ${base}.ps 
awk '{print $1, $4}' Kds_10GPa.dat | psxy -J -R -O -K -W1,red >> ${base}.ps 

awk '{print $1, $2}' Kds_15GPa.dat | psxy -J -R -O -K -W1,black,- >> ${base}.ps 
awk '{print $1, $3}' Kds_15GPa.dat | psxy -J -R -O -K -W1,blue,- >> ${base}.ps 
awk '{print $1, $4}' Kds_15GPa.dat | psxy -J -R -O -K -W1,red,- >> ${base}.ps 


pslegend -J -R -O -Dx6/10/0/0/TL -C0.1i/0.1i -L1.2 << EOF >> ${base}.ps
N 1
S 0.1i c 0.2c - 1,black 0.6c Olivine
S 0.1i + 0.2c - 1,blue 0.6c Wadsleyite
S 0.1i x 0.2c - 1,red 0.6c Ringwoodite
S 0.1i - 0.5c - 1,black 0.6c Model, 10 GPa
S 0.1i - 0.5c - 1,black,- 0.6c Model, 15 GPa
EOF


ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

convert -density 600 ${base}.pdf ${base}.jpg
evince ${base}.pdf
