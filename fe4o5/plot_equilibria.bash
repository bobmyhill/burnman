#!/bin/bash

base="equilibria"

psbasemap -JX10/10 -R0/1/0/1 -B0.2f0.1:"X@-Mg@- observed":/0.2f0.1:"X@-Mg@- calculated":SWen -Sc0.1c -P -K > ${base}.ps 

awk '$2=="ol" {print $6, $5}' ${base}.dat | psxy -J -R -O -K -Sc0.2c >> ${base}.ps 

awk '$2=="wad" {print $6, $5}' ${base}.dat | psxy -J -R -O -K -S+0.2c >> ${base}.ps 

awk '$2=="rw" {print $6, $5}' ${base}.dat | psxy -J -R -O -K -Sx0.2c >> ${base}.ps 

pslegend -J -R -O -K -Dx0/10/0/0/TL -C0.1i/0.1i -L1.2 << EOF >> ${base}.ps
N 1
S 0.1i c 0.2c - 0.25p 0.6c Olivine
S 0.1i + 0.2c - 0.25p 0.6c Wadsleyite
S 0.1i x 0.2c - 0.25p 0.6c Ringwoodite
EOF

printf "0 0 \n 1 1" | psxy -J -R -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps 
mv ${base}.epsi ${base}.eps

convert -density 600 ${base}.pdf ${base}.jpg
evince ${base}.pdf
