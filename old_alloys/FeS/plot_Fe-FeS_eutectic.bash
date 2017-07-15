#!/bin/bash

base='Fe-FeS_liquidus'

psbasemap -JX10 -R0/36.47/673/2473 -K -P -B10f5:"S (wt%)":/400f100:"T (K)":SWen > ${base}.ps

awk 'NR>1 {print $3, $4}' data/eutectic_PTX.dat | psxy -J -R -O -K -W1,0/0/0 >> ${base}.ps
awk '{print $1, $2}' data/Fe-FeS_10GPa.dat | psxy -J -R -O -K -W1,100/0/0 >> ${base}.ps
awk '{print $1, $2}' data/Fe-FeS_14GPa.dat | psxy -J -R -O -K -W1,200/0/0 >> ${base}.ps
awk '{print $1, $2}' data/Fe-FeS_21GPa.dat | psxy -J -R -O -K -W1,255/0/0 >> ${base}.ps

echo 0 0 | psxy -J -R -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

evince ${base}.pdf