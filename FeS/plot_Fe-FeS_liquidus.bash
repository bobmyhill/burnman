#!/bin/bash

base='Fe-FeS_liquidus'

psbasemap -JX10 -R0/40/673/2473 -K -P -B10f5:"S (wt%)":/400f100:"T (K)":SWen > ${base}.ps

awk 'NR>1 {print $4, $2, $4-$5, $4, $4, $4+$5, $2-$3, $2, $2, $2+$3}' data/eutectic_PTX.dat | psxy -J -R -O -K -EXY0 -Sc0.1c -Gblue >> ${base}.ps

awk '{print $1, $2}' data/Fe-FeS_1bar.dat | psxy -J -R -O -K -W1,0/0/0 >> ${base}.ps
awk '{print $1, $2}' data/Fe-FeS_10GPa.dat | psxy -J -R -O -K -W1,100/0/0 >> ${base}.ps
awk '{print $1, $2}' data/Fe-FeS_14GPa.dat | psxy -J -R -O -K -W1,200/0/0 >> ${base}.ps
awk '{print $1, $2}' data/Fe-FeS_21GPa.dat | psxy -J -R -O -K -W1,255/0/0 >> ${base}.ps

echo 1 1 | awk '{print 100*$2*32.065/($1*55.845 + $2*32.065)}' | awk '{printf "%f 673\n%f 2473", $1, $1}' | psxy -J -R -O -K -W1,green,- >> ${base}.ps
echo 3 2 | awk '{print 100*$2*32.065/($1*55.845 + $2*32.065)}' | awk '{printf "%f 673\n%f 2473", $1, $1}' | psxy -J -R -O -K -W1,green,- >> ${base}.ps
echo 2 1 | awk '{print 100*$2*32.065/($1*55.845 + $2*32.065)}' | awk '{printf "%f 673\n%f 2473", $1, $1}' | psxy -J -R -O -K -W1,green,- >> ${base}.ps
echo 3 1 | awk '{print 100*$2*32.065/($1*55.845 + $2*32.065)}' | awk '{printf "%f 673\n%f 2473", $1, $1}' | psxy -J -R -O -K -W1,green,- >> ${base}.ps

awk '{print $1, $2}' data/rough_eutectic.dat | psxy -J -R -O -W0.5,blue,- >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

evince ${base}.pdf