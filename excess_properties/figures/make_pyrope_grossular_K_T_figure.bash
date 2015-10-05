#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="pyrope_grossular_K_0"

psbasemap -JX12/8 -R-0.01/1.01/140/180 -Ba1f0.5:"p@-py@-":/a10f5:"Bulk modulus (GPa)":SWen -K -P > ${base}.ps

awk '{print $1, $2}' data/pyrope_grossular_K_0.dat | psxy -J -R -O -K -W0.5,red >> ${base}.ps
awk '{print $1, $3}' data/pyrope_grossular_K_0.dat | psxy -J -R -O -K -W0.5,red,- >> ${base}.ps
awk '{print $1, $4}' data/pyrope_grossular_K_0.dat |  psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '{print $1, $5}' data/pyrope_grossular_K_0.dat |  psxy -J -R -O -K -W0.5,black,. >> ${base}.ps

awk '{print $1, $2, $2-$3, $2, $2, $2+$3}' data/pyrope_grossular_K_0_obs.dat | psxy -J -R -O -K -EY0 -Sc0.1c -Gred -W0.5,black >> ${base}.ps

printf "0.01 148 \n0.03 148" | psxy -J -R -O -K -W0.5,black,. >> ${base}.ps
printf "0.01 146 \n0.03 146" | psxy -J -R -O -K -W0.5,black >> ${base}.ps
printf "0.01 144 \n0.03 144" | psxy -J -R -O -K -W0.5,red >> ${base}.ps
printf "0.01 142 \n0.03 142" | psxy -J -R -O -K -W0.5,red,- >> ${base}.ps

echo "0.05 148 Ganguly et al. (1996; constant V@+xs@+)" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "0.05 146 Ganguly et al. (1996; with K@-T@-@+xs@+)" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "0.05 144 Du et al. (2015; py@-40@-, py@-60@-)" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "0.05 142 Du et al. (2015; py@-20@-, py@-80@-)" | pstext -J -R -O -F+jLM >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi

evince ${base}.pdf &

