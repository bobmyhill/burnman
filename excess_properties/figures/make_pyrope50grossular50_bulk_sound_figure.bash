#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="pyrope_grossular_bulk_sound_velocities"

psbasemap -JX12/8 -R0/25/6/9 -Ba5f1:"Pressure (GPa)":/a0.5f0.1:"Bulk sound velocity (km/s)":SWen -K -P > ${base}.ps

awk '{print $1, $3}' data/pyrope_grossular_bulk_sound_velocities.dat | psxy -J -R -O -K -W1,grey >> ${base}.ps

awk '{print $1, $2}' data/pyrope_grossular_bulk_sound_velocities.dat | psxy -J -R -O -K -W1,black >> ${base}.ps

awk '{print $1, $4}' data/pyrope_grossular_bulk_sound_velocities.dat | psxy -J -R -O -K -W1,black,- >> ${base}.ps



printf "1 8.8 \n 2.5 8.8" | psxy -J -R -O -K -W1,black >> ${base}.ps
echo "3 8.8 This study" | pstext -J -R -O -K -F+jLM >> ${base}.ps

printf "1 8.6 \n 2.5 8.6" | psxy -J -R -O -K -W1,black,- >> ${base}.ps
echo "3 8.6 Constant excess volume" | pstext -J -R -O -K -F+jLM >> ${base}.ps

printf "1 8.4 \n 2.5 8.4" | psxy -J -R -O -K -W1,grey >> ${base}.ps
echo "3 8.4 Ganguly et al. (1996)" | pstext -J -R -O -K -F+jLM >> ${base}.ps


echo "24 6.2 298.15 K" | pstext -J -R -O -F+jRM >> ${base}.ps
ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi

evince ${base}.pdf &