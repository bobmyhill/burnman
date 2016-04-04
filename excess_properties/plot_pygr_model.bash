#!/bin/bash

gmtset FONT_ANNOT_PRIMARY		= 10p,Times-Roman,black
gmtset FONT_ANNOT_SECONDARY		= 14p,Helvetica,black
gmtset FONT_LABEL			= 14p,Times-Roman,black
base="VSxs_pygr"
psbasemap -JX6/8 -R0/25/0/0.5 -B5f1:"Pressure (GPa)":/0.1f0.05:"Excess volume (cm@+3@+/mol)":SWen -K -P > ${base}.ps
psxy Vxs_300.0_K.dat -J -R -O -K -W1,black >> ${base}.ps
psxy Vxs_1000.0_K.dat -J -R -O -K -W1,black >> ${base}.ps
psxy Vxs_2000.0_K.dat -J -R -O -K -W1,black >> ${base}.ps

echo "0.5 0.21 300 K" | pstext -J -R -O -K -F+f12,4,black+jLB+a-32 >> ${base}.ps
echo "0.3 0.33 1000 K" | pstext -J -R -O -K -F+f12,4,black+jLB+a-45 >> ${base}.ps
echo "0.3 0.45 2000 K" | pstext -J -R -O -K -F+f12,4,black+jLB+a-60 >> ${base}.ps

psbasemap -X8c -JX6/8 -R0/2000/15/25 -B500f100:"Temperature (K)":/1f0.5:"Excess entropy (J/K/mol)":SWen -O -K >> ${base}.ps

psxy Sxs_0.0001_GPa.dat -J -R -O -K -W1,black >> ${base}.ps
psxy Sxs_10.0_GPa.dat -J -R -O -K -W1,black >> ${base}.ps
psxy Sxs_20.0_GPa.dat -J -R -O -K -W1,black >> ${base}.ps

echo "1900 22 0 GPa" | pstext -J -R -O -K -F+f12,4,black+jRB+a10 >> ${base}.ps
echo "1900 21.1 10 GPa" | pstext -J -R -O -K -F+f12,4,black+jRB+a7 >> ${base}.ps
echo "1900 20 20 GPa" | pstext -J -R -O -F+f12,4,black+jRB+a5 >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi


rm ${base}.ps ${base}.epsi
