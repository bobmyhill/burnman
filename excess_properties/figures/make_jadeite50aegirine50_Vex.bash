#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="jadeite50aegirine50_Vex"

psbasemap -JX12/8 -R0/25/-0.2/-0.05 -Ba5f1:"Pressure (GPa)":/a0.05f0.01:"Excess volume (cm@+3@+/mol)":SWen -K -P > ${base}.ps

psxy -J -R -O -K data/jadeite50aegirine50_Vex.dat >> ${base}.ps

echo "24 -0.18 298.15 K" | pstext -J -R -O -K -F+jRM >> ${base}.ps
echo "24 -0.19 Jd@-50@-Aeg@-50@-" | pstext -J -R -O -F+jRM >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi

evince ${base}.pdf &
