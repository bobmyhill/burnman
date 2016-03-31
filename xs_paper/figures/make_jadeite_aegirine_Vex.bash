#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="jadeite_aegirine_Vex"

psbasemap -JX12/8 -R0/25/-0.2/-0.0 -Ba5f1:"Pressure (GPa)":/a0.05f0.01:"Excess volume (cm@+3@+/mol)":SWen -K -P > ${base}.ps

awk '{print $1, $2, $3}' data/jadeite_aegirine_Vex.dat | psxy -J -R -O -K >> ${base}.ps
awk '{print $1, $2, $3}' data/jadeite_aegirine_Vex_obs.dat | psxy -J -R -O -K -Sc0.1c >> ${base}.ps

echo "24 -0.19 298.15 K" | pstext -J -R -O -K -F+jRM >> ${base}.ps
echo "24 -0.08 Jd@-50@-Aeg@-50@-" | pstext -J -R -O -K -F+jRM+fgrey >> ${base}.ps
echo "24 -0.11 Jd@-35@-Aeg@-65@-" | pstext -J -R -O -K -F+jRM >> ${base}.ps
echo "24 -0.03 Jd@-74@-Aeg@-26@-" | pstext -J -R -O -F+jRM >> ${base}.ps


ps2epsi ${base}.ps
rm ${base}.ps
mv ${base}.epsi ${base}.eps
