#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="pyrope_grossular_P_V"

psbasemap -JX12/8 -R0/10/106/126 -Ba1f0.5:"Pressure (GPa)":/a2f1:"Volume (cm@+3@+/mol)":SWen -K -P > ${base}.ps

psxy -J -R -O -K data/pyrope_grossular_P_V.dat >> ${base}.ps

awk '$3==298.15 {print $2, $4, $2-0.1, $2, $2, $2+0.1, $4-$5, $4, $4, $4+$5}' ../data/py_gr_PVT_data/Du_et_al_2015_py_gr_PVT.dat | psxy -J -R -O -K -EXY0 -Sc0.1c -Ggrey -W0.5,black >> ${base}.ps

echo "9.4 108.3 Py" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "9.3 111 Py@-20@-" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "9.3 113.4 Py@-40@-" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "9.3 115.8 Py@-60@-" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "9.3 118 Py@-80@-" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "9.4 120 Gr" | pstext -J -R -O -F+jLM >> ${base}.ps



ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi

evince ${base}.pdf &

