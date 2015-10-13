#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="Fe_FeO_solvus"

psbasemap -JX12/8 -R0/1/0/30 -Ba0.2f0.1:"X@-FeO@-":/a5f1:"Pressure (GPa)":SWen -K -P > ${base}.ps

awk '{print $0}' data/Fe_FeO_solvus.dat | sample1d -I0.001 | psxy -J -R -O -K >> ${base}.ps

# Now plot data
	
awk '$1!="%" {if ($3<2473) {print  $5/100, $2, ($5-$6)/100, $5/100, $5/100, ($5+$6)/100, $2-2, $2, $2, $2+2}}' ../data/Fe_O_data/Fe_FeO_solvus.dat | psxy -J -R -O -K -EXY0 -Sc0.15c -Gblue -W0.5,black>> ${base}.ps
awk '$1!="%" {if ($3<2473) {print  $9/100, $2, ($9-$10)/100, $9/100, $9/100, ($9+$10)/100, $2-2, $2, $2, $2+2}}' ../data/Fe_O_data/Fe_FeO_solvus.dat | psxy -J -R -O -K -EXY0 -Sc0.15c -Gred -W0.5,black>> ${base}.ps

awk '$1!="%" {if ($3>2472) {print  $5/100, $2, ($5-$6)/100, $5/100, $5/100, ($5+$6)/100, $2-2, $2, $2, $2+2}}' ../data/Fe_O_data/Fe_FeO_solvus.dat | psxy -J -R -O -K -EXY0 -Sa0.2c -Gblue -W0.5,black>> ${base}.ps
awk '$1!="%" {if ($3>2472) {print  $9/100, $2, ($9-$10)/100, $9/100, $9/100, ($9+$10)/100, $2-2, $2, $2, $2+2}}' ../data/Fe_O_data/Fe_FeO_solvus.dat | psxy -J -R -O -K -EXY0 -Sa0.2c -Gred -W0.5,black>> ${base}.ps



echo "0.5 27 2173 K" | pstext -J -R -O -K -F+jRM >> ${base}.ps
echo "0.5 20 2573 K" | pstext -J -R -O -K -F+jRM >> ${base}.ps

echo "0.4 4" | psxy -J -R -O -K -Sc0.15c -Gred -W0.5,black >> ${base}.ps
echo "0.4 2" | psxy -J -R -O -K -Sa0.2c -Gred -W0.5,black >> ${base}.ps
echo "0.39 4" | psxy -J -R -O -K -Sc0.15c -Gblue -W0.5,black >> ${base}.ps
echo "0.39 2" | psxy -J -R -O -K -Sa0.2c -Gblue -W0.5,black >> ${base}.ps


echo "0.45 4 T < 2473 K" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "0.45 2 T >= 2473 K" | pstext -J -R -O -F+jLM >> ${base}.ps


ps2epsi ${base}.ps
rm ${base}.ps
mv ${base}.epsi ${base}.eps

