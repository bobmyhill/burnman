#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="Fe_FeO_T_X_eutectic"

psbasemap -JX12/8 -R0/200/0/5000 -Ba50f10:"Pressure (GPa)":/a1000f250:"Temperature (K)":SWen -K -P > ${base}.ps

psxy  data/temperature_Fe_FeO_melt.dat -J -R -O -K >> ${base}.ps

# Now plot data
awk '$1!="%" {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' ../data/Fe_O_data/Fe_FeO_eutectic_temperature.dat | psxy -J -R -O -K -EXY0 -Sc0.15c -Gblack -W0.5,black>> ${base}.ps
awk '$1!="%" {print $2, $3, $2-2, $2, $2, $2+2, $3-$4, $3, $3, $3+$4}' ../data/Fe_O_data/Fe_FeO_eutectic.dat | psxy -J -R -O -K -EXY0 -Sc0.15c -Gblack -W0.5,black >> ${base}.ps

echo "40 4000 melt" | pstext -J -R -O -K -F+jRM >> ${base}.ps
echo "35 2000 FCC" | pstext -J -R -O -K -F+jRM >> ${base}.ps
echo "150 3800 HCP" | pstext -J -R -O -K -F+jRM >> ${base}.ps



printf "5 1200\n10 1300" | psxy -J -R -O -K -W0.5,red,- >> ${base}.ps
printf "5 950\n10 1050" | psxy -J -R -O -K -W0.5,blue  >> ${base}.ps
printf "7.5 750" | psxy -J -R -O -K -Sc0.1c -Gblack -W0.5,black >> ${base}.ps
printf "5 450\n10 550" | psxy -J -R -O -K -W1,black >> ${base}.ps
printf "5 200\n10 300" | psxy -J -R -O -K -W1,black,. >> ${base}.ps

echo "12 1250 FeO melting curve" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "12 1000 Fe phase boundaries" | pstext -J -R -O -K -F+jLM  >> ${base}.ps
echo "12 750 Eutectic data (Seagle et al., 2008)" | pstext -J -R -O -K -F+jLM  >> ${base}.ps
echo "12 500 Eutectic melting curve (This study)" | pstext -J -R -O -K -F+jLM  >> ${base}.ps
echo "12 250 Eutectic melting curve (Frost et al., 2010)" | pstext -J -R -O -K -F+jLM  >> ${base}.ps


# Inset

psbasemap -X7 -Y1 -JX4/3 -R0/200/0.20/0.55 -Ba50f25:"":/a0.1f0.05:"X@-FeO@-":SWen -O -K >> ${base}.ps
awk '{print $1, $2}' data/composition_Fe_FeO_melt.dat | psxy -J -R -O -K -W1,black >> ${base}.ps
awk '$1>30 {print $1, $3}' data/composition_Fe_FeO_melt.dat | psxy  -J -R -O -K -W1,black,. >> ${base}.ps

awk '$1!="%" {print $2, $5/100, $2-2, $2, $2, $2+2, ($5-$6)/100, $5/100, $5/100, ($5+$6)/100}' ../data/Fe_O_data/Fe_FeO_eutectic.dat | psxy -J -R -O -EXY0 -Sc0.15c -Gblack -W0.5,black >> ${base}.ps





ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi

evince ${base}.pdf &
