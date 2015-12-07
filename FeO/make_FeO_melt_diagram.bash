#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black

if [ ! -d "figures" ]; then
    mkdir figures
fi

base="FeO_melting"

psbasemap -JX12/10 -R0/120/100/250 -B20f5:"Pressure (GPa)":/50f10:"Entropy (J/K/mol)":SWn -P -K > ${base}.ps


awk 'NR>1' output/Smelt_model.dat | psxy -J -R -O -K -W1,blue >> ${base}.ps
awk 'NR>1' output/Smelt_Tallon_prediction.dat | psxy -J -R -O -K -Sc0.1c -Gblue >> ${base}.ps

psbasemap -J -R0/120/8/15 -B20f5:"Pressure (GPa)":/1f0.5:"Volume (cm@+3@+/mol)":E -O -K >> ${base}.ps

awk 'NR>1' output/Vmelt_model.dat | psxy -J -R -O -K -W1,red >> ${base}.ps
awk 'NR>1' output/Vmelt_Tallon_prediction.dat | psxy -J -R -O -K -Sc0.1c -Gred >> ${base}.ps


printf "590 110 \n 610 110" | psxy -J -R -O -K -W1,black >> ${base}.ps
printf "590 100 \n 610 100" | psxy -J -R -O -K -W1,red >> ${base}.ps
echo 600 130 | psxy -J -R -O -K -Sa0.2c -W0.5,blue -Gwhite >> ${base}.ps
echo 600 120 | psxy -J -R -O -K -Sc0.05c -Gblue >> ${base}.ps
echo "650 130 C@-p@- FeO (recommended)" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "650 120 2/1.9374 C@-p@- Fe@-0.9374@-O" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "650 110 C@-p@- (current model)" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "650 100 C@-p@- (Debye model)" | pstext -J -R -O -F+jLM >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

mv ${base}.pdf figures/
evince figures/${base}.pdf
	    
