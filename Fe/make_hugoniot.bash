#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black

if [ ! -d "figures" ]; then
    mkdir figures
fi

base="iron_hugoniot"

psbasemap -JX12/10 -R0/350/4/7 -B100f50:"Pressure (GPa)":/1f0.5:"Volume (cm@+3@+/mol)":SWen -P -K > ${base}.ps

awk 'NR>1 {print $1, $2}' output/hcp_hugoniot_298.15K.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk 'NR>1 {print $1, $2}' output/hcp_hugoniot_500.0K.dat | psxy -J -R -O -K -W0.5,black,- >> ${base}.ps


awk 'NR>1 {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' output/hcp_hugoniot_Brown_et_al_2000.dat | psxy -J -R -O -K -EXY0 -Sc0.1c -Gred -W0.5,red >> ${base}.ps

pslegend -J -R -O -Dx7.5c/0c/0c/9c/BR <<EOF >> ${base}.ps
S 0.1i c 0.1c red 0.5,red 0.3i Brown et al. (2000)
EOF

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

mv ${base}.pdf figures/
evince figures/${base}.pdf
	    
