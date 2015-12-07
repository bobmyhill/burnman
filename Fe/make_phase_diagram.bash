#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black

if [ ! -d "figures" ]; then
    mkdir figures
fi

base="iron_phase_diagram"

psbasemap -JX12/10 -R0/350/500/6500 -B100f50:"Pressure (GPa)":/1000f500:"Temperature (K)":SWen -P -K > ${base}.ps

for file in output/*
do
    awk 'NR>1' ${file} | psxy -J -R -O -K -W1,black >> ${base}.ps
done

echo 0 0 | psxy -J -R -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

mv ${base}.pdf figures/
evince figures/${base}.pdf
	    
