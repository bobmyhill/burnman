#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black

if [ ! -d "figures" ]; then
    mkdir figures
fi

base="FeO_melting_curve"

psbasemap -JX12/10 -R0/120/1000/6000 -B20f5:"Pressure (GPa)":/1000f200:"Temperature (K)":SWen -P -K > ${base}.ps

awk 'NR>1' output/Tmelt_model.dat | psxy -J -R -O -K -W1,red >> ${base}.ps

awk '$4==1 {print $1, $2, $2-$3, $2, $2, $2+$3}' data/Knittle_Jeanloz_1991_FeO_melting.dat | psxy -J -R -O -K -St0.2c -Gred -W0.5,red -EY0 >> ${base}.ps
awk '$4==2 {print $1, $2, $2-$3, $2, $2, $2+$3}' data/Knittle_Jeanloz_1991_FeO_melting.dat | psxy -J -R -O -K -Si0.2c -Gwhite -W0.5,red -EY0 >> ${base}.ps
awk '$4==3 {print $1, $2, $2-$3, $2, $2, $2+$3}' data/Knittle_Jeanloz_1991_FeO_melting.dat | psxy -J -R -O -K -Sc0.1c -Gblack -EY0 >> ${base}.ps

awk 'NR>1 {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/Fe0.94O_melting_curve_Fischer_Campbell_2010.dat | psxy -J -R -O -K -Sc0.1c -Gblue -EXY0 >> ${base}.ps

awk 'NR>1 {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/FeO_melting_Ringwood_Hibberson_1990.dat | psxy -J -R -O -K -Sc0.1c -Gorange -EXY0 >> ${base}.ps

awk 'NR>1 {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/FeO_melting_Zhang_Fei_2008.dat | psxy -J -R -O -K -Sc0.1c -Gpurple -EXY0 >> ${base}.ps


awk '$5==1 {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/FeO_melting_Seagle_et_al_2008.dat | psxy -J -R -O -K -St0.2c -Ggreen -W0.5,green -EXY0 >> ${base}.ps
awk '$5==2 {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/FeO_melting_Seagle_et_al_2008.dat | psxy -J -R -O -K -Si0.2c -Gwhite -W0.5,green -EXY0 >> ${base}.ps

pslegend -J -R -O -K -Dx0c/5c/5c/5c/BL <<EOF >> ${base}.ps
S 0.1i i 0.2c white 0.5,red 0.3i Knittle and Jeanloz (1991; liquid)
S 0.1i t 0.2c red 0.5,red 0.3i Knittle and Jeanloz (1991; solid)
S 0.1i i 0.2c white 0.5,green 0.3i Seagle et al. (2008; liquid)
S 0.1i t 0.2c green 0.5,green 0.3i Seagle et al. (2008; solid)
EOF

pslegend -J -R -O -Dx10c/0c/4c/0c/BR <<EOF >> ${base}.ps
S 0.1i c 0.1c black - 0.3i Lindsley (1966)
S 0.1i c 0.1c purple - 0.3i Zhang and Fei (2008)
S 0.1i c 0.1c orange - 0.3i Ringwood and Hibberson (1990)
S 0.1i c 0.1c blue - 0.3i Fischer and Campbell (2010)
EOF

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

mv ${base}.pdf figures/
evince figures/${base}.pdf
	    
