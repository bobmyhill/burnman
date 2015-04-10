#!/bin/bash

basename=SiO2-H2O
rgn=0/1/600/3000
psbasemap -JX10/10 -R${rgn} -P -Ba0.2f0.1:"X@-SiO2@-":/a500f100:"Temperature (@~\260@~C)":SWen -K > ${basename}.ps
awk '{print $1, $2}' X_SiO2-T_C_1GPa_Hunt_Manning_2012.dat | sample1d -Fa -I0.01 | psxy -J -R -O -K -W1,black,. >> ${basename}.ps
awk '{print $1, $2}' X_SiO2-T_C_2GPa_Hunt_Manning_2012.dat | sample1d -Fa -I0.01 | psxy -J -R -O -K -W1,black,- >> ${basename}.ps

awk '$3=="s" {print $2, $1}' 13GPa_SiO2-H2O.dat | psxy -J -R -O -K -St0.15c -Gred >> ${basename}.ps 
awk '$3=="l" {print $2, $1}' 13GPa_SiO2-H2O.dat | psxy -J -R -O -K -Sc0.15c -W0.5,blue -Gwhite >> ${basename}.ps 

printf "1.0 2700 \n 0.67 1800 \n 0.6 1700 \n 0.5 1600" > tmp

#greenspline tmp -R0.5/1 -I0.001  | awk '{if ($2<2550) {print $0}}' | psxy -J -R -O -K -W1,black >> ${basename}.ps 

last=`greenspline tmp -R0.5/1 -I0.001  | awk '{if ($2<2550) {print $0}}' | tail -1`


awk '{if ($1==">>") {print $0} else {print 1-$1, $2}}' 13GPa_stv_model_fit.dat | psxy -J -R${rgn}  -O -K -W1,grey,- >> ${basename}.ps 
awk '{if ($1==">>") {print $0} else {print 1-$1, $3}}' 13GPa_stv_model_fit.dat | psxy -J -R${rgn}  -O -K -W1,black >> ${basename}.ps 
awk '{if ($1==">>") {print $0} else {print 1-$1, $4}}' 13GPa_stv_model_fit.dat | psxy -J -R${rgn}  -O -K -W1,grey,. >> ${basename}.ps 

echo ${last} > tmp
echo 1.0 2550 >> tmp
psxy tmp -J -R${rgn} -O -K -W1,black >> ${basename}.ps 

echo ${last} > tmp
echo 1.0 2800 >> tmp
psxy tmp -J -R${rgn} -O -K -W1,black >> ${basename}.ps 


echo "0.6 2000 liquid" | pstext -J -R -O -K >> ${basename}.ps
echo "0.7 1500 stv + liquid" | pstext -J -R -O -K >> ${basename}.ps
echo "0.8 2700 coe + liquid" | pstext -J -R -O -K >> ${basename}.ps
printf "0.925 2700 \n 0.98 2650 " | psxy -J -R -O -K -W1,black >> ${basename}.ps

gmtset FONT_ANNOT_PRIMARY 10,4,black
gmt pslegend -R -J -Dx0.35c/-0.20c/10/10/BL -O << EOF >> ${basename}.ps
N 1
S 0.1i t 0.15c  red   -        0.8c stishovite + liquid
S 0.1i c 0.15c white 0.25,blue 0.8c liquid
S 0.1i - 0.6c  -     1,black   0.8c 13 GPa (this study)
S 0.1i - 0.6c  -     1,black,- 0.8c 2 GPa (Hunt and Manning, 2012)
S 0.1i - 0.6c  -     1,black,. 0.8c 1 GPa (Hunt and Manning, 2012)
V 0 1p
EOF

ps2epsi ${basename}.ps
epstopdf ${basename}.epsi

rm ${basename}.ps ${basename}.epsi tmp
evince ${basename}.pdf