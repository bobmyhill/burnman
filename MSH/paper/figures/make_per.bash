#!/bin/bash

basename=per-H2O
rgn=0/1/800/5200
psbasemap -JX10/10 -R${rgn} -P -Ba0.2f0.1:"X@-MgO@-":/a500f100:"Temperature (@~\260@~C)":SWen -K > ${basename}.ps

awk '$3=="l" {print 1-$2*0.01, $1}' 13GPa_per-H2O.dat | psxy -J -R -O -K -Sc0.15c -W0.5,blue -Gwhite >> ${basename}.ps 
awk '$3=="sp" {print 1-$2*0.01, $1}' 13GPa_per-H2O.dat | psxy -J -R -O -K -Sh0.10c -Gred >> ${basename}.ps 
awk '$3=="p" {print 1-$2*0.01, $1}' 13GPa_per-H2O.dat | psxy -J -R -O -K -Sh0.15c -Gred >> ${basename}.ps 
awk '$3=="b" {print 1-$2*0.01, $1}' 13GPa_per-H2O.dat | psxy -J -R -O -K -Ss0.15c -Gred >> ${basename}.ps 

# Alfe 2005 melting point of MgO at 13 GPa is 4313 K, i.e. 4040 C
# Zhang and Fei 2008 melting point of MgO at 13 GPa is 5373 K, i.e. 5100 C
printf "1.0 5100 \n 0.48 1900 \n 0.345 1200" > tmp

T_br_per=1210
start=`greenspline tmp -R0.34/1 -I0.001 | sample1d -T1 -S${T_br_per}/1310 -I1 | head -1`

startx=`greenspline tmp -R0.34/1 -I0.001 | sample1d -T1 -S${T_br_per}/1310 -I1 | head -1 | awk '{print $1}'`

echo ${start}
 
#greenspline tmp -R${startx}/1 -I0.001 | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps 

awk '{if ($1==">>") {print $0} else {print 1-$1, $2-273.15}}' 13GPa_per_model_fit_SS1985.dat | psxy -J -R${rgn}  -O -K -W1,black >> ${basename}.ps 

awk '{if ($1==">>") {print $0} else {print 1-$1, $2-273.15}}' 13GPa_per_model_fit.dat | psxy -J -R${rgn}  -O -K -W1,black >> ${basename}.ps 

#printf "1.0 1210 \n %s %s" ${start} | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps 

#printf "0.30 1150 \n %s %s" ${start} | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps 

echo "0.2 2500 liquid" | pstext -J -R${rgn} -O -K >> ${basename}.ps
echo "0.75 2000 per + liquid" | pstext -J -R -O -K >> ${basename}.ps
echo "0.65 1000 br + liquid" | pstext -J -R -O -K >> ${basename}.ps

gmtset FONT_ANNOT_PRIMARY 9,4,black
gmt pslegend -R -J -Dx0.35c/-0.20c/10/10/BL -O << EOF >> ${basename}.ps
N 1
S 0.1i s 0.15c red   -        0.8c br + liquid
S 0.1i h 0.15c  red   -       0.8c per + liquid
S 0.1i h 0.10c  red   -       0.8c (per) + liquid
S 0.1i c 0.15c white 0.25,blue 0.8c liquid
S 0.1i - 0.6c  -     1,grey   0.8c 13 GPa (K=0, K=K(T), K=inf)
S 0.1i - 0.6c  -     1,black   0.8c 13 GPa (K=K(T), W!=0)
S 0.1i - 0.6c  -     1,black,.   0.8c 13 GPa (Regular solution)
V 0 1p
EOF

ps2epsi ${basename}.ps
epstopdf ${basename}.epsi

rm ${basename}.ps ${basename}.epsi tmp
evince ${basename}.pdf