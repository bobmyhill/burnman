#!/bin/bash

basename=fo-H2O
rgn=0/1/1000/2500
psbasemap -JX10/10 -R${rgn} -P -Ba0.2f0.1:"X@-Mg2SiO4/3@-":/a500f100:"Temperature (@~\260@~C)":SWen -K > ${basename}.ps

awk '$4=="l_davide" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.15c -W0.5,150/150/255 -Gwhite >> ${basename}.ps 
awk '$4=="f_davide" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Sa0.20c -G255/150/150 >> ${basename}.ps 

awk '$4=="l" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.15c -W0.5,blue -Gwhite >> ${basename}.ps 
awk '$4=="c" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Si0.15c -Gred >> ${basename}.ps 
awk '$4=="e" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Sd0.15c -Gred >> ${basename}.ps 
awk '$4=="se" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Sd0.10c -Gred >> ${basename}.ps 
awk '$4=="f" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Sa0.2c -Gred >> ${basename}.ps 
awk '$4=="sf" {print $2*0.03, $1}' 13GPa_fo-H2O.dat | psxy -J -R -O -K -Sa0.1c -Gred >> ${basename}.ps

printf "1.0 2301 \n 0.44 1400 \n 0.36 1300" > tmp 

# Presnall and Walter (1993) say 2290 (from their graph) for fo breakdown at 13 GPa
# Extrapolating the metastable fo curve gives 2301 C 
# Melting just congruent at 10.1 GPa, 2250 C; 33.333 mol% SiO2
# Melting at 16 GPa, 2314 C at 41.8 mol% SiO2 (Liebske and Frost, 2012)

# Linear interpolation gives composition of melt during stable fo breakdown at 13GPa, 2290 C) of ~37.5 mol % SiO2
point1="1 2290" 

# If per melting is at 4040 C (Alfe, 2005) at zero, then linear interpolation gives melting T of residual per at fo composition of 
# 0.11*4040 + 0.89*2290 = 2484 C


# If per melting is at 5100 C at zero (Zhang and Fei, 2008), then linear interpolation gives melting T of residual per at fo composition of 
point2=`echo 0.11 5100 2290 | awk '{print 1, $1*$2 + (1-$1)*$3}'`

# Finally, estimate fo-per peritectic crosses fo-H2O binary at 11 mol % H2O.
point3=`greenspline tmp -R0.88/0.89 -I0.001 | tail -1`

greenspline tmp -R0.345/0.89 -I0.001 | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps

greenspline tmp -R0.89/1 -I0.001 | psxy -J -R${rgn} -O -K -W1,grey,- >> ${basename}.ps

printf "%s %s \n %s %s " $point2 $point3 | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps

printf "%s %s \n %s %s " $point1 $point3 | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps


printf "0.87 2300 \n 0.94 2240" | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps

echo "0.4 1750 liquid" | pstext -J -R${rgn} -O -K >> ${basename}.ps
echo "0.8 1500 fo + liquid" | pstext -J -R -O -K >> ${basename}.ps
echo "0.75 2300 per + liquid" | pstext -J -R -O -K >> ${basename}.ps

gmtset FONT_ANNOT_PRIMARY 10,4,black
gmt pslegend -R -J -Dx0.35c/-0.20c/10/10/BL -O << EOF >> ${basename}.ps
N 1
S 0.1i i 0.15c red    -         0.8c chond + liquid
S 0.1i d 0.15c red    -         0.8c cen + liquid
S 0.1i d 0.10c red    -         0.8c (cen) + liquid
S 0.1i a 0.20c red    -         0.8c fo + liquid
S 0.1i a 0.20c 255/150/150    -         0.8c fo + liquid (Novella et al., 2015)
S 0.1i a 0.1c  red   -          0.8c (fo) + liquid
S 0.1i c 0.15c white  0.25,blue 0.8c liquid
S 0.1i c 0.15c white  0.25,150/150/255 0.8c liquid (Novella et al., 2015)
S 0.1i - 0.6c  -      1,black   0.8c 13 GPa (this study)
V 0 1p
EOF

ps2epsi ${basename}.ps
epstopdf ${basename}.epsi

rm ${basename}.ps ${basename}.epsi tmp
evince ${basename}.pdf