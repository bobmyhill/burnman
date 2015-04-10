#!/bin/bash

basename=en-H2O
rgn=0/1/1000/2500
psbasemap -JX10/10 -R${rgn} -P -Ba0.2f0.1:"X@-MgSiO3/2@-":/a500f100:"Temperature (@~\260@~C)":SWen -K > ${basename}.ps

# Davide's results
awk '$3=="l_davide" {print $2*0.02, $1}' 13GPa_en-H2O.dat | psxy -J -R -O -K -Sc0.15c -W0.5,150/150/255 -Gwhite >> ${basename}.ps 
awk '$3=="e_davide" {print $2*0.02, $1}' 13GPa_en-H2O.dat | psxy -J -R -O -K -Sd0.15c -G255/150/150 >> ${basename}.ps 

awk '$3=="l" {print $2*0.02, $1}' 13GPa_en-H2O.dat | psxy -J -R -O -K -Sc0.15c -W0.5,blue -Gwhite >> ${basename}.ps 
awk '$3=="se" {print $2*0.02, $1}' 13GPa_en-H2O.dat | psxy -J -R -O -K -Sd0.1c -Gred >> ${basename}.ps 
awk '$3=="e" {print $2*0.02, $1}' 13GPa_en-H2O.dat | psxy -J -R -O -K -Sd0.15c -Gred >> ${basename}.ps 


# Dry melting from Presnall and Gasparik (2282.5 C at 13 GPa)
printf "1.0 2282.5 \n 0.77 1800 \n 0.60 1600 \n 0.38 1400" > tmp

greenspline tmp -R0.38/1 -I0.001 | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps 

# Melting from Yamada et al., 2004
printf "1.0 2140 \n 0.80 1738 \n 0.6666 1655" > tmp
greenspline tmp -R0.666/0.8 -I0.001 | psxy -J -R${rgn} -O -K -W1,black,- >> ${basename}.ps
#greenspline tmp -R0.8/1.0 -I0.001 | psxy -J -R${rgn} -O -K -W1,black,. >> ${basename}.ps

# More rigid agreement with experiments
# Dry melting from Presnall and Gasparik (2282.5 C at 13 GPa)
#printf "1.0 2282.5 \n 0.75 1700 \n 0.44 1500 \n 0.40 1470 \n 0.35 1400" > tmp

#greenspline tmp -R0.35/1 -I0.001 | psxy -J -R${rgn} -O -K -W1,black >> ${basename}.ps 


echo "0.6 2000 liquid" | pstext -J -R${rgn} -O -K >> ${basename}.ps
echo "0.75 1500 en + liquid" | pstext -J -R -O -K >> ${basename}.ps


gmtset FONT_ANNOT_PRIMARY 10,4,black
gmt pslegend -R -J -Dx0.35c/-0.20c/10/10/BL -O << EOF >> ${basename}.ps
N 1
S 0.1i d 0.15c red   -        0.8c cen + liquid
S 0.1i d 0.15c 255/150/150   -         0.8c cen + liquid (Novella et al., 2015)
S 0.1i d 0.1c  red   -         0.8c (cen) + liquid
S 0.1i c 0.15c white 0.25,blue 0.8c liquid
S 0.1i c 0.15c white 0.25,150/150/255 0.8c liquid (Novella et al., 2015)
S 0.1i - 0.6c  -     1,black,-   0.8c 13 GPa (Yamada et al., 2004)
S 0.1i - 0.6c  -     1,black   0.8c 13 GPa (this study)
V 0 1p
EOF


ps2epsi ${basename}.ps
epstopdf ${basename}.epsi

rm ${basename}.ps ${basename}.epsi tmp
evince ${basename}.pdf