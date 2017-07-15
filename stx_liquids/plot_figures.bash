#!/bin/bash

gmtset  FONT_ANNOT_PRIMARY 10p,4,black \
	FONT_LABEL 14p,4,black \
        MAP_GRID_CROSS_SIZE_PRIMARY 0.1i \
	MAP_ANNOT_OFFSET_PRIMARY 0.2c

# FORSTERITE
base="forsterite_melting"

echo "0 0" | psxy -JX12/8 -R0/20/1800/2400 -K -P > ${base}.ps
psxy forsterite_melting.PT -J -R -O -K >> ${base}.ps
awk '$5=="(Davis" {print $1, $2}' data/Presnall_Walter_1993_fo_melting.dat | psxy -J -R -O -K -Sa0.1c -W0.5,blue >> ${base}.ps  
awk '{if ($4=="fo") {if ($5!="(Davis") {print $1, $2}}}' data/Presnall_Walter_1993_fo_melting.dat | psxy -J -R -O -K -Sa0.2c -W0.5,blue >> ${base}.ps  
awk '$4=="fo_periclase" {print $1, $2}' data/Presnall_Walter_1993_fo_melting.dat | psxy -J -R -O -K -Sc0.1c -W0.5,blue >> ${base}.ps  
awk '$4=="fo_anhB" {print $1, $2}' data/Presnall_Walter_1993_fo_melting.dat | psxy -J -R -O -K -Si0.1c -W0.5,blue >> ${base}.ps  

pslegend -J -R -Dx5/0.4/10/2/BL -O -K << EOF >> ${base}.ps
N 1
S 0.3c a 0.1c  -      0.5,blue   0.7c fo (Davis and England, 1964)
S 0.3c a 0.2c  -      0.5,blue   0.7c fo (Presnall and Walter, 1993)
S 0.3c c 0.1c  -      0.5,blue   0.7c fo + per (Presnall and Walter, 1993)
S 0.3c i 0.1c  -      0.5,blue   0.7c fo + anhB (Presnall and Walter, 1993)
S 0.3c - 0.3c  -      1,black    0.7c Model liquidus
EOF

psbasemap -J -R -B2f1:"P (GPa)":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf

# ENSTATITE
base="enstatite_melting"

echo "0 0" | psxy -JX12/8 -R0/20/1500/2500 -K -P > ${base}.ps
psxy enstatite_melting.PT -J -R -O -K >> ${base}.ps
awk '$4=="(Boyd," {print $1, $2}' data/Presnall_Gasparik_1990_en_melting.dat | psxy -J -R -O -K -Sa0.1c -W0.5,blue >> ${base}.ps  
awk '{if ($3=="oen") {if ($4!="(Boyd,") {print $1, $2}}}' data/Presnall_Gasparik_1990_en_melting.dat | psxy -J -R -O -K -Sa0.2c -W0.5,blue >> ${base}.ps  
awk '$3=="cen" {print $1, $2}' data/Presnall_Gasparik_1990_en_melting.dat | psxy -J -R -O -K -Sc0.1c -W0.5,blue >> ${base}.ps  

pslegend -J -R -Dx5/0.3/10/2/BL -O -K << EOF >> ${base}.ps
N 1
S 0.3c a 0.1c  -      0.5,blue   0.7c oen (Boyd, 1964)
S 0.3c a 0.2c  -      0.5,blue   0.7c oen (Presnall and Gasparik, 1990)
S 0.3c c 0.1c  -      0.5,blue   0.7c cen (Presnall and Gasparik, 1990)
S 0.3c - 0.5c  -      1,blue,-    0.7c Liquidus (Kato and Kumazawa, 1986)
S 0.3c - 0.5c  -      1,black    0.7c Model liquidus
EOF

psbasemap -J -R -B2f1:"P (GPa)":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf


####

base="MS_melting_13GPa"
echo "0 0" | psxy -JX12/8 -R0/1/500/4000 -K -P > ${base}.ps
psxy MS_melting_13GPa.xT -J -R -O -K >> ${base}.ps
echo "0.1666 1200 per + fo" | pstext -J -R -O -K >> ${base}.ps
echo "0.4166 1200 fo + cen" | pstext -J -R -O -K >> ${base}.ps
echo "0.75 1200 cen + stv" | pstext -J -R -O -K >> ${base}.ps

echo "0.10 2500 per + L" | pstext -J -R -O -K >> ${base}.ps
echo "0.85 2100 stv + L" | pstext -J -R -O -K >> ${base}.ps
echo "0.9 2600 coe + L" | pstext -J -R -O -K >> ${base}.ps

echo "0.5 3000 L" | pstext -J -R -O -K >> ${base}.ps
psbasemap -J -R -B0.2f0.1:"X@-SiO2@-":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf


