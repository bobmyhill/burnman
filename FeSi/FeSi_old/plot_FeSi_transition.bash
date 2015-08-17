#!/bin/bash



gmt set FONT_ANNOT_PRIMARY 12p,4 \
    FONT_ANNOT_SECONDARY 12p,4 \
    FONT_LABEL 16p,4


out="B20_B2_transition"

psbasemap -JX12/10 -R0/50/1500/2500 -B10f5:"Pressure (GPa)":/500f100:"Temperature (K)":SWen -K > ${out}.ps

awk '{print $1, $2}' B2_B20.dat | psxy -J -R -O -K -W0.5,black >> ${out}.ps
awk '$3=="B20" {print $1, $2}' data/FeSi_Lord_2010.dat | psxy -J -R -O -K -Sc0.1c -Gred >> ${out}.ps
awk '$3=="B20_B2" {print $1, $2}' data/FeSi_Lord_2010.dat | psxy -J -R -O -K -Sc0.1c -Gpurple >> ${out}.ps
awk '$3=="B2" {print $1, $2}' data/FeSi_Lord_2010.dat | psxy -J -R -O -Sc0.1c -Gblue >> ${out}.ps
#awk '{print $1, $2, $3}' fcc_in.dat | psxy -J -R -O -K -Ey0 -Sc0.1c -Gred >> ${out}.ps
#awk '{print $1, $2, $3}' fcc_out.dat | psxy -J -R -O -Ey0 -Sc0.1c -Gblue >> ${out}.ps

