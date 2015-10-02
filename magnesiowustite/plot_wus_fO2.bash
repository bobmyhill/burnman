#!/bin/bash

gmt set FONT_ANNOT_PRIMARY 12p,4 \
    FONT_ANNOT_SECONDARY 14p,4 \
    FONT_LABEL 16p,4 
out="wus_fO2"

psbasemap -JX12/10 -R0.0/0.20/-20/-5 -B0.05f0.01:"(1-y) in Fe@-y@-O":/2f1:"log fO@-2@-":SWen -K > ${out}.ps

for file in *wus_fO2.dat
do
    awk '{print $1, $2}' ${file} | psxy -J -R -O -K -W0.5,black >> ${out}.ps
done

awk '{print $3/100, $2}' Bransky_Hed_1968.dat | psxy -J -R -O -K -Sc0.1c -Gred >> ${out}.ps
awk '{print $3/100, $2}' Giddings_Gordon_1973.dat | psxy -J -R -O -K -Sc0.1c -Ggreen >> ${out}.ps


pslegend -J -R -Dx0c/8c/0c/0c/BL \
-C0.1i/0.1i -L1.2 -O << EOF >> ${out}.ps
N 1
V 0 1p
S 0.1i c 0.1c green - 0.3i Bransky and Hed, 1968
S 0.1i c 0.1c red - 0.3i Giddings and Gordon, 1973
V 0 1p
EOF