#!/bin/bash

gmt set FONT_ANNOT_PRIMARY 12p,4 \
    FONT_ANNOT_SECONDARY 14p,4 \
    FONT_LABEL 16p,4 
out="wus_phase_diagram"

psbasemap -JX12/10 -R0.0/0.25/700/1700 -B0.1f0.05:"(1-y) in Fe@-y@-O":/200f100:"Temperature (K)":SWen -K > ${out}.ps


awk '{print 2 - 1/$1, $2}' Lykasov_Fe-FeO.dat | psxy -J -R -O -K -Sc0.1c -Gred >> ${out}.ps
awk '{print 2 - 1/$1, $2}' Vallet_FeOW1-Fe3O4.dat | psxy -J -R -O -K -Sc0.1c -Ggreen >> ${out}.ps
awk '{print 2 - 1/$1, $2}' Darken_Fe-FeO.dat | psxy -J -R -O -K -Sc0.1c -Gblue >> ${out}.ps
awk '{print 2 - 1/$1, $2}' Darken_FeOW1-Fe3O4.dat | psxy -J -R -O -K -Sc0.1c -Gblue >> ${out}.ps
awk '{print 2 - 1/$1, $2}' Asao_et_al_1970.dat | psxy -J -R -O -K -Sc0.1c -Gorange >> ${out}.ps
awk '{print 2 - 1/$1, $2}' Barbi_1964.dat | psxy -J -R -O -K -Sc0.1c -Gpurple >> ${out}.ps
awk '{print 2 - 1/$1, $2}' Barbera_et_al_1980.dat | psxy -J -R -O -K -Sc0.1c -Gnavy >> ${out}.ps

psxy bcc_wus.dat -J -R -O -K -W0.5,black >> ${out}.ps
psxy fcc_wus.dat -J -R -O -K -W0.5,black,- >> ${out}.ps
psxy mt_wus.dat -J -R -O -K -W0.5,black >> ${out}.ps
psxy T_eqm_iron_mt_wus.dat -J -R -O -K -W0.5,black >> ${out}.ps


pslegend -J -R -Dx6c/1.5c/0c/0c/BL \
-C0.1i/0.1i -L1.2 -O << EOF >> ${out}.ps
N 1
V 0 1p
S 0.1i c 0.1c blue - 0.3i Darken and Gurry, 1946
S 0.1i c 0.1c navy - 0.3i Barbera et al., 1960
S 0.1i c 0.1c purple - 0.3i Barbi, 1964
S 0.1i c 0.1c green - 0.3i Vallet and Raccah, 1965
S 0.1i c 0.1c red - 0.3i Lykasov et al., 1969
S 0.1i c 0.1c orange - 0.3i Asao et al., 1970
V 0 1p
EOF