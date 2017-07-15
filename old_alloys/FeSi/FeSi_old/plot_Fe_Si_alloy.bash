#!/bin/bash



gmt set FONT_ANNOT_PRIMARY 12p,4 \
    FONT_ANNOT_SECONDARY 12p,4 \
    FONT_LABEL 16p,4


out="bcc_Si_activities"

psbasemap -JX12/10 -R0/0.1/-4/-2.5 -B0.02f0.01:"X@-Si@-":/0.5f0.1:"log @~\g@~@-Si@-":SWen -K > ${out}.ps

awk '{print $2, $3}' data/Sakao_Elliott_1975_activity_coeff_Si_bcc.dat | psxy -J -R -O -K -Sc0.1c >> ${out}.ps
psxy 1373.15_Si_activities.dat -J -R -O -K -W0.5,black >> ${out}.ps
psxy 1473.15_Si_activities.dat -J -R -O -K -W0.5,red >> ${out}.ps
psxy 1573.15_Si_activities.dat -J -R -O -K -W0.5,blue >> ${out}.ps
psxy 1623.15_Si_activities.dat -J -R -O -K -W0.5,purple >> ${out}.ps
#awk '{print $1, $2, $3}' fcc_in.dat | psxy -J -R -O -K -Ey0 -Sc0.1c -Gred >> ${out}.ps
#awk '{print $1, $2, $3}' fcc_out.dat | psxy -J -R -O -Ey0 -Sc0.1c -Gblue >> ${out}.ps

pslegend -J -R -O -Dx0/7c/0/0/BL -C0.1i/0.1i -L1.2 << EOF >> ${out}.ps
N 1
S 0.1i c 0.1c - 0.25p 0.3i Sakao and Elliott, 1975
S 0.1i - 0.15i - 0.5,black 0.3i 1373.15 K
S 0.1i - 0.15i - 0.5,red 0.3i 1473.15 K
S 0.1i - 0.15i - 0.5,blue 0.3i 1573.15 K
S 0.1i - 0.15i - 0.5,purple 0.3i 1623.15 K
EOF
