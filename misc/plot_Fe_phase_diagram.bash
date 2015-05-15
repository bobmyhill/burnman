#!/bin/bash

out="Fe_phase_diagram"


gmt set FONT_ANNOT_PRIMARY 12p,4 \
    FONT_ANNOT_SECONDARY 12p,4 \
    FONT_LABEL 16p,4

psbasemap -JX12/10 -R0/80/300/2700 -B20f10:"Pressure (GPa)":/500f100:"Temperature (K)":SWen -K > ${out}.ps


psxy bcc-fcc.dat -J -R -O -K -W0.5,black >> ${out}.ps
psxy delta-fcc.dat -J -R -O -K -W0.5,black >> ${out}.ps
psxy bcc-hcp.dat -J -R -O -K -W0.5,black >> ${out}.ps
psxy hcp-fcc.dat -J -R -O -K -W0.5,black >> ${out}.ps
awk '{print $1, $2, $3}' fcc_in.dat | psxy -J -R -O -K -Ey0 -Sc0.1c -Gred >> ${out}.ps
awk '{print $1, $2, $3}' fcc_out.dat | psxy -J -R -O -Ey0 -Sc0.1c -Gblue >> ${out}.ps